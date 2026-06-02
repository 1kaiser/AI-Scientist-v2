"""
HuggingFace embedding backend for Qwen/Qwen3-VL-Embedding-2B.

Supports both CPU and GPU inference with automatic device selection.
Provides text and image embeddings (2048-dim, configurable 64-2048).

Usage:
    from ai_scientist.hf_embed import Qwen3VLEmbedder

    embedder = Qwen3VLEmbedder()                  # auto GPU/CPU
    text_emb  = embedder.embed_texts(["figure shows loss curve"])
    image_emb = embedder.embed_images(["figures/01_loss.png"])
    mixed_emb = embedder.embed_mixed(
        texts=["training loss", None],
        images=[None, "figures/02_mAP.png"],
    )

Designed for Figure RAG — retrieves relevant experiment plots for Stage B
by matching section descriptions to actual plot files on disk.
"""

import os
import os.path as osp
from typing import Optional
import numpy as np

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_DIM = 2048      # model native dim; can reduce to 64-2048
DEFAULT_INSTR = "Retrieve relevant figures for this section of an academic paper."


class Qwen3VLEmbedder:
    """
    Singleton-style embedder for Qwen3-VL-Embedding-2B.

    Args:
        device:     "cuda", "cpu", or None (auto-detect).
        output_dim: Embedding dimension 64-2048 (default 2048).
        cache_dir:  HuggingFace cache directory (default: ~/.cache/huggingface).
    """

    _instance: Optional["Qwen3VLEmbedder"] = None

    def __init__(
        self,
        device: Optional[str] = None,
        output_dim: int = DEFAULT_DIM,
        cache_dir: Optional[str] = None,
    ):
        import torch
        import torch.nn.functional as F
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.output_dim = min(max(output_dim, 64), DEFAULT_DIM)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.F = F
        self.torch = torch

        print(f"[hf_embed] Loading {MODEL_ID} on {self.device} ({self.dtype}) ...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.model.eval()
        print(f"[hf_embed] Model ready — output_dim={self.output_dim}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _pool_and_norm(self, hidden_states, attention_mask=None) -> np.ndarray:
        """Last-token pooling + L2 normalisation → numpy array."""
        torch = self.torch
        if attention_mask is not None:
            # Find index of last non-pad token per sample
            seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
            batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
            emb = hidden_states[batch, seq_lens]               # (B, H)
        else:
            emb = hidden_states[:, -1, :]                      # (B, H)

        # Optionally truncate to requested output_dim
        if self.output_dim < DEFAULT_DIM:
            emb = emb[:, : self.output_dim]

        emb = self.F.normalize(emb.float(), p=2, dim=-1)
        return emb.cpu().numpy()

    def _format_text_messages(self, texts: list[str], instruction: str) -> list[dict]:
        """Wrap texts in the VL chat format expected by the processor."""
        messages = []
        for text in texts:
            messages.append([
                {"role": "user", "content": [
                    {"type": "text", "text": f"{instruction}\n{text}"},
                ]}
            ])
        return messages

    def _format_image_messages(
        self, image_paths: list[str], instruction: str
    ) -> list[dict]:
        """Wrap image paths in VL chat format."""
        from PIL import Image

        messages = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            messages.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ]}
            ])
        return messages

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed_texts(
        self, texts: list[str], instruction: str = DEFAULT_INSTR, batch_size: int = 4
    ) -> np.ndarray:
        """
        Embed a list of strings.

        Returns: float32 numpy array of shape (len(texts), output_dim).
        """
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            msgs = self._format_text_messages(batch, instruction)
            texts_fmt = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in msgs
            ]
            inputs = self.processor(
                text=texts_fmt,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]  # last layer
            all_emb.append(self._pool_and_norm(hs, inputs.get("attention_mask")))
        return np.vstack(all_emb)

    def embed_images(
        self, image_paths: list[str], instruction: str = DEFAULT_INSTR
    ) -> np.ndarray:
        """
        Embed a list of image file paths (PNG, JPG).

        Returns: float32 numpy array of shape (len(image_paths), output_dim).
        """
        all_emb = []
        for path in image_paths:
            msgs = self._format_image_messages([path], instruction)
            text_fmt = self.processor.apply_chat_template(
                msgs[0], tokenize=False, add_generation_prompt=False
            )
            from PIL import Image
            img = Image.open(path).convert("RGB")
            inputs = self.processor(
                text=[text_fmt],
                images=[img],
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]
            all_emb.append(self._pool_and_norm(hs, inputs.get("attention_mask")))
        return np.vstack(all_emb)

    def embed_mixed(
        self,
        texts: list[Optional[str]],
        images: list[Optional[str]],
        instruction: str = DEFAULT_INSTR,
    ) -> np.ndarray:
        """
        Embed paired (text, image) inputs — either can be None.

        len(texts) must equal len(images).
        Returns: float32 numpy array of shape (N, output_dim).
        """
        assert len(texts) == len(images)
        all_emb = []
        for txt, img_path in zip(texts, images):
            if txt and img_path:
                # Both modalities
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"{instruction}\n{txt}"},
                ]}]
                text_fmt = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[text_fmt], images=[img], return_tensors="pt"
                ).to(self.device)
            elif img_path:
                all_emb.append(self.embed_images([img_path], instruction))
                continue
            else:
                all_emb.append(self.embed_texts([txt or ""], instruction))
                continue

            with self.torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]
            all_emb.append(self._pool_and_norm(hs, inputs.get("attention_mask")))
        return np.vstack(all_emb)

    # ------------------------------------------------------------------ #
    # LightRAG-compatible async embedding function
    # ------------------------------------------------------------------ #

    async def as_embedding_func(self, texts: list[str]) -> np.ndarray:
        """Drop-in replacement for LightRAG's EmbeddingFunc callable."""
        return self.embed_texts(texts)


def get_embedder(
    device: Optional[str] = None,
    output_dim: int = DEFAULT_DIM,
    cache_dir: Optional[str] = None,
) -> Qwen3VLEmbedder:
    """Return a cached Qwen3VLEmbedder (loads model once per process)."""
    if Qwen3VLEmbedder._instance is None:
        Qwen3VLEmbedder._instance = Qwen3VLEmbedder(device, output_dim, cache_dir)
    return Qwen3VLEmbedder._instance
