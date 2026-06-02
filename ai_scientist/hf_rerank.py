"""
HuggingFace reranker backend — Qwen/Qwen3-Reranker-4B.

Two-stage citation retrieval:
  Stage 1 (embed): Qwen3-VL-Embedding-2B → cosine similarity → top-N candidates
  Stage 2 (rerank): Qwen3-Reranker-4B   → cross-encoder score → top-k final

Qwen3-Reranker is a causal LM that scores (query, doc) pairs by predicting
the probability of the token "yes" at the final position.

Usage:
    from ai_scientist.hf_rerank import get_reranker

    reranker = get_reranker()
    scores = reranker.rerank("domain shift in hydrology", ["doc text 1", ...])
    ranked = reranker.rerank_with_indices(query, docs)  # [(idx, score), ...]
"""

import os
from typing import Optional

MODEL_ID = os.environ.get("CITATION_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")

_SYSTEM_MSG = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCTION = (
    "Given a scientific research query, retrieve relevant document passages "
    "that answer the query or provide supporting evidence."
)


class Qwen3Reranker:
    """
    Cross-encoder reranker using Qwen3-Reranker-4B.

    Scores (query, document) pairs by the softmax probability of the "yes"
    token at the last autoregressive position.
    """

    _instance: Optional["Qwen3Reranker"] = None

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.torch = torch

        print(f"[hf_rerank] Loading {model_id} on {self.device} ({dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.device,
            cache_dir=cache_dir,
        )
        self.model.eval()

        self.token_true_id  = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # Fixed prefix/suffix wrapping the (instruction, query, doc) body
        self._prefix = (
            f"<|im_start|>system\n{_SYSTEM_MSG}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        # Empty <think> block suppresses chain-of-thought for speed
        self._suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        print(
            f"[hf_rerank] Reranker ready "
            f"(yes_id={self.token_true_id}, no_id={self.token_false_id})"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _format(self, query: str, doc: str, instruction: str) -> str:
        body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return self._prefix + body + self._suffix

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        docs: list[str],
        instruction: str = _INSTRUCTION,
        batch_size: int = 2,
        max_length: int = 4096,
    ) -> list[float]:
        """
        Score each (query, doc) pair.

        Args:
            query:       Search query string.
            docs:        List of document / passage strings to score.
            instruction: Task description prepended to each pair.
            batch_size:  Number of pairs per forward pass (reduce if OOM).
            max_length:  Maximum input tokens (truncates docs that exceed this).

        Returns:
            List of floats in [0, 1] — one per doc, same order as input.
        """
        torch = self.torch
        scores: list[float] = []

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            formatted = [self._format(query, d, instruction) for d in batch]

            inputs = self.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Score = P(yes | query, doc) via softmax over {yes, no}
            logits = outputs.logits[:, -1, :]               # (B, vocab)
            true_log  = logits[:, self.token_true_id]
            false_log = logits[:, self.token_false_id]
            batch_scores = torch.nn.functional.softmax(
                torch.stack([false_log, true_log], dim=1), dim=1
            )[:, 1].tolist()
            scores.extend(batch_scores)

        return scores

    def rerank_with_indices(
        self,
        query: str,
        docs: list[str],
        **kwargs,
    ) -> list[tuple[int, float]]:
        """
        Score docs and return (original_index, score) sorted by score descending.
        """
        scores = self.rerank(query, docs, **kwargs)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


def get_reranker(
    model_id: str = MODEL_ID,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Qwen3Reranker:
    """Return a cached Qwen3Reranker (loads model once per process)."""
    if Qwen3Reranker._instance is None:
        Qwen3Reranker._instance = Qwen3Reranker(model_id, device, cache_dir)
    return Qwen3Reranker._instance
