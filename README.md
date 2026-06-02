<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: Workshop-Level Automated</b><br>
    <b>Scientific Discovery via Agentic Tree Search</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/"> [Blog Post]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment"> [ICLR2025 Workshop Experiment]</a>
</p>

Fully autonomous scientific research systems are becoming increasingly capable, with AI playing a pivotal role in transforming how scientific discoveries are made.
We are excited to introduce The AI Scientist-v2, a generalized end-to-end agentic system that has generated the first workshop paper written entirely by AI and accepted through peer review.

This system autonomously generates hypotheses, runs experiments, analyzes data, and writes scientific manuscripts. Unlike [its predecessor (AI Scientist-v1)](https://github.com/SakanaAI/AI-Scientist), the AI Scientist-v2 removes reliance on human-authored templates, generalizes across Machine Learning (ML) domains, and employs a progressive agentic tree search, guided by an experiment manager agent.

> **Note:**
> The AI Scientist-v2 doesn’t necessarily produce better papers than v1, especially when a strong starting template is available. v1 follows well-defined templates, leading to high success rates, while v2 takes a broader, more exploratory approach with lower success rates. v1 works best for tasks with clear objectives and a solid foundation, whereas v2 is designed for open-ended scientific exploration.

> **Caution!**
> This codebase will execute Large Language Model (LLM)-written code. There are various risks and challenges associated with this autonomy, including the potential use of dangerous packages, uncontrolled web access, and the possibility of spawning unintended processes. Ensure that you run this within a controlled sandbox environment (e.g., a Docker container). Use at your own discretion.

## Table of Contents

1.  [Requirements](#requirements)
    *   [Installation](#installation)
    *   [Supported Models and API Keys](#supported-models-and-api-keys)
2.  [Generate Research Ideas](#generate-research-ideas)
3.  [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments)
4.  [Citing The AI Scientist-v2](#citing-the-ai-scientist-v2)
5.  [Frequently Asked Questions](#frequently-asked-questions)
6.  [Acknowledgement](#acknowledgement)

## Requirements

Runs on Linux. GPU (NVIDIA CUDA) strongly recommended for the experiment stages; CPU-only mode works for writeup and embedding steps.

### Installation

#### Step 1 — Conda environment

```bash
# Create environment (Python 3.11 required)
conda create -n ai_scientist python=3.11
conda activate ai_scientist
```

#### Step 2 — PyTorch

**GPU (NVIDIA CUDA 12.x — recommended):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

**CPU-only (no GPU / testing on laptop):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

> Set `JAX_PLATFORMS=cpu` if using JAX-based components without GPU.

#### Step 3 — System tools

```bash
# PDF rendering + LaTeX linting
conda install anaconda::poppler conda-forge::chktex

# tectonic: standalone TeX engine (no admin rights needed, replaces pdflatex)
conda install conda-forge::tectonic
```

#### Step 4 — Python packages (via uv — 10× faster than pip)

```bash
# Install uv first
pip install uv

# Install all requirements
uv pip install -r requirements.txt
```

> `uv` resolves and installs packages in parallel. On a cold cache this takes ~2 min vs ~20 min with pip.

#### Step 5 — Ollama (local LLM server)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull gemma4:26b-a4b-it-q4_K_M   # compose / report  (MoE, 96 tok/s)
ollama pull qwen3.5:27b                  # code + LaTeX formatting
ollama pull gemma4:e4b                   # feedback / eval
ollama pull qwen2.5vl:7b                 # VLM plot review

# Pull embedding models (for LaTeX RAG + Figure RAG)
ollama pull qwen3-embedding:0.6b         # default embed model
ollama pull embeddinggemma:latest        # benchmark comparison

# Start Ollama with GPU optimisations (Blackwell/Ampere)
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

**CPU-only Ollama** (no GPU):
```bash
# Same pull commands above work on CPU — just slower inference
OLLAMA_NUM_GPU=0 ollama serve
```

#### Step 6 — HuggingFace VL Embedding (optional, for pixel-level Figure RAG)

```bash
# Downloads ~5 GB on first use — GPU strongly recommended
# Model auto-downloads from HuggingFace on first get_embedder() call
# No manual step needed; accelerate must be installed (included in requirements.txt)

# To pre-download explicitly:
python -c "from ai_scientist.hf_embed import get_embedder; get_embedder()"

# CPU-only (slower, ~4 min per image batch):
FIGURE_RAG_BACKEND=hf python -c "
from ai_scientist.hf_embed import get_embedder
emb = get_embedder(device='cpu')
print(emb.embed_texts(['test'])[:, :4])
"
```

#### Step 7 — LaTeX RAG index (one-time, ~10 min)

```bash
# Download the LaTeX reference
wget -O data/latex2e.txt \
  https://mirrors.in3.sahilister.net/ctan/info/latex2e-help-texinfo/latex2e.txt

# Build index (requires Ollama running)
python -c "
from ai_scientist.latex_rag import build_index_for_model
build_index_for_model('qwen3-embedding:0.6b')
"
```

Installation takes **15–30 minutes** (mostly model downloads).

### Supported Models and API Keys

#### OpenAI Models

By default, the system uses the `OPENAI_API_KEY` environment variable for OpenAI models.

#### Gemini Models

By default, the system uses the `GEMINI_API_KEY` environment variable for Gemini models through OpenAI API.

#### Claude Models via AWS Bedrock

To use Claude models provided by Amazon Bedrock, install the necessary additional packages:
```bash
pip install anthropic[bedrock]
```
Next, configure valid [AWS Credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html) and the target [AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html) by setting the following environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`.

#### Local Ollama Models

You can run AI Scientist-v2 fully locally using Ollama models. This eliminates the need for external API keys (OpenAI, Gemini, or Claude).

**Recommended model stack for a single 24 GB GPU:**

| Role | Model | Notes |
|------|-------|-------|
| Code generation | `ollama/qwen3.5:27b` | temp=0.2, deterministic |
| Feedback / eval | `ollama/gemma4:e4b` | MoE, fast |
| Compose / writeup | `ollama/gemma4:26b-a4b-it-q4_K_M` | MoE, 96 tok/s |
| LaTeX formatting | `ollama/qwen3.5:27b` | temp=0.7 |
| VLM (plot review) | `ollama/qwen2.5vl:7b` | vision-language |

**Key implementation details:**

1. **Task-aware model routing** (`ai_scientist/llm.py`): `route_model()` uses keyword fast-paths (eval→gemma4:e4b, code→qwen2.5-coder:7b, compose→gemma4:26b-a4b, latex→qwen3.5:27b) with a gemma4:26b LLM fallback for unknown tasks.
2. **Per-task sampling parameters** (`_ollama_options()`): code tasks use temp=0.2/top_p=0.75/top_k=15; paper tasks use temp=0.7/top_p=0.90/top_k=40.
3. **2-stage graph-ordered writeup**: Stage A composes 6 sections in dependency order (Methods→Results→Discussion→Conclusion→Introduction→Abstract) using gemma4:26b MoE. Stage B formats the markdown into LaTeX using qwen3.5:27b.
4. **LaTeX RAG injection** (see below): Stage B retrieves relevant LaTeX syntax from `data/latex2e.txt` via LightRAG before generating the document.
5. **Native REST client**: Bypasses the OpenAI SDK for `/api/chat` calls to avoid thinking-token overhead on Ollama thinking models.
6. **Spawn context**: Subprocess executor uses `multiprocessing.get_context("spawn")` to avoid CUDA fork crashes.
7. **Flash Attention + KV cache**: Set `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` for Blackwell/Ampere GPUs.

#### Semantic Scholar API (Literature Search)

Our code can optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput during literature search [if you have one](https://www.semanticscholar.org/product/api). This is used during both the ideation and paper writing stages. The system should work without it, though you might encounter rate limits or reduced novelty checking during ideation. If you experience issues with Semantic Scholar, you can skip the citation phase during paper generation.

#### Setting API Keys

Ensure you provide the necessary API keys as environment variables for the models you intend to use. For example:
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export S2_API_KEY="YOUR_S2_KEY_HERE"
# Set AWS credentials if using Bedrock
# export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
# export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
# export AWS_REGION_NAME="your-aws-region"
```

## Figure RAG — Hallucination-Free Figure References

Stage B (LaTeX formatting) sometimes invents `\includegraphics` filenames that don't exist, causing fatal tectonic compilation errors. Figure RAG solves this by retrieving only figures that physically exist on disk, ranked by relevance to the section being written.

### How it works

```
experiments/<run>/figures/*.png
        │
        ▼ text embed (filename → description)  e.g. "cross domain mAP degradation"
  .figure_rag/ index  (qwen3-embedding:0.6b)
        │
        ▼ query at Stage B write time
  section topic: "Results show performance drop across domains"
        │
        ▼ returns ranked filenames that EXIST on disk
  ["02_Cross_Domain_Generalization_Failure.png", "03_Aggregated_Robustness_Gap.png"]
        │
        ▼ Stage B prompt: "VALID FIGURES — use ONLY these filenames"
  no more tectonic "Unable to load picture" errors
```

### Module: `ai_scientist/figure_rag.py`

```python
from ai_scientist.figure_rag import get_figures_for_section, list_valid_figures

# Get all existing figures
valid = list_valid_figures("experiments/run/figures")

# Get top-4 most relevant to a section
figs = get_figures_for_section(
    figures_dir="experiments/run/figures",
    section_topic="cross-domain generalization failure mAP comparison",
    top_k=4,
)
# → ["02_Cross_Domain_Generalization_Failure.png", ...]
```

### Qwen3-VL-Embedding Status

`MedAIBase/Qwen3-VL-Embedding:2b` (3.4 GB) is available on Ollama but currently packaged as a **completion** model (`capabilities: completion`), making `/api/embed` unavailable. Text-description embeddings via `qwen3-embedding:0.6b` are used instead — sufficient for filename→topic matching.

When official Ollama VL-embedding support arrives, swap `FIGURE_RAG_EMBED_MODEL` to enable true pixel-level image retrieval (VQA-style figure search across saved plots).

```bash
# Future: when MedAIBase publishes embedding-capable variant
export FIGURE_RAG_EMBED_MODEL="MedAIBase/Qwen3-VL-Embedding:2b"
```

---

## LaTeX RAG — Syntax-Aware Paper Writing

AI Scientist-v2 integrates a local Retrieval-Augmented Generation (RAG) system that injects relevant LaTeX syntax documentation into the Stage B writeup prompt, reducing compilation errors caused by hallucinated commands.

### Architecture

```
latex2e.txt (873 KB)
      │
      ▼ one-time indexing (LightRAG + Ollama embeddings)
  data/latex_rag_<model>/   ← vector index + knowledge graph cache
      │
      ▼ query at write-time (~1-2s)
  Stage B prompt  ←  "LATEX SYNTAX REFERENCE: ..."
      │
      ▼
  qwen3.5:27b generates valid LaTeX
```

### Embedding Models

Two embedding backends are supported and benchmarked:

| Model | Ollama tag | Size | Dims | Context | MTEB Retrieval |
|-------|-----------|------|------|---------|----------------|
| **Qwen3-Embedding** *(default)* | `qwen3-embedding:0.6b` | 639 MB | 1024 | 32K | **64.65** |
| EmbeddingGemma | `embeddinggemma:latest` | 622 MB | 768 | 2K | 62.49 |

Qwen3-Embedding-0.6B is the default: higher retrieval score, larger context window (32K vs 2K), better for chunking large LaTeX reference sections.

### Setup

```bash
# Pull embedding models
ollama pull qwen3-embedding:0.6b
ollama pull embeddinggemma:latest

# Install LightRAG
pip install lightrag-hku

# Download LaTeX reference (one-time)
wget -O data/latex2e.txt \
  https://mirrors.in3.sahilister.net/ctan/info/latex2e-help-texinfo/latex2e.txt

# Pre-build index (one-time, ~5-10 min — cached after)
conda run -n ai_scientist python -c "
from ai_scientist.latex_rag import build_index_for_model
build_index_for_model('qwen3-embedding:0.6b')
build_index_for_model('embeddinggemma:latest')
"
```

### Benchmark

Run the retrieval benchmark to compare both embedding models across 8 LaTeX query categories (figures, bibliography, equations, tables, sections, lists, document structure, formatting):

```bash
conda run -n ai_scientist python scripts/benchmark_latex_rag.py
```

Results are saved to `data/latex_rag_benchmark.json`. Example output:

```
Model                               Mode     Avg Recall  Avg Latency
qwen3-embedding:0.6b                naive         87.5%        1.23s
qwen3-embedding:0.6b                local         75.0%        3.45s
embeddinggemma:latest               naive         75.0%        1.45s
embeddinggemma:latest               local         62.5%        3.78s
```

### Configuration

Override the default embedding model via environment variable:

```bash
export LATEX_RAG_EMBED_MODEL="embeddinggemma:latest"  # switch to EmbeddingGemma
export LATEX_RAG_EMBED_MODEL="qwen3-embedding:0.6b"   # default: Qwen3
```

The RAG index is built lazily on first use and cached — subsequent runs are instant.

---

## Generate Research Ideas

Before running the full AI Scientist-v2 experiment pipeline, you first use the `ai_scientist/perform_ideation_temp_free.py` script to generate potential research ideas. This script uses an LLM to brainstorm and refine ideas based on a high-level topic description you provide, interacting with tools like Semantic Scholar to check for novelty.

1.  **Prepare a Topic Description:** Create a Markdown file (e.g., `my_research_topic.md`) describing the research area or theme you want the AI to explore. This file should contain sections like `Title`, `Keywords`, `TL;DR`, and `Abstract` to define the scope of the research. Refer to the example file `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected structure and content format. Place your file in a location accessible by the script (e.g., the `ai_scientist/ideas/` directory).

2.  **Run the Ideation Script:** Execute the script from the main project directory, pointing it to your topic description file and specifying the desired LLM.

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`: Path to your topic description Markdown file.
    *   `--model`: The LLM to use for generating ideas (ensure you have the corresponding API key set).
    *   `--max-num-generations`: How many distinct research ideas to attempt generating.
    *   `--num-reflections`: How many refinement steps the LLM should perform for each idea.

3.  **Output:** The script will generate a JSON file named after your input Markdown file (e.g., `ai_scientist/ideas/my_research_topic.json`). This file will contain a list of structured research ideas, including hypotheses, proposed experiments, and related work analysis.

4.  **Proceed to Experiments:** Once you have the generated JSON file containing research ideas, you can proceed to the next section to run the experiments.

This ideation step guides the AI Scientist towards specific areas of interest and produces concrete research directions to be tested in the main experimental pipeline.

## Run AI Scientist-v2 Paper Generation Experiments

Using the JSON file generated in the previous ideation step, you can now launch the main AI Scientist-v2 pipeline. This involves running experiments via agentic tree search, analyzing results, and generating a paper draft.

Specify the models used for the write-up and review phases via command-line arguments.
The configuration for the best-first tree search (BFTS) is located in `bfts_config.yaml`. Adjust parameters in this file as needed.

Key tree search configuration parameters in `bfts_config.yaml`:

-   `agent` config:
    -   Set `num_workers` (number of parallel exploration paths) and `steps` (maximum number of nodes to explore). For example, if `num_workers=3` and `steps=21`, the tree search will explore up to 21 nodes, expanding 3 nodes concurrently at each step.
    -   `num_seeds`: Should generally be the same as `num_workers` if `num_workers` is less than 3. Otherwise, set `num_seeds` to 3.
    -   Note: Other agent parameters like `k_fold_validation`, `expose_prediction`, and `data_preview` are not used in the current version.
-   `search` config:
    -   `max_debug_depth`: The maximum number of times the agent will attempt to debug a failing node before abandoning that search path.
    -   `debug_prob`: The probability of attempting to debug a failing node.
    -   `num_drafts`: The number of initial root nodes (i.e., the number of independent trees to grow) during Stage 1.

Example command to run AI-Scientist-v2 using a generated idea file (e.g., `my_research_topic.json`). Please review `bfts_config.yaml` for detailed tree search parameters (the default config includes `claude-3-5-sonnet` for experiments). Do not set `load_code` if you do not want to initialize experimentation with a code snippet.

#### Cloud API Models Execution:
```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-preview-2024-09-12 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 20
```

#### Local Ollama Models Execution:
```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --model_writeup "ollama/gemma4:26b" \
 --model_citation "ollama/granite4.1:30b" \
 --model_review "ollama/gemma4:26b" \
 --model_agg_plots "ollama/granite4.1:30b" \
 --num_cite_rounds 2
```

Once the initial experimental stage is complete, you will find a timestamped log folder inside the `experiments/` directory. Navigate to `experiments/"timestamp_ideaname"/logs/0-run/` within that folder to find the tree visualization file `unified_tree_viz.html`.
After all experiment stages are complete, the writeup stage begins. The writeup stage typically takes about 20 to 30 minutes in total. Once it finishes, you should see `timestamp_ideaname.pdf` in the `timestamp_ideaname` folder.
For this example run, all stages typically finish within several hours.

## Citing The AI Scientist-v2

If you use **The AI Scientist-v2** in your research, please cite our work as follows:

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## Frequently Asked Questions

**Why wasn't a PDF or a review generated for my experiment?**

The AI Scientist-v2 completes experiments with a success rate that depends on the chosen foundation model, and the complexity of the idea. Higher success rates are generally observed when using powerful models like Claude 3.5 Sonnet for the experimentation phase.

**What is the estimated cost per experiment?**

The ideation step cost depends on the LLM used and the number of generations/reflections, but is generally low (a few dollars). For the main experiment pipeline, using Claude 3.5 Sonnet for the experimentation phase typically costs around $15–$20 per run. The subsequent writing phase adds approximately $5 when using the default models specified in the example command. Using GPT-4o for `model_citation` is recommended as it can help reduce writing costs.

**How do I run The AI Scientist-v2 for different subject fields?**

First, perform the [Generate Research Ideas](#generate-research-ideas) step. Create a new Markdown file describing your desired subject field or topic, following the structure of the example `ai_scientist/ideas/i_cant_believe_its_not_better.md`. Run the `perform_ideation_temp_free.py` script with this file to generate a corresponding JSON idea file. Then, proceed to the [Run AI Scientist-v2 Paper Generation Experiments](#run-ai-scientist-v2-paper-generation-experiments) step, using this JSON file with the `launch_scientist_bfts.py` script via the `--load_ideas` argument.

**What should I do if I have problems accessing the Semantic Scholar API?**

The Semantic Scholar API is used to assess the novelty of generated ideas and to gather citations during the paper write-up phase. If you don't have an API key, encounter rate limits, you may be able to skip these phases.

**I encountered a "CUDA Out of Memory" error. What can I do?**

This error typically occurs when the AI Scientist-v2 attempts to load or run a model that requires more GPU memory than available on your system. To resolve this, you can try updating your ideation prompt file (`ai_scientist/ideas/my_research_topic.md`) to suggest using smaller models for the experiments.

## Acknowledgement

The tree search component implemented within the `ai_scientist` directory is built on top of the [AIDE](https://github.com/WecoAI/aideml) project. We thank the AIDE developers for their valuable contributions and for making their work publicly available.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## ⚖️ License & Responsible Use

This project is licensed under **The AI Scientist Source Code License** (a derivative of the Responsible AI License). 

**Mandatory Disclosure:** By using this code, you are legally bound to clearly and prominently disclose the use of AI in any resulting scientific manuscripts or papers. 

We recommend the following attribution in your paper's Abstract or Methods section:
> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
