# AI Scientist v2: Local Setup & Ollama Integration

This document outlines the architecture, environment configurations, and modifications made to run **Sakana AI-Scientist-v2** fully locally on **Ollama** models.

---

## 1. Conda Environment & Jupyter Kernel

To isolate dependencies and support offline PDF compilation (using `chktex`, `poppler`, etc.), a dedicated Conda environment is used.

- **Environment Path:** `/home/kaiser/.conda/envs/ai_scientist`
- **Jupyter Kernel Name:** `ai_scientist`
- **Activating the Environment:**
  ```bash
  conda activate /home/kaiser/.conda/envs/ai_scientist
  ```

### Notebook Conversion & Execution
- **Jupytext** (to convert `.py` percent formats to `.ipynb`):
  ```bash
  conda run -n ai_scientist jupytext --to notebook script.py
  ```
- **Papermill** (to execute notebooks with the `ai_scientist` kernel):
  ```bash
  conda run -n ai_scientist papermill input.ipynb output.ipynb -k ai_scientist
  ```

---

## 2. Local Ollama Model Routing

The models are configured in [bfts_config.yaml](file:///home/kaiser/projects/test/AI-Scientist-v2/bfts_config.yaml) to run on local endpoints:

| Stage / Role | Configured Model | Purpose |
|---|---|---|
| **Coding & Feedback** | `ollama/granite4.1:30b` | Multi-step code generation, traceback debugging, and iterative coding. |
| **Writeup & Review** | `ollama/gemma4:26b` | Drafting sections, creating abstracts, and reviewing academic papers. |
| **Visual VLM Feedback** | `ollama/llava:7b` | Analyzing and critiquing plotted figures and images. |

---

## 3. Ollama REST API Integration (Bypassing OpenAI SDK)

To bypass the `openai` Python SDK and gain direct control over inference parameters, requests are routed directly to the native Ollama `/api/chat` endpoint.

### Key Modifications in [llm.py](file:///home/kaiser/projects/test/AI-Scientist-v2/ai_scientist/llm.py) & [vlm.py](file:///home/kaiser/projects/test/AI-Scientist-v2/ai_scientist/vlm.py):

1. **Direct HTTP Client (`call_ollama_v1`)**:
   - Replaced OpenAI client completion calls with direct `requests.post("http://localhost:11434/api/chat", json=payload)`.
   - Bypassed OpenAI initialization in `create_client(model)` by returning `None` as the client when an `ollama/` prefix is detected.

2. **Ollama Server Health Check**:
   - Added a `verify_ollama_health()` check before any API call. It pings `http://localhost:11434/` to ensure the Ollama daemon is running, raising an informative connection error if it is offline.

3. **Dynamic Context Size Handling**:
   - Every request payload dynamically sets `"options": {"num_ctx": 32768}` to leverage a 32k context size natively, removing the need to configure custom context lengths inside Modelfiles.

4. **Base64 Multimodal Message Mapping**:
   - The helper `map_openai_messages_to_ollama(messages)` converts standard OpenAI message list formats into Ollama-compatible roles, converting nested JSON image URLs into inline base64 arrays.

5. **Response Mocks for Compatibility**:
   - Stubs out the response object using `MockResponse`, `MockUsage`, `MockChoice`, `MockMessage`, and `MockToolCall` classes to replicate attributes accessed by the treesearch backend (e.g. `choices[0].message.content`, `usage.prompt_tokens`, `system_fingerprint`).

6. **Dynamic Model Loading & Unloading**:
   - Every REST request payload specifies `"keep_alive": 0` (or `"0s"` equivalent) to enforce immediate model unloading from VRAM/RAM after generation. This allows active scheduling and prevents OOM crashes when alternating between large coding, writing, and VLM models.

---

## 4. Pipeline Bugfixes & Stability Patches

1. **HuggingFace Token Check**:
   - Patched [i_cant_believe_its_not_better.py](file:///home/kaiser/projects/test/AI-Scientist-v2/ai_scientist/ideas/i_cant_believe_its_not_better.py) to check for `HF_TOKEN` presence in `os.environ` before executing `login()`. This prevents crashes during offline/local pipeline runs.
2. **Semantic Scholar Rate Limiting (429)**:
   - Patched [semantic_scholar.py](file:///home/kaiser/projects/test/AI-Scientist-v2/ai_scientist/tools/semantic_scholar.py) to catch `429 Too Many Requests` exceptions and skip academic literature checks gracefully, preventing infinite retry loops when running without a Semantic Scholar API key.

---

## 5. Execution Reference

To start the pipeline with local models and the newly generated idea (`idea_idx 3`):

```bash
/home/kaiser/.conda/envs/ai_scientist/bin/python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/i_cant_believe_its_not_better.json" \
  --idea_idx 3 \
  --load_code \
  --model_writeup "ollama/gemma4:26b" \
  --model_citation "ollama/granite4.1:30b" \
  --model_review "ollama/gemma4:26b" \
  --model_agg_plots "ollama/granite4.1:30b" \
  --num_cite_rounds 2
```
