import json
import os
import re
import time
import subprocess
import atexit
import httpx
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage

import anthropic
import backoff
import openai
import requests

class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class MockToolCall:
    def __init__(self, function):
        self.function = function

class MockMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class MockChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason

class MockCompletionTokensDetails:
    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens

class MockUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.completion_tokens_details = MockCompletionTokensDetails()

class MockResponse:
    def __init__(self, choices, usage=None, model="ollama-model", created=None):
        import time
        self.choices = choices
        self.usage = usage or MockUsage()
        self.system_fingerprint = "fp_ollama"
        self.model = model
        self.created = created or int(time.time())

_OLLAMA_HEALTHY = None
_OLLAMA_PROCESS = None
_LLM_CALL_COUNT = 0
_LLM_LOG_PATH = None

def _log_llm_call(model, prompt_tokens, completion_tokens, duration_s, attempt, empty_retries):
    """Append a line to the LLM call log for progress monitoring."""
    global _LLM_CALL_COUNT, _LLM_LOG_PATH
    _LLM_CALL_COUNT += 1
    
    if _LLM_LOG_PATH is None:
        # Try to find the experiment log dir from env or use a default
        root = os.environ.get("AI_SCIENTIST_ROOT", os.getcwd())
        log_dir = os.path.join(root, "llm_calls.jsonl")
        _LLM_LOG_PATH = log_dir
    
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "call_num": _LLM_CALL_COUNT,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_s": round(duration_s, 2),
        "attempt": attempt,
        "empty_retries": empty_retries,
        "pid": os.getpid(),
    }
    try:
        with open(_LLM_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never break the pipeline for logging

def verify_ollama_health():
    global _OLLAMA_HEALTHY, _OLLAMA_PROCESS
    if _OLLAMA_HEALTHY is True:
        return
    base_url = "http://localhost:11434"
    try:
        response = requests.get(f"{base_url}/", timeout=3)
        if response.status_code == 200:
            _OLLAMA_HEALTHY = True
            return
    except requests.exceptions.RequestException:
        pass

    # Try starting Ollama serve automatically in the background
    try:
        _OLLAMA_PROCESS = subprocess.Popen(
            ["/home/kaiser/ollama/bin/ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Wait up to 10 seconds for the server to listen
        for _ in range(20):
            try:
                response = requests.get(f"{base_url}/", timeout=1)
                if response.status_code == 200:
                    _OLLAMA_HEALTHY = True
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
    except Exception as e:
        pass

    raise ConnectionError(
        f"Ollama server is not reachable at {base_url} and auto-start failed. "
        "Please ensure the Ollama daemon is running."
    )

def cleanup_ollama():
    global _OLLAMA_PROCESS
    if _OLLAMA_PROCESS is not None:
        print("[Ollama] Shutting down automatically started server...", flush=True)
        _OLLAMA_PROCESS.terminate()
        try:
            _OLLAMA_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _OLLAMA_PROCESS.kill()

atexit.register(cleanup_ollama)

def strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by Qwen3/DeepSeek models."""
    if not text:
        return text
    # Remove <think>...</think> blocks (possibly multi-line)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def map_openai_messages_to_ollama(messages):
    mapped_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        images = []
        text_parts = []
        
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    img_url = part.get("image_url", {}).get("url", "")
                    if "," in img_url:
                        base64_data = img_url.split(",")[1]
                    else:
                        base64_data = img_url
                    images.append(base64_data)
            text_content = "".join(text_parts)
        else:
            text_content = content or ""
            
        mapped_msg = {"role": role, "content": text_content}
        if images:
            mapped_msg["images"] = images
        mapped_messages.append(mapped_msg)
        
    return mapped_messages

def call_ollama_v1(model, messages, temperature=0.7, max_tokens=4096, n=1, stop=None, tools=None, tool_choice=None):
    verify_ollama_health()
    
    mapped_messages = map_openai_messages_to_ollama(messages)
    clean_model = model.replace("ollama/", "")
    
    options = {
        "num_ctx": 16384
    }
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if stop is not None:
        options["stop"] = stop
        
    payload = {
        "model": clean_model,
        "messages": mapped_messages,
        "stream": False,
        "options": options,
        "keep_alive": 300,  # keep model warm for 5 min between calls
        "think": False,  # Disable chain-of-thought for Qwen3/supported models
    }
    if tools is not None:
        payload["tools"] = tools

    base_url = "http://localhost:11434/api/chat"

    max_retries = 3
    empty_retries = 0
    t_start = time.time()
    for attempt in range(max_retries):
        response = requests.post(base_url, json=payload, timeout=600)
        response.raise_for_status()
        res_data = response.json()
        
        msg_data = res_data.get("message", {})
        raw_content = msg_data.get("content") or ""
        content = strip_thinking_tags(raw_content)
        
        tool_calls = None
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                func_data = tc.get("function", {})
                func_args = func_data.get("arguments", {})
                args_str = json.dumps(func_args)
                tool_calls.append(
                    MockToolCall(
                        MockFunction(func_data.get("name"), args_str)
                    )
                )
        
        # If we have content or tool calls, we're good
        if content or tool_calls:
            break
        
        # Empty response — retry with slightly higher temperature
        empty_retries += 1
        print(f"[Ollama] Attempt {attempt + 1}/{max_retries}: empty response from {clean_model}, retrying...", flush=True)
        if attempt < max_retries - 1:
            payload["options"]["temperature"] = min((temperature or 0.7) + 0.1 * (attempt + 1), 1.0)
            time.sleep(1)
    
    if not content and not tool_calls:
        # Last resort: return a placeholder so the pipeline can handle it gracefully
        print(f"[Ollama] WARNING: {clean_model} returned empty content after {max_retries} attempts.", flush=True)
        content = "I was unable to generate a response. Please try again."
    
    prompt_tokens = res_data.get("prompt_eval_count", 0)
    completion_tokens = res_data.get("eval_count", 0)
    duration_s = time.time() - t_start
    
    # Log call for progress monitoring
    _log_llm_call(clean_model, prompt_tokens, completion_tokens, duration_s, attempt + 1, empty_retries)
    
    choices = []
    msg = MockMessage(content, tool_calls)
    choices.append(MockChoice(msg, res_data.get("done_reason", "stop")))
    
    usage = MockUsage(prompt_tokens, completion_tokens)
    
    return MockResponse(
        choices, 
        usage, 
        model=res_data.get("model", model), 
        created=int(time.time())
    )

async def async_call_ollama_v1(model, messages, temperature=0.7, max_tokens=4096, n=1, stop=None, tools=None, tool_choice=None):
    import asyncio
    verify_ollama_health()
    
    mapped_messages = map_openai_messages_to_ollama(messages)
    clean_model = model.replace("ollama/", "")
    
    options = {
        "num_ctx": 16384
    }
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if stop is not None:
        options["stop"] = stop
        
    payload = {
        "model": clean_model,
        "messages": mapped_messages,
        "stream": False,
        "options": options,
        "keep_alive": 300,  # keep model warm for 5 min between calls
        "think": False,  # Disable chain-of-thought for Qwen3/supported models
    }
    if tools is not None:
        payload["tools"] = tools

    base_url = "http://localhost:11434/api/chat"

    max_retries = 3
    res_data = {}
    content = ""
    tool_calls = None

    async with httpx.AsyncClient(timeout=600.0) as client:
        for attempt in range(max_retries):
            response = await client.post(base_url, json=payload)
            response.raise_for_status()
            res_data = response.json()
            
            msg_data = res_data.get("message", {})
            raw_content = msg_data.get("content") or ""
            content = strip_thinking_tags(raw_content)
            
            tool_calls = None
            if "tool_calls" in msg_data and msg_data["tool_calls"]:
                tool_calls = []
                for tc in msg_data["tool_calls"]:
                    func_data = tc.get("function", {})
                    func_args = func_data.get("arguments", {})
                    args_str = json.dumps(func_args)
                    tool_calls.append(
                        MockToolCall(
                            MockFunction(func_data.get("name"), args_str)
                        )
                    )
            
            if content or tool_calls:
                break
            
            print(f"[Ollama-async] Attempt {attempt + 1}/{max_retries}: empty response from {clean_model}, retrying...", flush=True)
            if attempt < max_retries - 1:
                payload["options"]["temperature"] = min((temperature or 0.7) + 0.1 * (attempt + 1), 1.0)
                await asyncio.sleep(1)
    
    if not content and not tool_calls:
        print(f"[Ollama-async] WARNING: {clean_model} returned empty content after {max_retries} attempts.", flush=True)
        content = "I was unable to generate a response. Please try again."
    
    choices = []
    msg = MockMessage(content, tool_calls)
    choices.append(MockChoice(msg, res_data.get("done_reason", "stop")))
    
    prompt_tokens = res_data.get("prompt_eval_count", 0)
    completion_tokens = res_data.get("eval_count", 0)
    usage = MockUsage(prompt_tokens, completion_tokens)
    
    return MockResponse(
        choices, 
        usage, 
        model=res_data.get("model", model), 
        created=int(time.time())
    )


MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # DeepSeek Models
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama 3 models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # Google Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    # GPT-OSS models via Ollama
    "ollama/gpt-oss:20b",
    "ollama/gpt-oss:120b",
    # Qwen models via Ollama
    "ollama/qwen3:8b",
    "ollama/qwen3:32b",
    "ollama/qwen3:235b",

    "ollama/qwen2.5vl:8b",
    "ollama/qwen2.5vl:32b",

    "ollama/qwen3-coder:70b",
    "ollama/qwen3-coder:480b",

    # Deepseek models via Ollama
    "ollama/deepseek-r1:8b",
    "ollama/deepseek-r1:32b",
    "ollama/deepseek-r1:70b",
    "ollama/deepseek-r1:671b",
    # User's local Ollama models
    "ollama/qwen3.5:27b",
    "ollama/qwen2.5:7b",
    "ollama/deepseek-r1:7b",
    "ollama/gemma4:26b",
    "ollama/granite4.1:30b",
    "ollama/nemotron-3-nano:30b",
    "ollama/llava:7b",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if model.startswith("ollama/"):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = call_ollama_v1(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif 'gemini' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if model.startswith("ollama/"):
        return call_ollama_v1(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        # response = make_llm_call(client, model, temperature, system_message=system_message, prompt=new_msg_history)
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model.startswith("ollama/"):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = call_ollama_v1(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepcoder-14b":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(
                model="agentica-org/DeepCoder-14B-Preview",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
        except Exception as e:
            # Fallback to direct API call if OpenAI client doesn't work with HuggingFace
            import requests
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                content = response.json()["generated_text"]
            else:
                raise ValueError(f"Error from HuggingFace API: {response.text}")

        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif 'gemini' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None: 
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif model.startswith("ollama/"):
        print(f"Using Ollama (direct HTTP requests) with model {model}.")
        return None, model
    elif "gpt" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    elif model == "deepcoder-14b":
        print(f"Using HuggingFace API with {model}.")
        # Using OpenAI client with HuggingFace API
        if "HUGGINGFACE_API_KEY" not in os.environ:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return (
            openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ),
            model,
        )
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "meta-llama/llama-3.1-405b-instruct",
        )
    elif 'gemini' in model:
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            model,
        )
    else:
        raise ValueError(f"Model {model} not supported.")
