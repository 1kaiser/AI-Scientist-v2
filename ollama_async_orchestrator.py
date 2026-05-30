import asyncio
import subprocess
import socket
import time
import httpx
import sys

# Configuration
OLLAMA_PATH = "/home/kaiser/ollama/bin/ollama"
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

def is_port_in_use(host, port):
    """Check if the target port is already occupied by a running server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

async def start_ollama_server():
    """
    Start the Ollama server programmatically if it is not already running.
    Returns the Popen process reference if started by this script, otherwise None.
    """
    if is_port_in_use(OLLAMA_HOST, OLLAMA_PORT):
        print(f"[Info] Ollama is already running on port {OLLAMA_PORT}. Reusing existing server.")
        return None

    print(f"[Ollama] Starting server process from {OLLAMA_PATH}...")
    try:
        # Start the server daemon in the background
        process = subprocess.Popen(
            [OLLAMA_PATH, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll the server endpoint until it responds with 200 OK
        for attempt in range(20):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(OLLAMA_URL, timeout=1.0)
                    if response.status_code == 200:
                        print(f"[Ollama] Server started successfully (PID: {process.pid}).")
                        return process
            except (httpx.RequestError, httpx.HTTPError):
                pass
            await asyncio.sleep(0.5)
            
        raise TimeoutError("Ollama server failed to start within 10 seconds.")
    except Exception as e:
        print(f"[Error] Failed to automatically launch Ollama: {e}")
        sys.exit(1)

async def perform_async_inference(model_name: str, prompt: str, keep_alive: int | str = 0):
    """
    Perform asynchronous inference on a specific model.
    Passes the keep_alive parameter to control model caching.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0.7
        },
        # keep_alive controls model unloading:
        # - 0 or "0s": Unload the model immediately after responding.
        # - "-1" or "forever": Keep the model loaded in memory indefinitely.
        # - e.g. "5m": Keep the model loaded for 5 minutes of idle time.
        "keep_alive": keep_alive
    }
    
    print(f"[{model_name}] Starting query...")
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            
        elapsed = time.time() - start_time
        content = data.get("message", {}).get("content", "")
        print(f"[{model_name}] Completed in {elapsed:.2f}s.")
        return content
    except Exception as e:
        print(f"[{model_name}] Error during inference: {e}")
        return None

async def explicit_unload_model(model_name: str):
    """
    Explicitly unload a model from memory.
    This is done by sending an empty generation request with keep_alive set to 0.
    """
    print(f"[Ollama] Sending explicit unload request for model: {model_name}...")
    payload = {
        "model": model_name,
        "keep_alive": 0
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # We hit the base generate endpoint with keep_alive: 0
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            if response.status_code == 200:
                print(f"[Ollama] Model {model_name} has been successfully unloaded from VRAM.")
    except Exception as e:
        print(f"[Ollama] Failed to unload model {model_name}: {e}")

async def main():
    # 1. Ensure Ollama is running
    server_process = await start_ollama_server()
    
    # Give the runner sub-processes a brief moment to settle
    await asyncio.sleep(1.0)
    
    # 2. Define concurrent tasks to execute inference on multiple models
    # Setting keep_alive=0 tells Ollama to immediately free the model after query
    tasks = [
        perform_async_inference("qwen2.5:7b", "Solve: what is 123 * 456? Give just the final number.", keep_alive=0),
        perform_async_inference("llava:7b", "Solve: what is 987 - 654? Give just the final number.", keep_alive=0)
    ]
    
    print("\n--- Running Concurrent Inferences ---")
    results = await asyncio.gather(*tasks)
    
    print("\n--- Results ---")
    for model, res in zip(["qwen2.5:7b", "llava:7b"], results):
        print(f"{model}: {repr(res)}")
        
    # 3. Demonstration: Explictly loading a model, then unloading it later
    print("\n--- Explicit Load / Unload Demonstration ---")
    # Load model and keep it active in memory for up to 5 minutes
    await perform_async_inference("qwen2.5:7b", "Hello!", keep_alive="5m")
    
    # Unload it immediately when we are done with it
    await explicit_unload_model("qwen2.5:7b")
    
    # 4. Cleanup/Exit the programmatically started server
    if server_process is not None:
        print("\n[Ollama] Shutting down the programmatically started server...")
        server_process.terminate()
        try:
            # Wait for clean exit
            server_process.wait(timeout=5.0)
            print("[Ollama] Server exited cleanly.")
        except subprocess.TimeoutExpired:
            print("[Ollama] Server did not exit in time. Killing...")
            server_process.kill()
            print("[Ollama] Server killed.")
    else:
        print("\n[Info] Reused an existing Ollama server. Leaving it running on exit.")

if __name__ == "__main__":
    asyncio.run(main())
