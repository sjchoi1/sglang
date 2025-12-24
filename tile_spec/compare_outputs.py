"""
Simple string matching comparison: EAGLE3 vs TileSpec

Runs both configurations sequentially on the same GPU and compares outputs.
At temperature=0, outputs should be identical.
"""
import subprocess
import sys
import time
import requests
import signal

PROMPTS = [
    "What is the capital of France?",
    "List the first 5 prime numbers.",
    "The quick brown fox jumps over",
]

def wait_for_server(url, proc, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        # Check if process died
        if proc.poll() is not None:
            print("Server process died!")
            out, _ = proc.communicate()
            print(f"Output: {out[-2000:] if out else 'None'}")
            return False
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                print(f"Server ready in {time.time()-start:.1f}s")
                return True
        except:
            pass
        time.sleep(2)
    print("Timeout waiting for server")
    return False

def generate(url, prompt, max_tokens=64):
    resp = requests.post(
        f"{url}/generate",
        json={"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens}},
        timeout=60,
    )
    return resp.json()["text"]

def run_server(args, port):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", "meta-llama/Llama-3.1-8B-Instruct",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
        "--speculative-num-steps", "4",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--dtype", "float16",
        "--port", str(port),
        "--mem-fraction-static", "0.7",
    ] + args

    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc

def main():
    port = 30030
    url = f"http://127.0.0.1:{port}"

    # === Run EAGLE3 (baseline) ===
    print("=" * 60)
    print("Running EAGLE3 (baseline)...")
    print("=" * 60)

    proc = run_server([], port)
    if not wait_for_server(url, proc):
        print("EAGLE3 server failed to start")
        proc.kill()
        return 1

    eagle3_outputs = {}
    for prompt in PROMPTS:
        output = generate(url, prompt)
        eagle3_outputs[prompt] = output
        print(f"Prompt: {prompt[:40]}...")
        print(f"Output: {output[:60]}...\n")

    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(5)  # Wait for GPU memory cleanup

    # === Run TileSpec ===
    print("=" * 60)
    print("Running EAGLE3 + TileSpec...")
    print("=" * 60)

    proc = run_server(["--tile-spec", "--disable-cuda-graph"], port)
    if not wait_for_server(url, proc):
        print("TileSpec server failed to start")
        proc.kill()
        return 1

    tilespec_outputs = {}
    for prompt in PROMPTS:
        output = generate(url, prompt)
        tilespec_outputs[prompt] = output
        print(f"Prompt: {prompt[:40]}...")
        print(f"Output: {output[:60]}...\n")

    # Check TileSpec status
    try:
        info = requests.get(f"{url}/get_server_info", timeout=5).json()
        ready = info.get("internal_states", [{}])[0].get("tile_spec_ready", False)
        accept = info.get("internal_states", [{}])[0].get("avg_spec_accept_length", 0)
        print(f"TileSpec ready: {ready}, avg_accept_length: {accept:.2f}\n")
    except:
        pass

    proc.terminate()
    proc.wait(timeout=10)

    # === Compare outputs ===
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    all_match = True
    for prompt in PROMPTS:
        e3 = eagle3_outputs[prompt]
        ts = tilespec_outputs[prompt]
        match = e3 == ts

        print(f"\nPrompt: {prompt[:40]}...")
        print(f"Match: {'YES' if match else 'NO'}")

        if not match:
            all_match = False
            print(f"EAGLE3:   {e3[:80]}...")
            print(f"TileSpec: {ts[:80]}...")

    print("\n" + "=" * 60)
    if all_match:
        print("SUCCESS: All outputs match!")
    else:
        print("FAILURE: Some outputs differ!")
    print("=" * 60)

    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())
