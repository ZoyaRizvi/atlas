import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_transformer import TinyTransformer, VOCAB_SIZE


def generate_no_cache(model: TinyTransformer, prompt_ids: list[int], n_new: int):
    tokens = list(prompt_ids)
    total_flops = 0
    per_step_times = []

    for _ in range(n_new):
        t0 = time.perf_counter()
        logits, stats = model.forward_no_cache(tokens)
        per_step_times.append(time.perf_counter() - t0)
        total_flops += stats["flops"]

        next_token = int(np.argmax(logits))  # greedy decoding
        tokens.append(next_token)

    return tokens, total_flops, per_step_times


def generate_with_cache(model: TinyTransformer, prompt_ids: list[int], n_new: int):
    tokens = list(prompt_ids)
    total_flops = 0
    per_step_times = []
    cache_sizes = []

    cache = None
    for tok in prompt_ids:
        logits, cache, stats = model.forward_with_cache(tok, cache)
        total_flops += stats["flops"]

    for _ in range(n_new):
        t0 = time.perf_counter()
        logits, cache, stats = model.forward_with_cache(tokens[-1], cache)
        per_step_times.append(time.perf_counter() - t0)
        total_flops += stats["flops"]
        cache_sizes.append(model.cache_size_bytes(cache))

        next_token = int(np.argmax(logits))
        tokens.append(next_token)

    return tokens, total_flops, per_step_times, cache_sizes


def run_benchmark(n_new_values, prompt_len=8, d_model=64, n_layers=4, n_heads=8):
    model = TinyTransformer(d_model=d_model, n_heads=n_heads,
                             n_layers=n_layers, seed=7)
    rng = np.random.default_rng(0)
    prompt = rng.integers(3, VOCAB_SIZE, size=prompt_len).tolist()

    results = {
        "n_new": [], "time_no_cache": [], "time_with_cache": [],
        "flops_no_cache": [], "flops_with_cache": [],
        "final_cache_bytes": [],
    }

    for n_new in n_new_values:
        t0 = time.perf_counter()
        _, flops_nc, _ = generate_no_cache(model, prompt, n_new)
        t_nc = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, flops_c, _, cache_sizes = generate_with_cache(model, prompt, n_new)
        t_c = time.perf_counter() - t0

        results["n_new"].append(n_new)
        results["time_no_cache"].append(t_nc)
        results["time_with_cache"].append(t_c)
        results["flops_no_cache"].append(flops_nc)
        results["flops_with_cache"].append(flops_c)
        results["final_cache_bytes"].append(cache_sizes[-1] if cache_sizes else 0)

        speedup = t_nc / t_c if t_c > 0 else float("inf")
        print(f"  n_new={n_new:4d}  no_cache={t_nc*1000:8.2f}ms  "
              f"with_cache={t_c*1000:8.2f}ms  speedup={speedup:5.2f}x  "
              f"final_cache={cache_sizes[-1] if cache_sizes else 0} bytes")

    return results


def plot_results(results, out_path="kv_cache_benchmark.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    n_new = results["n_new"]

    axes[0].plot(n_new, [t*1000 for t in results["time_no_cache"]],
                 marker="o", color="#888780", label="No cache")
    axes[0].plot(n_new, [t*1000 for t in results["time_with_cache"]],
                 marker="o", color="#1D9E75", label="With cache")
    axes[0].set_xlabel("Tokens generated")
    axes[0].set_ylabel("Wall-clock time (ms)")
    axes[0].set_title("Generation time")
    axes[0].legend()

    axes[1].plot(n_new, results["flops_no_cache"],
                 marker="o", color="#888780", label="No cache  O(n²)")
    axes[1].plot(n_new, results["flops_with_cache"],
                 marker="o", color="#1D9E75", label="With cache  O(n)")
    axes[1].set_xlabel("Tokens generated")
    axes[1].set_ylabel("Total FLOPs")
    axes[1].set_yscale("log")
    axes[1].set_title("Compute cost (log scale)")
    axes[1].legend()

    axes[2].plot(n_new, [b/1024 for b in results["final_cache_bytes"]],
                 marker="o", color="#EF9F27")
    axes[2].set_xlabel("Tokens generated")
    axes[2].set_ylabel("KV cache size (KB)")
    axes[2].set_title("Cache memory (grows linearly)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    print("="*70)
    print(" KV cache benchmark — TinyTransformer (pure NumPy)")
    print("="*70)
    print()
    print("Model: d_model=64, n_layers=4, n_heads=8 (toy size, real math)")
    print()

    results = run_benchmark(
        n_new_values=[5, 10, 20, 40, 80, 160],
        prompt_len=8,
        d_model=64,
        n_layers=4,
        n_heads=8,
    )

    print()
    print("="*70)
    avg_speedup = np.mean([
        nc/c for nc, c in zip(results["time_no_cache"], results["time_with_cache"])
    ])
    print(f" Average speedup across all lengths: {avg_speedup:.2f}x")
    print(" Speedup GROWS with sequence length — this is the O(n²) vs O(n) effect.")
    print("="*70)

    plot_results(results)
