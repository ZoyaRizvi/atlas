import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_transformer import TinyTransformer, VOCAB_SIZE
from quantization import (
    quantize_model_weights, dequantize_into_model, model_size_bytes,
)


def compare_precisions(model: TinyTransformer, test_sequences: list[list[int]]):
    results = {"fp32": {"logit_mse": [], "token_changed": 0},
               "int8":  {"logit_mse": [], "token_changed": 0},
               "int4":  {"logit_mse": [], "token_changed": 0}}

    # Quantize once, reuse across all test sequences
    quantized8 = quantize_model_weights(model, bits=8)
    model_int8 = dequantize_into_model(model, quantized8)

    quantized4 = quantize_model_weights(model, bits=4)
    model_int4 = dequantize_into_model(model, quantized4)

    for seq in test_sequences:
        logits_fp32, _ = model.forward_no_cache(seq)
        logits_int8, _ = model_int8.forward_no_cache(seq)
        logits_int4, _ = model_int4.forward_no_cache(seq)

        results["fp32"]["logit_mse"].append(0.0)  # reference vs itself
        results["int8"]["logit_mse"].append(np.mean((logits_fp32 - logits_int8) ** 2))
        results["int4"]["logit_mse"].append(np.mean((logits_fp32 - logits_int4) ** 2))

        top_fp32 = int(np.argmax(logits_fp32))
        top_int8 = int(np.argmax(logits_int8))
        top_int4 = int(np.argmax(logits_int4))

        if top_int8 != top_fp32:
            results["int8"]["token_changed"] += 1
        if top_int4 != top_fp32:
            results["int4"]["token_changed"] += 1

    return results, model_int8, model_int4


def main():
    print("="*70)
    print(" Quantization benchmark — TinyTransformer (pure NumPy)")
    print("="*70)
    print()

    # A slightly larger model so the size numbers are more realistic
    model = TinyTransformer(d_model=128, n_heads=8, n_layers=6, d_ff=256, seed=3)

    total_params = (
        model.W_embed.size + model.W_unembed.size +
        sum(layer[n].size for layer in model.layers
            for n in ["W_q","W_k","W_v","W_o","W_ff1","W_ff2"])
    )
    print(f"Model: d_model=128, n_layers=6, n_heads=8")
    print(f"Total parameters: {total_params:,}")
    print()

    # Size comparison
    print("--- Model size at each precision ---")
    sizes = {}
    for bits in [32, 8, 4]:
        size = model_size_bytes(model, bits=bits)
        sizes[bits] = size
        print(f"  {bits:2d}-bit: {size:10,} bytes  ({size/1024:8.1f} KB)  "
              f"compression: {sizes[32]/size:.1f}x")
    print()

    # Accuracy comparison
    print("--- Output accuracy vs FP32 reference ---")
    rng = np.random.default_rng(99)
    test_sequences = [
        rng.integers(3, VOCAB_SIZE, size=rng.integers(5, 20)).tolist()
        for _ in range(50)
    ]

    results, model_int8, model_int4 = compare_precisions(model, test_sequences)

    for precision in ["int8", "int4"]:
        mse = np.mean(results[precision]["logit_mse"])
        changed = results[precision]["token_changed"]
        pct = 100 * changed / len(test_sequences)
        print(f"  {precision.upper()}: mean logit MSE = {mse:.6f}  |  "
              f"greedy token changed in {changed}/{len(test_sequences)} "
              f"sequences ({pct:.1f}%)")

    print()
    print("="*70)
    print(" Interpretation:")
    print("   - INT8 should show small logit error and few/no token changes")
    print("   - INT4 trades more accuracy for 8x smaller file size")
    print("   - On a REAL trained model the token-change rate would directly")
    print("     predict how often quantization changes generated text")
    print("="*70)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bits_list = [32, 8, 4]
    size_kb = [sizes[b]/1024 for b in bits_list]
    axes[0].bar([str(b) for b in bits_list], size_kb,
               color=["#888780", "#3B8BD4", "#1D9E75"])
    axes[0].set_xlabel("Bits per weight")
    axes[0].set_ylabel("Model size (KB)")
    axes[0].set_title("Model size vs precision")
    for i, v in enumerate(size_kb):
        axes[0].text(i, v + max(size_kb)*0.02, f"{v:.1f} KB", ha="center")

    mse_int8 = np.mean(results["int8"]["logit_mse"])
    mse_int4 = np.mean(results["int4"]["logit_mse"])
    axes[1].bar(["INT8", "INT4"], [mse_int8, mse_int4],
               color=["#3B8BD4", "#1D9E75"])
    axes[1].set_ylabel("Mean logit MSE vs FP32")
    axes[1].set_title("Accuracy loss vs precision")

    plt.tight_layout()
    plt.savefig("quantization_benchmark.png", dpi=130)
    print("\nChart saved to quantization_benchmark.png")


if __name__ == "__main__":
    main()