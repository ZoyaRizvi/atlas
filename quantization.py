from __future__ import annotations
import numpy as np

def quantize_tensor(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    qmax = 2 ** (bits - 1) - 1   # e.g. bits=4 -> qmax=7,  bits=8 -> qmax=127
    qmin = -(2 ** (bits - 1))    # e.g. bits=4 -> qmin=-8, bits=8 -> qmin=-128

    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / qmax

    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, qmin, qmax)
    return q, scale


def dequantize_tensor(q: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct an approximate float32 array from quantized ints."""
    return (q.astype(np.float32) * scale)


def pack_int4(q: np.ndarray) -> np.ndarray:
    flat = q.flatten()
    if flat.size % 2 != 0:
        flat = np.append(flat, 0)  # pad odd-length arrays

    unsigned = (flat + 8).astype(np.uint8)  # shift -8..7 -> 0..15
    low = unsigned[0::2]
    high = unsigned[1::2]

    packed = (high << 4) | low  # combine two nibbles into one byte
    return packed.astype(np.uint8)


def unpack_int4(packed: np.ndarray, original_size: int) -> np.ndarray:
    """
    Reverse pack_int4(): expand packed bytes back into individual
    int4 values (range -8..7).
    """
    low = packed & 0x0F           # mask out low nibble
    high = (packed >> 4) & 0x0F   # shift down, mask out high nibble

    unsigned = np.empty(packed.size * 2, dtype=np.uint8)
    unsigned[0::2] = low
    unsigned[1::2] = high

    signed = unsigned[:original_size].astype(np.int32) - 8  # undo the +8 shift
    return signed


def verify_int4_packing(n_values: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n_values).astype(np.float32)

    q, scale = quantize_tensor(x, bits=4)
    packed = pack_int4(q)
    unpacked = unpack_int4(packed, original_size=n_values)

    matches = np.array_equal(q, unpacked)
    return matches, q, unpacked, packed


def quantize_model_weights(model, bits: int = 8) -> dict:

    quantized = {}

    q, s = quantize_tensor(model.W_embed, bits)
    quantized["W_embed"] = (q, s)
    q, s = quantize_tensor(model.W_unembed, bits)
    quantized["W_unembed"] = (q, s)

    for i, layer in enumerate(model.layers):
        for name in ["W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2"]:
            q, s = quantize_tensor(layer[name], bits)
            quantized[f"layer{i}.{name}"] = (q, s)

    return quantized


def dequantize_into_model(model, quantized: dict):
    import copy
    new_model = copy.deepcopy(model)

    q, s = quantized["W_embed"]
    new_model.W_embed = dequantize_tensor(q, s)
    q, s = quantized["W_unembed"]
    new_model.W_unembed = dequantize_tensor(q, s)

    for i, layer in enumerate(new_model.layers):
        for name in ["W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2"]:
            q, s = quantized[f"layer{i}.{name}"]
            layer[name] = dequantize_tensor(q, s)

    return new_model


def model_size_bytes(model, bits: int = 32) -> int:
    total_params = model.W_embed.size + model.W_unembed.size
    for layer in model.layers:
        for name in ["W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2"]:
            total_params += layer[name].size

    if bits == 32:
        return total_params * 4
    elif bits == 8:
        return total_params * 1
    elif bits == 4:
        return (total_params + 1) // 2  # 2 values per byte, round up
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
