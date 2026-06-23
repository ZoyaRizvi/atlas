from __future__ import annotations
import numpy as np

VOCAB = ["<pad>", "<bos>", "<eos>"] + [f"tok{i}" for i in range(61)]
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
 
 
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax - subtract max before exponentiating"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
 
 
class TinyTransformer:
    def __init__(self, d_model=32, n_heads=4, n_layers=2, d_ff=64, seed=42):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_head = d_model // n_heads
 
        rng = np.random.default_rng(seed)
        scale = 0.02  # small init, standard for transformers
 
        self.W_embed = rng.normal(0, scale, (VOCAB_SIZE, d_model)).astype(np.float32)
        self.W_unembed = rng.normal(0, scale, (d_model, VOCAB_SIZE)).astype(np.float32)
 
        self.layers = []
        for _ in range(n_layers):
            layer = {
                "W_q": rng.normal(0, scale, (d_model, d_model)).astype(np.float32),
                "W_k": rng.normal(0, scale, (d_model, d_model)).astype(np.float32),
                "W_v": rng.normal(0, scale, (d_model, d_model)).astype(np.float32),
                "W_o": rng.normal(0, scale, (d_model, d_model)).astype(np.float32),
                "W_ff1": rng.normal(0, scale, (d_model, d_ff)).astype(np.float32),
                "W_ff2": rng.normal(0, scale, (d_ff, d_model)).astype(np.float32),
            }
            self.layers.append(layer)
 
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq, d_model) -> (n_heads, seq, d_head)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 0, 2)
 
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(n_heads, seq, d_head) -> (seq, d_model)"""
        n_heads, seq_len, d_head = x.shape
        return x.transpose(1, 0, 2).reshape(seq_len, n_heads * d_head)
 
    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Standard scaled dot-product causal attention.
        q: (n_heads, q_len, d_head)   — queries for the NEW token(s)
        k: (n_heads, kv_len, d_head)  — keys for ALL tokens so far (cached or fresh)
        v: (n_heads, kv_len, d_head)  — values for ALL tokens so far
 
        q_len can be 1 (generation with cache) or seq_len (no cache / prefill).
        kv_len is always the full sequence length up to this point.
        """
        d_head = q.shape[-1]
        scores = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(d_head)
 
        q_len, kv_len = scores.shape[1], scores.shape[2]
        if q_len > 1:
            mask = np.triu(np.ones((q_len, kv_len)), k=kv_len - q_len + 1).astype(bool)
            scores = np.where(mask, -1e9, scores)
 
        weights = softmax(scores, axis=-1)
        out = np.einsum("hqk,hkd->hqd", weights, v)
        return out
 
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Simplified layer norm (no learned scale/bias, for clarity)"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)
 
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation — the standard nonlinearity in transformer FFNs."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
 
    def forward_no_cache(self, token_ids: list[int]) -> tuple[np.ndarray, dict]:
        x = self.W_embed[token_ids]  # (seq_len, d_model)
        flops = 0
 
        for layer in self.layers:
            x_norm = self._layer_norm(x)
 
            q = x_norm @ layer["W_q"]
            k = x_norm @ layer["W_k"]
            v = x_norm @ layer["W_v"]
            flops += 3 * x_norm.shape[0] * self.d_model * self.d_model * 2
 
            q_h, k_h, v_h = self._split_heads(q), self._split_heads(k), self._split_heads(v)
            attn_out = self._attention(q_h, k_h, v_h)
            flops += self.n_heads * x_norm.shape[0] * x_norm.shape[0] * self.d_head * 2 * 2
 
            attn_out = self._merge_heads(attn_out) @ layer["W_o"]
            flops += x_norm.shape[0] * self.d_model * self.d_model * 2
            x = x + attn_out
 
            x_norm2 = self._layer_norm(x)
            ff = self._gelu(x_norm2 @ layer["W_ff1"]) @ layer["W_ff2"]
            flops += 2 * x_norm2.shape[0] * self.d_model * self.d_ff * 2
            x = x + ff
 
        logits = x[-1] @ self.W_unembed
        return logits, {"flops": flops, "seq_len": len(token_ids)}
 
 
    def forward_with_cache(self, new_token_id: int, cache: list[dict] | None
                            ) -> tuple[np.ndarray, list[dict], dict]:
        if cache is None:
            cache = [{"k": None, "v": None} for _ in range(self.n_layers)]
 
        x = self.W_embed[new_token_id][None, :]  # (1, d_model) — just ONE token
        flops = 0
        new_cache = []
 
        for layer, layer_cache in zip(self.layers, cache):
            x_norm = self._layer_norm(x)
 
            q = x_norm @ layer["W_q"]   # (1, d_model)
            k_new = x_norm @ layer["W_k"]
            v_new = x_norm @ layer["W_v"]
            flops += 3 * 1 * self.d_model * self.d_model * 2
 
            q_h = self._split_heads(q)            # (n_heads, 1, d_head)
            k_new_h = self._split_heads(k_new)    # (n_heads, 1, d_head)
            v_new_h = self._split_heads(v_new)    # (n_heads, 1, d_head)
 
            if layer_cache["k"] is None:
                k_h = k_new_h
                v_h = v_new_h
            else:
                k_h = np.concatenate([layer_cache["k"], k_new_h], axis=1)
                v_h = np.concatenate([layer_cache["v"], v_new_h], axis=1)
 
            new_cache.append({"k": k_h, "v": v_h})
 
            attn_out = self._attention(q_h, k_h, v_h)  # (n_heads, 1, d_head)
            flops += self.n_heads * 1 * k_h.shape[1] * self.d_head * 2 * 2
 
            attn_out = self._merge_heads(attn_out) @ layer["W_o"]
            flops += 1 * self.d_model * self.d_model * 2
            x = x + attn_out
 
            x_norm2 = self._layer_norm(x)
            ff = self._gelu(x_norm2 @ layer["W_ff1"]) @ layer["W_ff2"]
            flops += 2 * 1 * self.d_model * self.d_ff * 2
            x = x + ff
 
        logits = x[0] @ self.W_unembed
        seq_len = new_cache[0]["k"].shape[1]
        return logits, new_cache, {"flops": flops, "seq_len": seq_len}
 
    def cache_size_bytes(self, cache: list[dict], dtype_bytes: int = 4) -> int:
        if cache is None or cache[0]["k"] is None:
            return 0
        seq_len = cache[0]["k"].shape[1]
        return 2 * self.n_layers * self.n_heads * seq_len * self.d_head * dtype_bytes
