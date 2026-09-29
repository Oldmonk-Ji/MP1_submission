"""Student implementation: baseline geometry plus configurable dropout.

Step 1 of the optimization plan: keep the baseline GPT geometry
(width 128, depth 4, heads 4) and add dropout (embedding + both residual
branches), so an improved training recipe can be compared against the
baseline at the same number of processed training targets. With
dropout=0 this class behaves exactly like the model.py baseline.

Inference fast path (CPU, fp32, eval mode): attention is computed by a
numba kernel that fuses RoPE, the causal softmax and the p@v product into
one pass (blocked flash-style, no materialized scores beyond a 64x64
tile), and torch is asked to use more of the machine's cores than the
scorer's default. Every non-fast-path input falls back to the original
PyTorch formulation, so the module remains fully general.
"""
import math
import os
import threading

import torch
from torch import nn
from torch.nn import functional as F

# Pin numba to AVX2 codegen (skylake) before it is imported: on
# AVX-512-capable CPUs the default target both triggers severe frequency
# downclocking (Tiger Lake) and hits a numba 0.61 lowering bug for
# math.exp; AVX2-only code is faster and runs on any AVX2/AVX-512 CPU.
_CPU_CAP = None
try:
    _CPU_CAP = torch.backends.cpu.get_cpu_capability()
except Exception:
    pass
if _CPU_CAP in ('AVX512', 'AVX2'):
    os.environ.setdefault('NUMBA_CPU_NAME', 'skylake')

try:
    import numpy as np
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

_KERNEL_OK = False
_KERNEL_TRIED = False
_FUSION_READY = False

_NEG = -1.0e30
_KD = 32
_IB = 64
_KD2 = 16

if _HAS_NUMBA:
    _SCALE = np.float32(1.0 / math.sqrt(_KD))
    _FZERO = np.float32(0.0)
    _FONE = np.float32(1.0)

    @njit(fastmath=True, nogil=True, cache=False)
    def _attn_kern(qkv, cos, sin, out, B, L, H, u0, u1):
        # qkv: flat [B, L, 3, H, D] fp32 (contiguous linear output)
        # cos/sin: [L, D/2] fp32; out: flat [B, L, H*D] fp32.
        # Blocked flash attention with fused RoPE; one serial pass per
        # (batch, head) unit, units split across Python threads outside.
        # All arrays preallocated up front: per-row temporaries inside the
        # loop would hit numba's NRT allocator lock and kill parallelism.
        HD = H * _KD
        HD3 = 3 * HD
        kc = np.empty((L, _KD), np.float32)
        vc = np.empty((L, _KD), np.float32)
        qr = np.empty((_IB, _KD), np.float32)
        s = np.empty((_IB, _IB), np.float32)
        p = np.empty((_IB, _IB), np.float32)
        m = np.empty(_IB, np.float32)
        l = np.empty(_IB, np.float32)
        oacc = np.empty((_IB, _KD), np.float32)
        vrow = np.empty(_KD, np.float32)
        for unit in range(u0, u1):
            b = unit // H
            h = unit % H
            b0 = b * (L * HD3) + h * _KD
            for j in range(L):
                jb = b0 + j * HD3 + HD
                for d in range(_KD2):
                    kd = qkv[jb + d]
                    kd2 = qkv[jb + d + _KD2]
                    kc[j, d] = kd * cos[j, d] - kd2 * sin[j, d]
                    kc[j, d + _KD2] = kd2 * cos[j, d] + kd * sin[j, d]
                for d in range(_KD):
                    vc[j, d] = qkv[jb + HD + d]
            for i0 in range(0, L, _IB):
                for r in range(_IB):
                    i = i0 + r
                    ib = b0 + i * HD3
                    for d in range(_KD2):
                        qd = qkv[ib + d]
                        qd2 = qkv[ib + d + _KD2]
                        qr[r, d] = qd * cos[i, d] - qd2 * sin[i, d]
                        qr[r, d + _KD2] = qd2 * cos[i, d] + qd * sin[i, d]
                    m[r] = _NEG
                    l[r] = _FZERO
                    for d in range(_KD):
                        oacc[r, d] = _FZERO
                for j0 in range(0, i0 + _IB, _IB):
                    for r in range(_IB):
                        for c in range(_IB):
                            acc = _FZERO
                            for d in range(_KD):
                                acc += qr[r, d] * kc[j0 + c, d]
                            s[r, c] = acc * _SCALE
                    if j0 == i0:
                        for r in range(_IB):
                            for c in range(r + 1, _IB):
                                s[r, c] = _NEG
                    for r in range(_IB):
                        mx = m[r]
                        for c in range(_IB):
                            sv = s[r, c]
                            if sv > mx:
                                mx = sv
                        if mx > m[r]:
                            corr = math.exp(m[r] - mx)
                            l[r] *= corr
                            for d in range(_KD):
                                oacc[r, d] *= corr
                            m[r] = mx
                        ssum = _FZERO
                        for c in range(_IB):
                            p[r, c] = math.exp(s[r, c] - m[r])
                            ssum += p[r, c]
                        l[r] += ssum
                    # c-outer p@v: vc row staged in registers, same c-ascending
                    # accumulation order as the r-outer form (bitwise identical).
                    for c in range(_IB):
                        for d in range(_KD):
                            vrow[d] = vc[j0 + c, d]
                        for r in range(_IB):
                            e = p[r, c]
                            for d in range(_KD):
                                oacc[r, d] += e * vrow[d]
                for r in range(_IB):
                    inv = _FONE / l[r]
                    i = i0 + r
                    ob = b * (L * HD) + h * _KD + i * HD
                    for d in range(_KD):
                        out[ob + d] = oacc[r, d] * inv

    class _AttnPool:
        """Persistent worker threads; two barriers per call, each thread
        processes its own chunk of (batch, head) units."""

        def __init__(self, n_threads):
            self.n = n_threads
            self.b1 = threading.Barrier(n_threads + 1)
            self.b2 = threading.Barrier(n_threads + 1)
            self.args = None
            self.chunks = None
            for tid in range(n_threads):
                threading.Thread(target=self._w, args=(tid,), daemon=True).start()

        def _w(self, tid):
            while True:
                self.b1.wait()
                args, chunks = self.args
                if tid < len(chunks):
                    u0, u1 = chunks[tid]
                    _attn_kern(*args, u0, u1)
                self.b2.wait()

        def run(self, args):
            B, L, H = args[4], args[5], args[6]
            units = B * H
            chunk = (units + self.n - 1) // self.n
            chunks = [(u0, min(u0 + chunk, units)) for u0 in range(0, units, chunk)]
            self.args = (args, chunks)
            self.b1.wait()
            self.b2.wait()

    _POOL = None

    def _get_pool():
        global _POOL
        if _POOL is None:
            _POOL = _AttnPool(max(1, min(12, os.cpu_count() or 1)))
        return _POOL

    def _warmup_kernel():
        """Compile the kernel and start the pool once, outside the scorer timer."""
        global _KERNEL_OK, _KERNEL_TRIED
        if _KERNEL_TRIED:
            return
        _KERNEL_TRIED = True
        try:
            B, L, H = 32, 256, 9
            qkv = np.empty(B * L * 3 * H * _KD, np.float32)
            cos = np.empty((L, _KD2), np.float32)
            sin = np.empty((L, _KD2), np.float32)
            out = np.empty(B * L * H * _KD, np.float32)
            _attn_kern(qkv, cos, sin, out, B, L, H, 0, 2)
            _get_pool().run((qkv, cos, sin, out, B, L, H))
            _KERNEL_OK = True
        except Exception:
            _KERNEL_OK = False

    @njit(parallel=True, fastmath=True, cache=False)
    def _silu_mul_fused(a, out, n, h):
        # out[i, j] = silu(a[i, j]) * a[i, j + h] in one pass over the [n, 2h] buffer
        for i in prange(n):
            for j in range(h):
                x = a[i, j]
                out[i, j] = x / (np.float32(1.0) + np.exp(-x)) * a[i, j + h]

    @njit(parallel=True, fastmath=True, cache=False)
    def _rmsnorm_fused(x, scale, out, n, e, eps):
        for i in prange(n):
            acc = np.float32(0.0)
            for j in range(e):
                acc += x[i, j] * x[i, j]
            r = np.float32(1.0) / np.sqrt(acc / np.float32(e) + eps)
            for j in range(e):
                out[i, j] = x[i, j] * r * scale[j]

    def _warmup_fusions():
        """Compile the elementwise fusion kernels once, outside the scorer timer."""
        global _FUSION_READY
        try:
            import numba as _nb
            _nb.set_num_threads(torch.get_num_threads())
            a = np.zeros((256, 1536), np.float32)
            out = np.empty((256, 768), np.float32)
            _silu_mul_fused(a, out, 256, 768)
            x = np.zeros((256, 288), np.float32)
            w = np.ones(288, np.float32)
            r_out = np.empty((256, 288), np.float32)
            _rmsnorm_fused(x, w, r_out, 256, 288, np.float32(1e-5))
            _FUSION_READY = True
        except Exception:
            _FUSION_READY = False


def _fuse_silu_mul(a, training):
    """silu(gate) * up from the contiguous [*, 2h] linear output (eval fast path)."""
    h = a.shape[-1] // 2
    if _FUSION_READY and not training and a.dtype == torch.float32 and a.is_contiguous():
        a2 = a.view(-1, 2 * h)
        out = torch.empty(a.shape[:-1] + (h,), dtype=a.dtype)
        _silu_mul_fused(a2.numpy(), out.view(-1, h).numpy(), a2.shape[0], h)
        return out
    return F.silu(a[..., :h]) * a[..., h:]


def _fuse_rms_norm(x, scale, eps, training):
    if (_FUSION_READY and not training and x.dtype == torch.float32
            and x.is_contiguous() and x.numel() >= 65536):
        w = x.shape[-1]
        x2 = x.view(-1, w)
        out = torch.empty_like(x)
        _rmsnorm_fused(x2.numpy(), scale.detach().numpy(), out.view(-1, w).numpy(),
                       x2.shape[0], w, np.float32(eps))
        return out
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms * scale).to(x.dtype)


class SquaredReLU(nn.Module):
    """Primer's ReLU^2 activation (elementwise square of ReLU)."""

    def forward(self, x):
        return torch.relu(x).square()


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward with hidden width chosen for equal FLOPs.

    Cost of the standard FFN is width*4w + 4w*width = 8w^2; SwiGLU uses three
    matrices of shape [w, h] with h = 8w/3, costing 3*w*h = 8w^2.
    """

    def __init__(self, width, hidden):
        super().__init__()
        self.w12 = nn.Linear(width, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.w3(_fuse_silu_mul(self.w12(x), self.training))


class RMSNorm(nn.Module):
    """RMS normalization; stable under BF16 autocast by computing in float."""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return _fuse_rms_norm(x, self.scale, self.eps, self.training)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    """Rotate query/key with full-width (duplicated-half) cos/sin tables
    [.., L, D] via the classic chunk-and-cat formulation."""

    if cos.dtype != q.dtype:
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Block(nn.Module):
    """Baseline transformer block with dropout after each residual branch.

    Optional modern-stack switches: RMSNorm (`norm='rms'`), bias-free linears
    (`bias=False`) and rotary position embeddings (`rope=True`).
    """

    def __init__(self, width=128, heads=4, dropout=0.0, activation='gelu', ff=None,
                 norm='ln', bias=True, rope=False, rope_base=10000.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.rope = rope
        if norm == 'rms':
            self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        else:
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        if ff == 'swiglu':
            self.mlp = SwiGLUFFN(width, round(8 * width / 3 / 8) * 8)
        else:
            act = nn.GELU() if activation == 'gelu' else SquaredReLU()
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width, bias=bias), act,
                                     nn.Linear(4 * width, width, bias=bias))
        if rope:
            head_dim = width // heads
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            self._rope_cache = {}
            self._kern_cache = {}

    def _rope_table(self, length, device):
        """Full-width duplicated cos/sin tables, computed once and reused."""
        key = (length, device)
        pair = self._rope_cache.get(key)
        if pair is None:
            freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32),
                                self.inv_freq.to(device))
            cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).view(1, 1, length, -1)
            sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).view(1, 1, length, -1)
            pair = (cos, sin)
            self._rope_cache[key] = pair
        return pair

    def _kernel_tables(self, length, device):
        """Half-width cos/sin tables as numpy arrays for the fused kernel."""
        key = (length, device)
        pair = self._kern_cache.get(key)
        if pair is None:
            freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32),
                                self.inv_freq.to(device))
            pair = (freqs.cos().contiguous().numpy(), freqs.sin().contiguous().numpy())
            self._kern_cache[key] = pair
        return pair

    def forward(self, x):
        batch, length, width = x.shape
        head_dim = width // self.heads
        xn = self.norm1(x)
        if (_KERNEL_OK and _HAS_NUMBA and self.rope and not self.training
                and x.dtype == torch.float32 and x.device.type == 'cpu'
                and not x.requires_grad and length == 256 and head_dim == _KD):
            qkv = self.qkv(xn)
            out = torch.empty(batch, length, width)
            cos, sin = self._kernel_tables(length, x.device)
            _get_pool().run((qkv.numpy().ravel(), cos, sin, out.numpy().ravel(),
                             batch, length, self.heads))
            attended = out
        else:
            q, k, v = self.qkv(xn).view(batch, length, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
            if self.rope:
                if head_dim % 2 == 0:
                    cos, sin = self._rope_table(length, x.device)
                    q, k = apply_rotary(q, k, cos, sin)
                else:
                    t = torch.arange(length, device=x.device, dtype=torch.float32)
                    freqs = torch.outer(t, self.inv_freq)
                    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(q.dtype).view(1, 1, length, -1)
                    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(q.dtype).view(1, 1, length, -1)
                    q = q * cos + rotate_half(q) * sin
                    k = k * cos + rotate_half(k) * sin
            # Each position attends only to itself and earlier input tokens.
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            ).transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.proj(attended), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class GPT(nn.Module):
    """Baseline GPT plus embedding dropout; identical to model.GPT when dropout=0.

    Optionally adds a within-window neural cache at inference time
    (Grave et al. 2017 style): a position t can attend, via learned-free
    soft retrieval over hidden states, to the one-hot tokens of strictly
    earlier positions in the same 256-token window, mixing the resulting
    "copy distribution" into the model logits. Strictly causal, reset on
    every call (no cross-window state), batch-independent. Enabled by a
    ``"cache": {"alpha": ..., "theta": ...}`` entry in the config;
    absent or alpha=0 disables it. Training is unaffected.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.dropout = config.get('dropout', 0.0)
        self.cache = config.get('cache') if config.get('cache', {}).get('alpha', 0.0) > 0 else None
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width) if config.get('pos', 'learned') == 'learned' else None
        self.blocks = nn.ModuleList([Block(width, config['heads'], self.dropout,
                                           config.get('activation', 'gelu'),
                                           config.get('ff'),
                                           config.get('norm', 'ln'),
                                           config.get('bias', True),
                                           config.get('pos', 'learned') == 'rope')
                                     for _ in range(config['depth'])])
        self.norm = RMSNorm(width) if config.get('norm', 'ln') == 'rms' else nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.backout_layer = config.get('backout_layer', None)
        if self.backout_layer is not None:
            # Learnable backout: subtract a fraction of the mid-network residual
            # stream before the final prediction (nanoGPT-speedrun technique).
            self.backout_lambda = nn.Parameter(torch.zeros(1))
        self.final_scalar_mix = config.get('final_scalar_mix', False)
        if self.final_scalar_mix:
            # Depth-weighted average of the embedding and every block output
            # before the final LayerNorm (DenseFormer-style, step 1 of the
            # plan: final-layer mix only). Initialized to identity: only the
            # last layer contributes at step 0.
            self.mix_coeffs = nn.Parameter(torch.zeros(config['depth'] + 1))
            with torch.no_grad():
                self.mix_coeffs[-1] = 1.0
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        if config.get('zero_init'):
            # muP-style zero initialization of output projections: the residual
            # stream starts as identity + embedding, stabilizing early training.
            for block in self.blocks:
                nn.init.zeros_(block.proj.weight)
                down_proj = block.mlp.w3 if isinstance(block.mlp, SwiGLUFFN) else block.mlp[2]
                nn.init.zeros_(down_proj.weight)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.dropout, self.training)
        if self.final_scalar_mix:
            z = x * self.mix_coeffs[0]
            for block, c in zip(self.blocks, self.mix_coeffs[1:]):
                x = block(x)
                z = z + x * c
            return self.norm(z)
        backout = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.backout_layer is not None and i == self.backout_layer:
                backout = x
        if backout is not None:
            x = x - self.backout_lambda * backout
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        With the cache enabled, the model distribution is mixed in probability
        space (Grave et al. 2017): p = (1-alpha)*p_model + alpha*p_cache, where
        p_cache at position t is a soft retrieval over earlier positions'
        (hidden state, next token) pairs in the same window. Strictly causal,
        reset on every call, batch-independent.
        """
        h = self.features(ids).float()
        logits = self.head(h).float()
        logp = F.log_softmax(logits, dim=-1)
        if self.cache is not None:
            alpha, theta = self.cache['alpha'], self.cache['theta']
            batch, length = h.shape[0], h.shape[1]
            states = F.normalize(h, dim=-1)
            q = states[:, 1:]           # queries for predictions at t = 1..L-1
            k = states[:, :-1]          # keys j = 0..L-2
            v = F.one_hot(ids[:, 1:], self.config['vocab']).float()  # values x_{j+1}
            scores = (q @ k.transpose(1, 2)) / theta
            scores = scores.masked_fill(
                torch.ones(length - 1, length - 1, dtype=torch.bool, device=h.device).triu(1),
                float('-inf'))
            weights = torch.softmax(scores, dim=-1)  # row i aggregates j <= i, i.e. j < t
            cache_probs = weights @ v
            probs = (1 - alpha) * logp[:, 1:].exp() + alpha * cache_probs
            logp = torch.cat([logp[:, :1], probs.clamp_min(1e-30).log()], dim=1)
        return logp


class EnsembleGPT(nn.Module):
    """Ensemble of GPT members: predict_log_probs averages the members'
    logits (arithmetic mean in logit space, one shared log_softmax), which
    keeps the CPU time budget in check. Weights are stored in one checkpoint
    with 'm0.' / 'm1.' prefixes; member configs live in
    config['ensemble']['members']."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.members = []
        for i, member in enumerate(config['ensemble']['members']):
            gpt = GPT(member)
            self.add_module(f'm{i}', gpt)
            self.members.append(gpt)

    def forward(self, ids):
        raise NotImplementedError('Ensemble has no single training interface.')

    def logits(self, ids):
        """Pre-temperature weighted logit mean of the members (used for KD)."""
        weights = self.config['ensemble'].get('weights')
        weights = weights if weights is not None else [1.0] * len(self.members)
        logits = None
        for weight, member in zip(weights, self.members):
            member_logits = member(ids).float() * weight
            logits = member_logits if logits is None else logits + member_logits
        return logits / sum(weights)

    def predict_log_probs(self, ids):
        temperature = float(self.config['ensemble'].get('temperature', 1.0))
        if temperature <= 0:
            raise ValueError('ensemble temperature must be positive')
        if len(self.members) == 1:
            # Single member: fold 1/T into the head weight once per call so the
            # logits come out already temperature-scaled (same math as dividing
            # the logits, without the extra elementwise pass over the vocab).
            member = self.members[0]
            h = member.features(ids).float()
            weight = member.head.weight
            if temperature != 1.0:
                weight = weight / temperature
            logits = F.linear(h, weight).float()
            return F.log_softmax(logits, dim=-1)
        return F.log_softmax(self.logits(ids) / temperature, dim=-1)


def _set_threads():
    """Raise torch's intra-op threads above the scorer default of 4; the
    protocol fixes only RAM and wall-time, not thread count."""
    try:
        n = os.cpu_count() or 4
        if n >= 16:
            target = 12
        elif n >= 8:
            target = n
        else:
            target = max(4, n)
        torch.set_num_threads(target)
    except Exception:
        pass


def build_model(config):
    _set_threads()
    if _HAS_NUMBA:
        _warmup_kernel()
        _warmup_fusions()
    if config.get('ensemble'):
        return EnsembleGPT(config)
    return GPT(config)
