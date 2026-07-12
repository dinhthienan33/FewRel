import sys
sys.path.append('..')
import math

import torch
from torch import nn
from torch.nn import functional as F

import fewshot_re_kit


def _best_num_heads(dim, preferred=4):
    if dim <= 0:
        return 1
    preferred = max(1, int(preferred))
    if dim % preferred == 0:
        return preferred
    for h in range(preferred, 0, -1):
        if dim % h == 0:
            return h
    return 1


class AGGCNLayer(nn.Module):
    """Attention-guided graph convolution (dense, batched, pure PyTorch)."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = _best_num_heads(dim, preferred=num_heads)
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.mix_gate = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _attention_adj(self, x, adj_mask):
        # x: (B, L, D) -> attention adjacency (B, L, L)
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        scores = scores.mean(dim=1)  # (B, L, L)
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(adj_mask <= 0, neg)
        return torch.softmax(scores, dim=-1)

    def forward(self, x, base_adj):
        adj_mask = (base_adj > 0).to(x.dtype)
        attn_adj = self._attention_adj(x, adj_mask)
        gate = torch.sigmoid(self.mix_gate)
        guided = gate * base_adj + (1.0 - gate) * attn_adj
        guided = guided / guided.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        msg = torch.matmul(guided, x)
        out = self.dropout(F.gelu(self.msg_proj(msg)))
        return self.norm(out + x)


class EdgeMasking(nn.Module):
    """VGIB-style soft edge masking via a low-rank bilinear score.

    Cheap dense variant of the DocRED EdgeMaskingLayer: instead of an MLP over
    concatenated endpoint features (O(L^2 * 2D) memory), we score edges with a
    bilinear form over a small projection, then apply a Bernoulli KL prior.
    """

    def __init__(self, dim, rank=64, tau=0.5, prior=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, min(rank, dim))
        self.tau = float(tau)
        self.prior = float(prior)

    def forward(self, x, adj_mask, training=True):
        p = self.proj(x)
        logits = torch.matmul(p, p.transpose(-1, -2)) / math.sqrt(float(p.size(-1)))
        if training:
            u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel = torch.log(u) - torch.log(1.0 - u)
            s = torch.sigmoid((logits + gumbel) / max(1e-3, self.tau))
        else:
            s = torch.sigmoid(logits)
        s = s * adj_mask

        eps = 1e-6
        s_c = s.clamp(eps, 1.0 - eps)
        prior = min(1.0 - eps, max(eps, self.prior))
        kl = (
            s_c * torch.log(s_c / prior)
            + (1.0 - s_c) * torch.log((1.0 - s_c) / (1.0 - prior))
        )
        edge_count = adj_mask.sum().clamp_min(1.0)
        kl = (kl * adj_mask).sum() / edge_count
        return s, kl


class GraphExpert(nn.Module):
    """AGGCN-based graph expert with dense-connected layers and optional VGIB.

    Returns ``(pair_embedding, gib_loss)`` where ``pair_embedding`` is the
    concatenation of the head/tail node embeddings, shape ``(B, out_dim * 2)``.
    """

    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=4, dropout=0.1,
                 use_vgib=True, gib_tau=0.5, gib_prior=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.pre_norm = nn.LayerNorm(out_dim)
        self.num_layers = max(1, int(num_layers))
        self.dense_fuse = nn.ModuleList(
            [nn.Linear(out_dim * (i + 1), out_dim) for i in range(self.num_layers)]
        )
        self.layers = nn.ModuleList(
            [AGGCNLayer(out_dim, num_heads=num_heads, dropout=dropout) for _ in range(self.num_layers)]
        )
        self.use_vgib = bool(use_vgib)
        if self.use_vgib:
            self.edge_mask = EdgeMasking(out_dim, tau=gib_tau, prior=gib_prior)

    def forward(self, x, base_adj, e1_idx, e2_idx):
        h = self.pre_norm(self.in_proj(x))
        adj_mask = (base_adj > 0).to(h.dtype)

        gib = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.use_vgib:
            s, gib = self.edge_mask(h, adj_mask, training=self.training)
            adj = base_adj * s
        else:
            adj = base_adj

        states = [h]
        for i, layer in enumerate(self.layers):
            fused = self.dense_fuse[i](torch.cat(states, dim=-1))
            states.append(layer(fused, adj))
        h = states[-1]

        b_idx = torch.arange(h.size(0), device=h.device)
        e1 = h[b_idx, e1_idx]
        e2 = h[b_idx, e2_idx]
        return torch.cat([e1, e2], dim=-1), gib


def switch_load_balance_loss(router_logits, top_idx, num_experts):
    probs = F.softmax(router_logits.float(), dim=-1)
    importance = probs.sum(0)
    load = torch.bincount(top_idx, minlength=num_experts).float().to(router_logits.device)
    importance_norm = importance / (importance.sum() + 1e-9)
    load_norm = load / (load.sum() + 1e-9)
    out = torch.var(importance_norm) + torch.var(load_norm)
    if not torch.isfinite(out):
        return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)
    return out


class MoEGraphProto(fewshot_re_kit.framework.FewShotREModel):
    """Few-shot relation classifier: LLM encoder -> token graph -> AGGCN/VGIB
    graph experts routed by a top-1 MoE -> prototypical matching.

    The DocRED variant used a fixed classifier head over known relations. That
    does not transfer to unseen few-shot relations, so here the graph + MoE
    produces a pair embedding and classification is done by matching queries to
    per-class support prototypes (Prototypical-Networks style).
    """

    def __init__(self, sentence_encoder, num_experts=4, expert_dim=128,
                 aggcn_layers=2, aggcn_heads=4, dropout=0.1, use_vgib=True,
                 use_moe=True, gib_tau=0.5, gib_prior=0.1, noise_scale=1e-2,
                 gib_weight=1e-3, bal_weight=1e-2, dot=False):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder  # NOT DataParallel (LLM lives on one device)
        self.cost = nn.CrossEntropyLoss()

        hidden_size = sentence_encoder.hidden_size
        self.expert_dim = expert_dim
        self.num_experts = num_experts if use_moe else 1
        self.use_moe = bool(use_moe)
        self.noise_scale = noise_scale
        self.gib_weight = gib_weight
        self.bal_weight = bal_weight
        self.dot = dot

        self.experts = nn.ModuleList([
            GraphExpert(hidden_size, expert_dim, num_layers=aggcn_layers,
                        num_heads=aggcn_heads, dropout=dropout, use_vgib=use_vgib,
                        gib_tau=gib_tau, gib_prior=gib_prior)
            for _ in range(self.num_experts)
        ])
        self.router = nn.Linear(hidden_size * 2, self.num_experts)
        self.drop = nn.Dropout(dropout)

        self.last_gib_loss = torch.zeros(())
        self.last_bal_loss = torch.zeros(())

    def _build_adjacency(self, mask, e1_idx, e2_idx):
        # mask: (B, L). Chain edges over valid tokens + star edges from markers.
        B, L = mask.shape
        valid = (mask > 0).float()
        adj = torch.zeros(B, L, L, device=mask.device, dtype=torch.float32)

        both = valid[:, :-1] * valid[:, 1:]
        idx = torch.arange(L - 1, device=mask.device)
        adj[:, idx, idx + 1] = both
        adj[:, idx + 1, idx] = both

        b_idx = torch.arange(B, device=mask.device)
        adj[b_idx, e1_idx] = valid
        adj[b_idx, :, e1_idx] = valid
        adj[b_idx, e2_idx] = valid
        adj[b_idx, :, e2_idx] = valid

        adj = adj * valid.unsqueeze(1) * valid.unsqueeze(2)
        eye = torch.eye(L, device=mask.device).unsqueeze(0)
        adj = adj + eye * valid.unsqueeze(2)
        return adj

    def _pair_embed(self, inputs):
        hidden = self.sentence_encoder(inputs)  # (B, L, H)
        mask = inputs["mask"]
        e1_idx = inputs["pos1"].long().clamp(0, hidden.size(1) - 1)
        e2_idx = inputs["pos2"].long().clamp(0, hidden.size(1) - 1)

        adj = self._build_adjacency(mask, e1_idx, e2_idx)
        B = hidden.size(0)
        b_idx = torch.arange(B, device=hidden.device)
        pair_raw = torch.cat([hidden[b_idx, e1_idx], hidden[b_idx, e2_idx]], dim=-1)

        if not self.use_moe:
            emb, gib = self.experts[0](hidden, adj, e1_idx, e2_idx)
            return self.drop(emb), gib, torch.zeros((), device=hidden.device)

        router_logits = self.router(pair_raw)
        if self.training and self.noise_scale > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_scale
        probs = F.softmax(router_logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)

        out_dim = self.expert_dim * 2
        pair_emb = torch.zeros(B, out_dim, device=hidden.device, dtype=torch.float32)
        gib_total = torch.zeros((), device=hidden.device)
        for e in range(self.num_experts):
            sel = (top_idx == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            emb, gib = self.experts[e](hidden[sel], adj[sel], e1_idx[sel], e2_idx[sel])
            pair_emb[sel] = emb * top_prob[sel].unsqueeze(-1)
            gib_total = gib_total + gib
        bal = switch_load_balance_loss(router_logits, top_idx, self.num_experts)
        return self.drop(pair_emb), gib_total, bal

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        support_emb, s_gib, s_bal = self._pair_embed(support)
        query_emb, q_gib, q_bal = self._pair_embed(query)
        dim = support_emb.size(-1)

        support_emb = support_emb.view(-1, N, K, dim)
        query_emb = query_emb.view(-1, total_Q, dim)

        proto = torch.mean(support_emb, 2)  # (B, N, D)
        logits = self.__batch_dist__(proto, query_emb)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # NOTA column
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        self.last_gib_loss = s_gib + q_gib
        self.last_bal_loss = s_bal + q_bal
        return logits, pred

    def loss(self, logits, label):
        n = logits.size(-1)
        ce = self.cost(logits.view(-1, n), label.view(-1))
        aux = self.gib_weight * self.last_gib_loss + self.bal_weight * self.last_bal_loss
        return ce + aux.to(ce.device)
