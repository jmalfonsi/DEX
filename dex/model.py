"""Trainable neural definition of DEX, with bounded fine corrections.

Randomly initialized. No semantic quality or empirical calibration is implied.
The dense forward computes the reference distribution. Deployment can call fine
modules lazily with dex.refine. The mathematical function is independent of budget.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 32768
    width: int = 384
    depth: int = 8
    heads: int = 6
    ff: int = 1152
    window: int = 128
    rank: int = 128
    rho: float = 1.0
    option_bound: float = 1.0
    scale: float = 8.0


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.width = cfg.heads, cfg.width
        self.n1, self.n2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv, self.out = nn.Linear(cfg.width, 3 * cfg.width), nn.Linear(cfg.width, cfg.width)
        self.gate, self.up = nn.Linear(cfg.width, cfg.ff), nn.Linear(cfg.width, cfg.ff)
        self.down = nn.Linear(cfg.ff, cfg.width)

    def forward(self, x, valid):
        b, t, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
        score = score.masked_fill(~valid[:, None, None, :], -1e4)
        a = torch.softmax(score, dim=-1)
        x = x + self.out((a @ v).transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LocalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width, padding_idx=0)
        self.position = nn.Embedding(cfg.window, cfg.width)
        self.kind = nn.Embedding(8, cfg.width)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)
        self.project = nn.Linear(cfg.width, cfg.rank)
        self.pool = nn.Linear(cfg.rank, 1)

    def forward(self, ids, kinds):
        # Every sequence must contain at least one non-pad token, checked by caller.
        valid = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.position(pos)[None] + self.kind(kinds)[:, None]
        for layer in self.layers:
            x = layer(x, valid)
        tokens = self.project(self.norm(x))
        weights = torch.softmax(self.pool(tokens).squeeze(-1).masked_fill(~valid, -1e4), -1)
        return tokens, (weights.unsqueeze(-1) * tokens).sum(1)


class DecisionCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        r = cfg.rank
        self.layout = nn.Linear(4, r, bias=False)
        self.route = nn.Linear(r, r, bias=False)
        self.coarse = nn.Sequential(nn.Linear(2*r, 2*r), nn.SiLU(), nn.Linear(2*r, r))
        self.fine_query = nn.Linear(2*r, r)
        self.state_delta = nn.Sequential(nn.Linear(3*r, 2*r), nn.SiLU(), nn.Linear(2*r, r))
        self.option_delta = nn.Sequential(nn.Linear(3*r, r), nn.SiLU(), nn.Linear(r, 1))

    def prepare(self, question, summaries):
        # One state, F questions, C chunks. No attention between questions.
        # Cheap layout features are applied AFTER cached local encoding.
        pos = torch.arange(summaries.shape[0], device=summaries.device, dtype=summaries.dtype)
        layout = torch.stack((torch.log1p(pos), torch.log1p(summaries.shape[0] - pos),
                              torch.sin(pos / 100), torch.cos(pos / 100)), -1)
        summaries = summaries + self.layout(layout)
        weights = torch.softmax(self.route(question) @ summaries.T / math.sqrt(self.cfg.rank), -1)
        context = weights @ summaries
        coarse = self.coarse(torch.cat((question, context), -1))
        return coarse, weights

    def refine_state(self, question, coarse, tokens, valid):
        # Paired batches: each row is one (field, chunk) refinement.
        q = self.fine_query(torch.cat((question, coarse), -1))
        att = (tokens * q[:, None]).sum(-1) / math.sqrt(self.cfg.rank)
        att = torch.softmax(att.masked_fill(~valid, -1e4), -1)
        read = (att[..., None] * tokens).sum(1)
        raw = self.state_delta(torch.cat((question, coarse, read), -1))
        # ||delta||_2 <= rho, for ANY weights. Never learned from a promise.
        return self.cfg.rho * torch.tanh(raw) / math.sqrt(self.cfg.rank)

    def refine_options(self, question, coarse, bank):
        return self.cfg.option_bound * torch.tanh(
            self.option_delta(torch.cat((question, coarse, bank), -1)).squeeze(-1))

    def forward(self, question, summaries, tokens, valid, bank):
        coarse, weights = self.prepare(question, summaries)
        f, c, n = question.shape[0], summaries.shape[0], bank.shape[0]
        r, t = self.cfg.rank, tokens.shape[1]
        delta = self.refine_state(
            question[:, None].expand(f, c, r).reshape(-1, r),
            coarse[:, None].expand(f, c, r).reshape(-1, r),
            tokens[None].expand(f, c, t, r).reshape(-1, t, r),
            valid[None].expand(f, c, t).reshape(-1, t)).reshape(f, c, r)
        latent = coarse + (weights[..., None] * delta).sum(1)
        correction = self.refine_options(
            question[:, None].expand(f, n, r).reshape(-1, r),
            coarse[:, None].expand(f, n, r).reshape(-1, r),
            bank[None].expand(f, n, r).reshape(-1, r)).reshape(f, n)
        return self.cfg.scale * latent @ bank.T + correction


class DEX(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        if (self.cfg.width % self.cfg.heads or self.cfg.window < 1 or
            self.cfg.rank < 1 or self.cfg.rho < 0 or self.cfg.option_bound < 0):
            raise ValueError("invalid model config")
        self.encoder = LocalEncoder(self.cfg)
        self.core = DecisionCore(self.cfg)

    def forward(self, state_ids, question_ids, option_ids):
        device = state_ids.device
        tokens, summaries = self.encoder(state_ids, torch.zeros(len(state_ids), dtype=torch.long, device=device))
        _, questions = self.encoder(question_ids, torch.ones(len(question_ids), dtype=torch.long, device=device))
        _, options = self.encoder(option_ids, torch.full((len(option_ids),), 2, dtype=torch.long, device=device))
        bank = F.normalize(options, dim=-1)
        return self.core(questions, summaries, tokens, state_ids != 0, bank)


def training_loss(logits, targets, *, teacher_probabilities=None, teacher_weight=0.0):
    """Proper log score. Teacher disagreement is not a correctness probability."""
    loss = F.cross_entropy(logits, targets)
    if teacher_probabilities is not None and teacher_weight > 0:
        loss = loss + teacher_weight * F.kl_div(
            F.log_softmax(logits, -1), teacher_probabilities, reduction="batchmean")
    return loss
