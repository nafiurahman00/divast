"""Narrow a fine-tuned RoBERTa classifier: hidden size 768 -> H, FFN 3072 -> F.

The narrow model is a plain RobertaForSequenceClassification (the organisers'
notebook loads it unchanged) and starts from the teacher's weights instead of
random ones:

* residual stream: projected onto the top-H principal directions P (768 x H) of
  the teacher's token embeddings, i.e. x_s = P^T x;
* attention: the H/64 heads with the largest weight norms are kept;
* feed-forward: the F neurons with the largest weight norms are kept;
* LayerNorm scales: diag(P^T diag(g) P), an approximation, as LayerNorm does not
  commute with the projection.

Distillation from the full teacher then repairs the approximation errors.
"""
from __future__ import annotations

import copy

import torch


def _lin(old: torch.nn.Linear, w: torch.Tensor, b: torch.Tensor | None) -> torch.nn.Linear:
    new = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    new.weight.data.copy_(w)
    if b is not None:
        new.bias.data.copy_(b)
    return new


def _ln(old: torch.nn.LayerNorm, P: torch.Tensor) -> torch.nn.LayerNorm:
    new = torch.nn.LayerNorm(P.shape[1], eps=old.eps)
    new.weight.data.copy_((P ** 2).T @ old.weight.data)   # diag of P^T diag(g) P
    new.bias.data.copy_(P.T @ old.bias.data)
    return new


@torch.no_grad()
def narrow_roberta(model, hidden: int, ffn: int):
    """Return a copy of a RobertaForSequenceClassification with hidden size
    `hidden` (a multiple of the head size) and feed-forward size `ffn`."""
    cfg = copy.deepcopy(model.config)
    d = cfg.hidden_size // cfg.num_attention_heads           # head size, 64
    assert hidden % d == 0 and hidden <= cfg.hidden_size and ffn <= cfg.intermediate_size
    n_heads = hidden // d

    E = model.roberta.embeddings.word_embeddings.weight.data.float()
    # principal directions of the embedding matrix (uncentred, as the residual
    # stream also carries the mean)
    _, _, Vh = torch.linalg.svd(E, full_matrices=False)
    P = Vh[:hidden].T.contiguous()                            # 768 x H, orthonormal columns

    cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size = hidden, n_heads, ffn
    new = type(model)(cfg)

    # embeddings
    emb, nemb = model.roberta.embeddings, new.roberta.embeddings
    nemb.word_embeddings.weight.data.copy_(emb.word_embeddings.weight.data @ P)
    nemb.position_embeddings.weight.data.copy_(emb.position_embeddings.weight.data @ P)
    nemb.token_type_embeddings.weight.data.copy_(emb.token_type_embeddings.weight.data @ P)
    nemb.LayerNorm = _ln(emb.LayerNorm, P)

    for old, nl in zip(model.roberta.encoder.layer, new.roberta.encoder.layer):
        att, sa = old.attention, old.attention.self
        # head importance: norm of its value rows times its output columns
        imp = torch.stack([sa.value.weight[h * d:(h + 1) * d].norm() *
                           att.output.dense.weight[:, h * d:(h + 1) * d].norm()
                           for h in range(sa.value.weight.shape[0] // d)])
        heads = sorted(imp.topk(n_heads).indices.tolist())
        r = torch.cat([torch.arange(h * d, (h + 1) * d) for h in heads])
        for name in ("query", "key", "value"):
            lin = getattr(sa, name)
            setattr(nl.attention.self, name, _lin(lin, lin.weight.data[r] @ P, lin.bias.data[r]))
        o = att.output.dense
        nl.attention.output.dense = _lin(o, P.T @ o.weight.data[:, r], P.T @ o.bias.data)
        nl.attention.output.LayerNorm = _ln(att.output.LayerNorm, P)
        # feed-forward neurons
        w1, w2 = old.intermediate.dense, old.output.dense
        nimp = w1.weight.data.norm(dim=1) * w2.weight.data.norm(dim=0)
        n = nimp.topk(ffn).indices.sort().values
        nl.intermediate.dense = _lin(w1, w1.weight.data[n] @ P, w1.bias.data[n])
        nl.output.dense = _lin(w2, P.T @ w2.weight.data[:, n], P.T @ w2.bias.data)
        nl.output.LayerNorm = _ln(old.output.LayerNorm, P)

    # classification head: dense (H x H) then out_proj (labels x H)
    hd, nhd = model.classifier, new.classifier
    nhd.dense = _lin(hd.dense, P.T @ hd.dense.weight.data @ P, P.T @ hd.dense.bias.data)
    nhd.out_proj = _lin(hd.out_proj, hd.out_proj.weight.data @ P, hd.out_proj.bias.data)
    return new
