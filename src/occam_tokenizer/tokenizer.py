"""
Occam Tokenizer
===============

Parameter-free unsupervised tokenization by Minimum Description Length.

There is no vocabulary size to choose. Merging stops when no candidate merge
lowers the description length of the corpus.

    from occam_tokenizer import OccamTokenizer

    tok = OccamTokenizer()
    tok.train(documents)

    enc = tok.encode("hello world")
    enc.ids, enc.tokens

    tok.save("tokenizer.json")
    tok = OccamTokenizer.from_file("tokenizer.json")

The criterion
-------------
The corpus is modelled as one multinomial per document over the V token types.
Writing n_dv for the count of token v in document d, K_d for the length of
document d, N for the character count and Vc for the constructed vocabulary
size, the code length being minimised is

    C = log N                                     prior on V
      + 2 [lnGamma(Vc) - lnGamma(|A|)]            vocabulary construction
      + sum_d [ lnGamma(K_d + V) - lnGamma(V) ]   per-document prior + ordering
      - sum_dv lnGamma(n_dv + 1)                  multinomial likelihood

The middle line collapses a stars-and-bars prior and its enumerative
likelihood, since log C(n+V-1, V-1) + log(n!) = lnGamma(n+V) - lnGamma(V).

Conditioning on the document is what keeps the vocabulary honest. Every entry
costs roughly D log(1 + K_d/V) -- paid in *every* document, earned only in the
ones where the token occurs. A token spread across the corpus repays that; a
proper noun confined to one document does not. Without it the criterion learns
literals: 'caius*marcius*coriolanus' rather than the morphemes around it.

Reference: Boullé, Clérot & Hue, "Revisiting enumerative two-part crude MDL
for Bernoulli and multinomial distributions", arXiv:1608.05522.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.special import gammaln

from .__about__ import __version__

__all__ = ["OccamTokenizer", "Encoding", "SPACE"]

SPACE = "\u2581"          # visible word-boundary marker, SentencePiece style


# --------------------------------------------------------------------------- #
#  Encoding result
# --------------------------------------------------------------------------- #

@dataclass
class Encoding:
    """Result of encoding one text."""

    ids: list[int]
    tokens: list[str]
    attention_mask: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.attention_mask:
            self.attention_mask = [1] * len(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"Encoding(len={len(self.ids)}, tokens={self.tokens[:8]}...)"


# --------------------------------------------------------------------------- #
#  Internals
# --------------------------------------------------------------------------- #

def _dedup(p: np.ndarray) -> np.ndarray:
    """
    Greedy left-to-right de-overlap of sorted positions.

    Only matters when merging a token with itself: "sss" contains two (s,s)
    adjacencies but admits one merge. Counting both would make a token's
    remaining count go negative. In a run of consecutive positions the greedy
    scan keeps every other one, so run index parity reproduces it exactly.
    """
    if len(p) < 2:
        return p
    brk = np.empty(len(p), dtype=bool)
    brk[0] = True
    brk[1:] = np.diff(p) > 1
    run = np.cumsum(brk) - 1
    starts = np.concatenate(([0], np.cumsum(np.bincount(run))[:-1]))
    return p[(np.arange(len(p)) - starts[run]) % 2 == 0]


class _Cells:
    """
    Sparse (document x token) counts, held as sorted parallel arrays keyed on

        key = token * D + document

    Token-major packing makes a column a contiguous slice, a point lookup one
    vectorised searchsorted, and -- because a new token always receives the
    largest id so far -- a new column an append rather than a rebuild. That
    last property is why this is not CSC: merging changes the sparsity
    structure at every step.

    Memory is O(nnz) with nnz <= K, against O(D * V) for a dense table.
    """

    __slots__ = ("D", "maxv", "_k", "_v", "_n", "tstart")

    def __init__(self, seq: np.ndarray, dseq: np.ndarray, D: int, maxv: int):
        self.D, self.maxv = D, maxv
        k, v = np.unique(seq.astype(np.int64) * D + dseq.astype(np.int64),
                         return_counts=True)
        self._n = len(k)
        cap = max(self._n * 2, 1024)
        self._k = np.zeros(cap, dtype=np.int64); self._k[:self._n] = k
        self._v = np.zeros(cap, dtype=np.int64); self._v[:self._n] = v
        self.tstart = np.searchsorted(k, np.arange(maxv + 1, dtype=np.int64) * D)

    @property
    def keys(self): return self._k[:self._n]

    @property
    def vals(self): return self._v[:self._n]

    @property
    def nbytes(self): return self._k.nbytes + self._v.nbytes + self.tstart.nbytes

    def lookup(self, doc: np.ndarray, tok: np.ndarray) -> np.ndarray:
        """n_dv for a batch of (document, token) pairs; 0 where absent."""
        if len(doc) == 0:
            return np.zeros(0, dtype=np.float64)
        key = tok.astype(np.int64) * self.D + doc.astype(np.int64)
        i = np.minimum(np.searchsorted(self.keys, key), self._n - 1)
        return np.where(self.keys[i] == key, self._v[i], 0).astype(np.float64)

    def sum_lgamma(self) -> float:
        nz = self.vals[self.vals > 0].astype(np.float64)
        return float(gammaln(nz + 1.0).sum())

    def sub_column(self, v_id: int, md: np.ndarray) -> None:
        lo, hi = int(self.tstart[v_id]), int(self.tstart[v_id + 1])
        if hi > lo:
            self._v[lo:hi] -= md[self._k[lo:hi] - v_id * self.D]

    def add_column(self, c_id: int, md: np.ndarray) -> None:
        nz = np.flatnonzero(md)
        old = self._n
        if len(nz):
            need = old + len(nz)
            if need > len(self._k):
                cap = max(need, len(self._k) * 2)
                for name in ("_k", "_v"):
                    a = np.zeros(cap, dtype=np.int64)
                    a[:old] = getattr(self, name)[:old]
                    setattr(self, name, a)
            self._k[old:need] = c_id * self.D + nz
            self._v[old:need] = md[nz]
            self._n = need
        self.tstart[c_id] = old
        self.tstart[c_id + 1:] = self._n

    def grow(self, maxv: int) -> None:
        if maxv > self.maxv:
            t = np.full(maxv + 1, self._n, dtype=np.int64)
            t[:len(self.tstart)] = self.tstart
            self.tstart, self.maxv = t, maxv

    def compact(self) -> None:
        """Reclaim zeroed cells once they exceed half the stored entries."""
        live = self.vals > 0
        if live.sum() * 2 >= self._n:
            return
        k, v = self.keys[live].copy(), self.vals[live].copy()
        cap = max(len(k) * 2, 1024)
        self._k = np.zeros(cap, dtype=np.int64); self._k[:len(k)] = k
        self._v = np.zeros(cap, dtype=np.int64); self._v[:len(k)] = v
        self._n = len(k)
        self.tstart = np.searchsorted(
            k, np.arange(self.maxv + 1, dtype=np.int64) * self.D)


# --------------------------------------------------------------------------- #
#  Tokenizer
# --------------------------------------------------------------------------- #

class OccamTokenizer:
    """
    Parameter-free MDL tokenizer.

    Parameters
    ----------
    max_token_length : int, optional
        Cap on merged token length in characters. ``None`` for no cap.
    candidates : int
        Merge candidates scored exactly each step. Higher searches wider at
        proportional cost; 12 is ample in practice.
    unk_token, pad_token : str
        Special tokens. Both are assigned ids ahead of the learned vocabulary.
    lowercase : bool
        Lowercase during training and encoding.
    """

    def __init__(
        self,
        max_token_length: int | None = 25,
        candidates: int = 12,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        lowercase: bool = False,
    ):
        self.max_token_length = max_token_length
        self.candidates = candidates
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase

        self.merges: list[tuple[str, str]] = []
        self.alphabet: list[str] = []
        self._vocab: dict[str, int] = {}
        self._ids: list[str] = []
        self._ranks: dict[tuple[str, str], int] = {}
        self._cache: dict[str, list[str]] = {}
        self.training_stats: dict = {}

    # -- representation ---------------------------------------------------- #

    def __repr__(self) -> str:
        n = self.vocab_size
        return (f"OccamTokenizer(vocab_size={n}, merges={len(self.merges)}, "
                f"trained={bool(self.merges) or n > 0})")

    # -- training ---------------------------------------------------------- #

    def train(self, documents: Sequence[str], max_merges: int = 100_000,
              verbose: bool = False) -> "OccamTokenizer":
        """
        Learn a vocabulary from ``documents``.

        Document boundaries are load-bearing, not cosmetic: the criterion
        charges each vocabulary entry across every document, so how the corpus
        is split determines what counts as a corpus-wide token. Passing one
        giant string collapses that signal entirely.
        """
        docs = [d.lower() for d in documents] if self.lowercase else list(documents)
        docs = [d for d in docs if d and d.strip()]
        if not docs:
            raise ValueError("no non-empty documents to train on")

        chars = sorted({c for d in docs for c in self._pre(d)})
        self.alphabet = chars
        c2i = {c: i for i, c in enumerate(chars)}
        A = len(chars)

        seq_parts, dseq_parts = [], []
        for i, d in enumerate(docs):
            t = self._pre(d)
            seq_parts.append(np.fromiter((c2i[c] for c in t), np.int32, len(t)))
            dseq_parts.append(np.full(len(t), i, dtype=np.int32))
        seq = np.concatenate(seq_parts)
        dseq = np.concatenate(dseq_parts)
        del seq_parts, dseq_parts

        D = len(docs)
        N = len(seq)                       # characters (one char per symbol now)
        maxv = max(512, A * 4)
        tok_len = np.zeros(maxv, dtype=np.int32)
        tok_len[:A] = 1
        tok_cnt = np.bincount(seq, minlength=maxv).astype(np.int64)
        Kd = np.bincount(dseq, minlength=D).astype(np.int64)
        cells = _Cells(seq, dseq, D, maxv)
        id2tok = list(chars)
        nid = A

        Kdf = Kd.astype(np.float64)
        V = float((tok_cnt[:nid] > 0).sum())
        cost = (math.log(N) + self._lvoc(nid, A)
                + float((gammaln(Kdf + V) - gammaln(V)).sum())
                - cells.sum_lgamma())
        history = [cost]

        for step in range(max_merges):
            cand = self._screen(seq, dseq, tok_cnt, tok_len, nid, N)
            if cand is None:
                break
            a_ids, b_ids = cand
            d = self._deltas(seq, dseq, cells, Kdf, D, a_ids, b_ids,
                             nid, V, tok_cnt)
            i = int(np.argmin(d))
            if not np.isfinite(d[i]) or d[i] >= 0.0:
                break

            a, b = int(a_ids[i]), int(b_ids[i])
            c_id = nid
            if c_id >= maxv:
                maxv *= 2
                tok_len = np.resize(tok_len, maxv); tok_len[c_id:] = 0
                tok_cnt = np.resize(tok_cnt, maxv); tok_cnt[c_id:] = 0
                cells.grow(maxv)
            id2tok.append(id2tok[a] + id2tok[b])
            tok_len[c_id] = tok_len[a] + tok_len[b]
            nid += 1

            seq, dseq, md = self._merge(seq, dseq, a, b, c_id, D)
            cells.sub_column(a, md); cells.sub_column(b, md)
            cells.add_column(c_id, md); cells.compact()
            n = int(md.sum())
            tok_cnt[c_id] = n; tok_cnt[a] -= n; tok_cnt[b] -= n
            Kd -= md; Kdf = Kd.astype(np.float64)
            V = float((tok_cnt[:nid] > 0).sum())

            self.merges.append((id2tok[a], id2tok[b]))
            cost += float(d[i])
            history.append(cost)
            if verbose and step % 25 == 0:
                print(f"  merge {step:>4}  {id2tok[a]+id2tok[b]!r:<24} "
                      f"C={cost:.1f}")

        self._build_vocab()
        self.training_stats = {
            "documents": D,
            "characters": N,
            "merges": len(self.merges),
            "vocab_size": self.vocab_size,
            "tokens_in_use": int((tok_cnt[:nid] > 0).sum()),
            "tokens_after": int(tok_cnt[:nid].sum()),
            "compression_ratio": round(N / max(int(tok_cnt[:nid].sum()), 1), 2),
            "initial_criterion": round(history[0], 2),
            "final_criterion": round(history[-1], 2),
            "bits_per_char": round(history[-1] / N / math.log(2), 4),
        }
        return self

    def train_from_iterator(self, iterator: Iterable[str], **kw) -> "OccamTokenizer":
        """Train from any iterable of strings."""
        return self.train(list(iterator), **kw)

    # -- training internals ------------------------------------------------- #

    @staticmethod
    def _lvoc(vc: int, a: int) -> float:
        """Cost of naming each merge as an ordered pair of existing tokens."""
        return 0.0 if vc <= a else float(2.0 * (gammaln(vc) - gammaln(a)))

    def _screen(self, seq, dseq, tok_cnt, tok_len, nid, N):
        """Shortlist candidate merges by the cheap memoryless delta."""
        same = dseq[:-1] == dseq[1:]
        if not same.any():
            return None
        key = seq[:-1].astype(np.int64)[same] * nid + seq[1:][same]
        u, c = np.unique(key, return_counts=True)
        a_ids = (u // nid).astype(np.int32)
        b_ids = (u % nid).astype(np.int32)
        cab = c.astype(np.float64)

        if self.max_token_length is not None:
            keep = (tok_len[a_ids] + tok_len[b_ids]) <= self.max_token_length
            a_ids, b_ids, cab = a_ids[keep], b_ids[keep], cab[keep]
        if len(a_ids) == 0:
            return None

        na = tok_cnt[a_ids].astype(np.float64)
        nb = tok_cnt[b_ids].astype(np.float64)
        K = float(tok_cnt.sum()); V = float((tok_cnt > 0).sum())
        na2, nb2, K2 = na - cab, nb - cab, K - cab
        V2 = V + 1.0 - (na2 == 0) - (nb2 == 0)
        d = (gammaln(K2 + V2) - gammaln(V2) - gammaln(K2 + 1)
             - (gammaln(K + V) - gammaln(V) - gammaln(K + 1))
             + gammaln(K2 + 1) - gammaln(K + 1)
             + gammaln(na + 1) - gammaln(na2 + 1)
             + gammaln(nb + 1) - gammaln(nb2 + 1)
             - gammaln(cab + 1))
        m = min(self.candidates, len(d))
        idx = np.argpartition(d, m - 1)[:m]
        return a_ids[idx], b_ids[idx]

    @staticmethod
    def _pair_cells(seq, dseq, a_ids, b_ids, nid, D):
        """Sparse (candidate, document, count) triplets in one pass."""
        M = len(a_ids)
        s0, s1 = seq[:-1], seq[1:]
        same = dseq[:-1] == dseq[1:]
        eq = a_ids == b_ids
        slots, docs = [], []

        diff = np.where(~eq)[0]
        if len(diff):
            key = np.where(same, s0.astype(np.int64) * nid + s1, -1)
            ck = a_ids[diff].astype(np.int64) * nid + b_ids[diff]
            order = np.argsort(ck); cks = ck[order]
            pos = np.minimum(np.searchsorted(cks, key), len(cks) - 1)
            hit = cks[pos] == key
            if hit.any():
                slots.append(diff[order[pos[hit]]])
                docs.append(dseq[:-1][hit])

        for j in np.where(eq)[0]:                 # self-merges need de-overlap
            p = _dedup(np.where(same & (s0 == a_ids[j]) & (s1 == a_ids[j]))[0])
            if len(p):
                slots.append(np.full(len(p), j, dtype=np.int64))
                docs.append(dseq[p])

        if not slots:
            e = np.zeros(0, dtype=np.int64)
            return e, e, e.astype(np.float64)
        slot = np.concatenate(slots).astype(np.int64)
        doc = np.concatenate(docs).astype(np.int64)
        fc = np.bincount(slot * D + doc, minlength=M * D)
        nz = np.flatnonzero(fc)
        return nz // D, nz % D, fc[nz].astype(np.float64)

    def _deltas(self, seq, dseq, cells, Kdf, D, a_ids, b_ids, nid, V, tok_cnt):
        """
        Exact change in code length for each candidate.

        The per-document term shifts for every document because V moves, but
        for documents the merge does not touch it collapses to a function of
        V' alone -- and V' takes at most three values. So the corpus-wide part
        is computed once per distinct V' and only occupied cells are corrected,
        turning O(candidates * D) into O(3D + nnz).
        """
        M = len(a_ids)
        cand, doc, cnt = self._pair_cells(seq, dseq, a_ids, b_ids, nid, D)
        tot = (np.bincount(cand, weights=cnt, minlength=M) if len(cand)
               else np.zeros(M))

        ca = tok_cnt[a_ids].astype(np.float64)
        cb = tok_cnt[b_ids].astype(np.float64)
        eq = a_ids == b_ids
        gone_a = np.where(eq, (ca - 2 * tot) == 0, (ca - tot) == 0)
        gone_b = np.where(eq, False, (cb - tot) == 0)
        Vp = V + 1.0 - gone_a.astype(float) - gone_b.astype(float)

        base = float(gammaln(Kdf + V).sum())
        uV, inv = np.unique(Vp, return_inverse=True)
        T = np.array([float(gammaln(Kdf + v).sum()) for v in uV])
        out = T[inv] - base - D * (gammaln(Vp) - gammaln(V))

        if len(cand):
            vp, kd = Vp[cand], Kdf[doc]
            out += np.bincount(
                cand, weights=gammaln(kd - cnt + vp) - gammaln(kd + vp),
                minlength=M)
            e = eq[cand]
            na = cells.lookup(doc, a_ids[cand])
            nb = cells.lookup(doc, b_ids[cand])
            sub = np.where(e, 2.0 * cnt, cnt)
            w = gammaln(na + 1) - gammaln(na - sub + 1) - gammaln(cnt + 1)
            w += np.where(e, 0.0, gammaln(nb + 1) - gammaln(nb - cnt + 1))
            out += np.bincount(cand, weights=w, minlength=M)

        out = 2.0 * math.log(nid) + out
        out[tot == 0] = math.inf
        return out

    @staticmethod
    def _merge(seq, dseq, a, b, c, D):
        same = dseq[:-1] == dseq[1:]
        p = np.where(same & (seq[:-1] == a) & (seq[1:] == b))[0]
        if a == b:
            p = _dedup(p)
        if len(p) == 0:
            return seq, dseq, np.zeros(D, dtype=np.int64)
        md = np.bincount(dseq[p], minlength=D).astype(np.int64)
        mask = np.ones(len(seq), dtype=bool); mask[p + 1] = False
        s = seq.copy(); s[p] = np.int32(c)
        return s[mask], dseq[mask], md

    # -- vocabulary --------------------------------------------------------- #

    def _build_vocab(self) -> None:
        """
        Vocabulary = specials + alphabet + every merge product.

        Both extras matter. A character may have every occurrence absorbed
        into a longer token, and an intermediate merge product may be entirely
        consumed by a later merge -- so neither survives a count-based filter.
        But encoding unseen text can stop at exactly those tokens, because the
        surrounding context that triggered the later merge need not recur.
        Dropping them would emit [UNK] for text the tokenizer can represent.
        """
        specials = [t for t in (self.pad_token, self.unk_token) if t]
        keep = set(self.alphabet) | {a + b for a, b in self.merges}
        ordered = specials + sorted(keep - set(specials))
        self._ids = ordered
        self._vocab = {t: i for i, t in enumerate(ordered)}
        self._ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self._cache.clear()

    def _pre(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return [SPACE if c == " " else c for c in text.strip()]

    @property
    def vocab_size(self) -> int:
        return len(self._ids)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)

    def id_to_token(self, i: int) -> str | None:
        return self._ids[i] if 0 <= i < len(self._ids) else None

    # -- encoding ----------------------------------------------------------- #

    def _segment(self, text: str) -> list[str]:
        """
        Apply learned merges by rank.

        Merges are order-dependent -- a merge creates a token later merges
        consume -- so they must be applied lowest-rank-first. A heap over a
        linked list does that in one pass instead of replaying every merge
        over the whole sequence.
        """
        if text in self._cache:
            return self._cache[text]
        sym = self._pre(text)
        n = len(sym)
        if n < 2:
            out = list(sym)
            self._cache[text] = out
            return out

        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        alive = [True] * n
        heap: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            r = self._ranks.get((sym[i], sym[i + 1]))
            if r is not None:
                heapq.heappush(heap, (r, i, i + 1))

        while heap:
            r, i, j = heapq.heappop(heap)
            if not alive[i] or j >= n or not alive[j] or nxt[i] != j:
                continue
            if self._ranks.get((sym[i], sym[j])) != r:
                continue
            sym[i] = sym[i] + sym[j]
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k < n:
                prev[k] = i
            p = prev[i]
            if p >= 0 and alive[p]:
                rr = self._ranks.get((sym[p], sym[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p, i))
            if k < n and alive[k]:
                rr = self._ranks.get((sym[i], sym[k]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i, k))

        out = [sym[i] for i in range(n) if alive[i]]
        if len(self._cache) < 100_000:
            self._cache[text] = out
        return out

    def encode(self, text: str) -> Encoding:
        """Encode one text."""
        if not self._ids:
            raise RuntimeError("tokenizer is not trained; call train() first")
        toks = self._segment(text)
        unk = self._vocab.get(self.unk_token, 0)
        ids, out = [], []
        for t in toks:
            i = self._vocab.get(t)
            if i is None:
                i, t = unk, self.unk_token
            ids.append(i); out.append(t)
        return Encoding(ids=ids, tokens=out)

    def encode_batch(self, texts: Sequence[str]) -> list[Encoding]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token, self.unk_token}
        parts = []
        for i in ids:
            t = self.id_to_token(int(i))
            if t is None or (skip_special_tokens and t in specials):
                continue
            parts.append(t)
        return "".join(parts).replace(SPACE, " ").strip()

    def decode_batch(self, batch: Sequence[Sequence[int]], **kw) -> list[str]:
        return [self.decode(x, **kw) for x in batch]

    def __call__(self, texts: str | Sequence[str], padding: bool = False,
                 truncation: int | None = None) -> dict:
        """
        Encode for model input.

        Returns ``{"input_ids": [...], "attention_mask": [...]}``, batched
        unless a single string was passed.
        """
        single = isinstance(texts, str)
        encs = [self.encode(texts)] if single else self.encode_batch(texts)
        ids = [e.ids for e in encs]
        if truncation:
            ids = [x[:truncation] for x in ids]
        mask = [[1] * len(x) for x in ids]
        if padding and len(ids) > 1:
            width = max(len(x) for x in ids)
            pad = self._vocab.get(self.pad_token, 0)
            mask = [m + [0] * (width - len(m)) for m in mask]
            ids = [x + [pad] * (width - len(x)) for x in ids]
        if single:
            return {"input_ids": ids[0], "attention_mask": mask[0]}
        return {"input_ids": ids, "attention_mask": mask}

    # -- persistence -------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        """Write the tokenizer to a JSON file."""
        Path(path).write_text(json.dumps({
            "version": __version__,
            "config": {
                "max_token_length": self.max_token_length,
                "candidates": self.candidates,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "lowercase": self.lowercase,
            },
            "alphabet": self.alphabet,
            "merges": [list(m) for m in self.merges],
            "vocab": self._ids,
            "training_stats": self.training_stats,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "OccamTokenizer":
        """Load a tokenizer written by :meth:`save`."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(**d["config"])
        tok.alphabet = d["alphabet"]
        tok.merges = [tuple(m) for m in d["merges"]]
        tok.training_stats = d.get("training_stats", {})
        tok._ids = d["vocab"]
        tok._vocab = {t: i for i, t in enumerate(tok._ids)}
        tok._ranks = {(a, b): i for i, (a, b) in enumerate(tok.merges)}
        return tok

    def save_pretrained(self, directory: str | Path) -> None:
        p = Path(directory); p.mkdir(parents=True, exist_ok=True)
        self.save(p / "tokenizer.json")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "OccamTokenizer":
        return cls.from_file(Path(directory) / "tokenizer.json")
