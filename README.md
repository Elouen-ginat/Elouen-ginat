# occam-tokenizer

Parameter-free unsupervised tokenization by Minimum Description Length.

**There is no vocabulary size to choose.** Merging stops when no candidate
merge lowers the description length of the corpus.

```python
from occam_tokenizer import OccamTokenizer

tok = OccamTokenizer()
tok.train(documents)  # list[str] — that's the whole config

enc = tok.encode("hello world")
enc.tokens  # ['hello▁', 'wor', 'ld']
enc.ids  # [412, 88, 130]
tok.decode(enc.ids)  # 'hello world'
```

## Install

```bash
pip install -e .
```

Requires numpy and scipy.

## API

```python
tok = OccamTokenizer(
    max_token_length=25,  # cap on token length in characters, or None
    candidates=12,  # merges scored exactly per step
    lowercase=False,
)

tok.train(documents)  # or train_from_iterator(iterable)

tok.encode(text)  # -> Encoding(ids, tokens, attention_mask)
tok.encode_batch(texts)  # -> list[Encoding]
tok(texts, padding=True, truncation=64)  # -> {'input_ids', 'attention_mask'}

tok.decode(ids)
tok.decode_batch(list_of_ids)

tok.vocab_size
tok.get_vocab()  # {token: id}
tok.token_to_id(tok), tok.id_to_token(i)
tok.training_stats  # merges, compression, code length

tok.save("tokenizer.json")
OccamTokenizer.from_file("tokenizer.json")

tok.save_pretrained("dir/")
OccamTokenizer.from_pretrained("dir/")
```

Run `python example.py` for a worked example.

## How it works

The corpus is modelled as one multinomial per document over the V token types.
With `n_dv` the count of token v in document d, `K_d` the length of document d,
`N` the character count and `Vc` the constructed vocabulary size, the code
length being minimised is

```
C = log N                                     prior on V
  + 2 [lnΓ(Vc) − lnΓ(|A|)]                    vocabulary construction
  + Σ_d [ lnΓ(K_d + V) − lnΓ(V) ]             per-document prior + ordering
  − Σ_dv lnΓ(n_dv + 1)                        multinomial likelihood
```

The third line collapses a stars-and-bars prior and its enumerative likelihood,
since `log C(n+V−1, V−1) + log(n!) = lnΓ(n+V) − lnΓ(V)`.

A merge is accepted only if it lowers `C`. That is the entire stopping rule.

### Why condition on the document

Every vocabulary entry costs roughly `D log(1 + K_d/V)` — paid in *every*
document, earned back only where the token occurs. A token spread across the
corpus repays that; one confined to a few documents does not.

Without this, a corpus-wide multinomial only sees total counts and learns
literals. Measured on the same corpora:

```
                 global multinomial              per-document
shakespeare   'caius·marcius·coriolanus'   '·have', '·that', 'ther'
python        '+-------+-------+------+'   'default_factory', 'dataclass'
```

```
"def __init__(self, name, default=None):"
  global    (5)  def·__init__(self,·|name,·|default|=none|):
  document (14)  def|·|__|init|__|(|self|,·|name|,·|default|=|none|):
```

The first memorised the signature. The second recovered reusable units.

**Document boundaries are load-bearing.** How you split the corpus determines
what counts as a corpus-wide token. Passing one giant string removes the
signal entirely; repeating identical documents makes local phrases look global.

## Performance

Roughly **500k characters/second** on one core, scaling sub-linearly
(α = 0.83 fitted log-log up to 1M characters).

Memory for the document × token cell store is `O(nnz)` with `nnz ≤ K`, not
`O(D·V)`. Cells are packed token-major (`key = token * D + document`), which
makes a column a contiguous slice, a lookup one vectorised `searchsorted`,
and — since a new token always gets the largest id — a new column an append
rather than a rebuild. That last property is why this is not CSC: merging
changes the sparsity structure at every step.

At 30k documents this is 7.6 MB against 123 MB dense. The gap widens as
documents get shorter and closes entirely around 640 characters per document.

## Limitations

- Reported behaviour is from in-sample description length on English, French
  and Python source (~45k characters each). No downstream task evaluation.
  Code length and tokenizer quality demonstrably diverge — this criterion has
  a *higher* code length than a plain global multinomial while producing the
  better vocabulary — so downstream evaluation is not optional.
- Merge selection is greedy over a screened shortlist. Widening `candidates`
  searches more but was not observed to change the outcome materially.
- Document count is still a linear term in *training time* even though it is
  gone from memory.
- Single-threaded.

## Reference

Boullé, Clérot & Hue (2016), *Revisiting enumerative two-part crude MDL for
Bernoulli and multinomial distributions*, [arXiv:1608.05522](https://arxiv.org/abs/1608.05522).

The MODL framework underlies [Khiops](https://khiops.org).

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

## License

Not yet chosen — see repository owner.
