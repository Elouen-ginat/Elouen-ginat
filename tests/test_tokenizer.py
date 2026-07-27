"""Tests for OccamTokenizer."""

import json
import tempfile
from pathlib import Path

import pytest

from occam_tokenizer import SPACE, Encoding, OccamTokenizer

DOCS = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "a rat sat on a mat",
    "walking walked walker walks",
    "running runner runs ran",
    "the dog slept by the door",
    "nationalization globalization",
] * 6


@pytest.fixture(scope="module")
def tok():
    return OccamTokenizer(max_token_length=20).train(DOCS)


def test_train_learns_merges(tok):
    assert tok.merges
    assert tok.vocab_size > len(tok.alphabet)
    assert tok.training_stats["compression_ratio"] > 1.0


def test_criterion_decreases(tok):
    s = tok.training_stats
    assert s["final_criterion"] < s["initial_criterion"]


def test_no_vocab_size_argument():
    """The whole point: vocabulary size is chosen by the criterion."""
    import inspect

    p = inspect.signature(OccamTokenizer.__init__).parameters
    assert not {"vocab_size", "num_merges", "target_vocab"} & set(p)


def test_roundtrip(tok):
    for t in ["the cat sat on the mat", "walking", "a", "z"]:
        assert tok.decode(tok.encode(t).ids) == t


def test_alphabet_always_encodable(tok):
    """Characters absorbed entirely into merges stay in the vocabulary."""
    for ch in tok.alphabet:
        if ch != SPACE:
            assert tok.token_to_id(ch) is not None


def test_encoding_shape(tok):
    e = tok.encode("the cat sat")
    assert isinstance(e, Encoding)
    assert len(e.ids) == len(e.tokens) == len(e.attention_mask) == len(e)
    assert all(tok.id_to_token(i) == t for i, t in zip(e.ids, e.tokens, strict=True))


def test_batch(tok):
    encs = tok.encode_batch(["the cat", "the rat"])
    assert len(encs) == 2
    assert all(isinstance(e, Encoding) for e in encs)


def test_call_padding(tok):
    out = tok(["the cat sat on the mat", "a"], padding=True)
    ids, mask = out["input_ids"], out["attention_mask"]
    assert len(ids[0]) == len(ids[1])
    assert sum(mask[1]) < len(mask[1])


def test_call_single_is_unbatched(tok):
    out = tok("the cat")
    assert isinstance(out["input_ids"][0], int)


def test_truncation(tok):
    long = "the cat sat on the mat and the rat ate the bat by the door"
    full = len(tok(long)["input_ids"])
    assert len(tok(long, truncation=3)["input_ids"]) == min(3, full)


def test_unknown_character(tok):
    e = tok.encode("\u03c9")
    assert e.tokens == [tok.unk_token]


def test_empty_and_short(tok):
    assert tok.encode("").ids == []
    assert len(tok.encode("a").ids) == 1


def test_self_merge_overlap():
    """'sss' has two (s,s) adjacencies but admits one merge."""
    t = OccamTokenizer(max_token_length=8).train(["sss ssss sssss"] * 20)
    assert t.decode(t.encode("sssss").ids) == "sssss"


def test_save_load_roundtrip(tok):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        tok.save(p)
        other = OccamTokenizer.from_file(p)
        assert other.vocab_size == tok.vocab_size
        for s in ["the cat sat", "walking"]:
            assert other.encode(s).ids == tok.encode(s).ids


def test_save_pretrained(tok):
    with tempfile.TemporaryDirectory() as d:
        tok.save_pretrained(d)
        assert OccamTokenizer.from_pretrained(d).merges == tok.merges


def test_json_is_readable(tok):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.json"
        tok.save(p)
        assert {"merges", "vocab", "config"} <= set(json.loads(p.read_text()))


def test_lowercase_option():
    t = OccamTokenizer(lowercase=True).train(DOCS)
    assert t.encode("THE CAT").ids == t.encode("the cat").ids


def test_untrained_raises():
    with pytest.raises(RuntimeError, match="not trained"):
        OccamTokenizer().encode("x")


def test_empty_corpus_raises():
    with pytest.raises(ValueError, match="no non-empty documents"):
        OccamTokenizer().train(["", "   "])


def test_document_split_matters():
    """Merged into one document the criterion loses its selection pressure."""
    split = OccamTokenizer(max_token_length=20).train(DOCS)
    joined = OccamTokenizer(max_token_length=20).train([" ".join(DOCS)])
    assert split.merges != joined.merges
