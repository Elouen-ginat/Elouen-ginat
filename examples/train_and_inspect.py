"""
Train an Occam tokenizer and inspect what it learned.

Note the corpus construction. The criterion charges each vocabulary entry
across every document and earns it back only where the token occurs, so a
phrase confined to a few documents does not pay. That check only works if the
documents genuinely differ -- repeating the same sentence 30 times makes the
whole phrase look corpus-wide, and the tokenizer will happily memorise it.
Even so, this corpus is templated, so phrases like "past the wooden gate"
genuinely do recur corpus-wide and the tokenizer is right to absorb them.
Real text has Zipfian diversity and produces morpheme-like tokens instead;
max_token_length is capped at 12 here to keep the demo output readable.
"""

import random

from occam_tokenizer import OccamTokenizer

random.seed(0)
SUBJ = ["the cat", "the dog", "a bird", "the child", "my neighbour", "the runner"]
VERB = [
    "walked",
    "walking",
    "walks",
    "ran",
    "running",
    "runs",
    "jumped",
    "jumping",
    "jumps",
    "opened",
    "opening",
    "opens",
]
ADV = ["quickly", "slowly", "carefully", "quietly", "suddenly", "gladly"]
OBJ = [
    "the wooden gate",
    "a stone wall",
    "the open door",
    "the tall window",
    "a narrow path",
    "the small box",
]
ABST = [
    "nationalization",
    "globalization",
    "modernization",
    "privatization",
    "happiness",
    "kindness",
    "darkness",
    "readability",
    "teachability",
]

docs = [
    f"{random.choice(SUBJ)} {random.choice(VERB)} {random.choice(ADV)} "
    f"past {random.choice(OBJ)}"
    for _ in range(300)
]
docs += [
    f"{random.choice(ABST)} and {random.choice(ABST)} are different" for _ in range(150)
]

tok = OccamTokenizer(max_token_length=12, lowercase=True)
tok.train(docs)

print(tok)
for k, v in tok.training_stats.items():
    print(f"  {k:<20} {v}")

print("\nlearned tokens, longest first:")
for t in sorted({a + b for a, b in tok.merges}, key=len, reverse=True)[:12]:
    print(f"  {t!r}")

print("\nencoding:")
for text in ["the cat walking quickly", "globalization", "unseen wordiness"]:
    enc = tok.encode(text)
    print(f"  {text!r}")
    print(f"    tokens {enc.tokens}")
    print(f"    ids    {enc.ids}")
    print(f"    decode {tok.decode(enc.ids)!r}")

print("\nbatch for a model:")
print(" ", tok(["the dog runs", "kindness"], padding=True))

tok.save("tokenizer.json")
reloaded = OccamTokenizer.from_file("tokenizer.json")
print(
    "\nreload identical:",
    reloaded.encode("the cat walking").ids == tok.encode("the cat walking").ids,
)
