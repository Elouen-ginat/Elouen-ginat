"""Parameter-free unsupervised tokenization by Minimum Description Length.

There is no vocabulary size to choose. Merging stops when no candidate merge
lowers the description length of the corpus.

    from occam_tokenizer import OccamTokenizer

    tok = OccamTokenizer()
    tok.train(documents)
    tok.encode("hello world").tokens
"""

from .__about__ import __version__
from .tokenizer import SPACE, Encoding, OccamTokenizer

__all__ = ["SPACE", "Encoding", "OccamTokenizer", "__version__"]
