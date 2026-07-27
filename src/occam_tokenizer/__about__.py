"""Single source of truth for the package version.

Kept import-free so that the build backend can read it without executing the
package, and so that :mod:`occam_tokenizer.tokenizer` can stamp saved
tokenizers without importing the package root.
"""

__version__ = "0.1.0"
