"""Shared tokenization helpers for extract_v4_* and steer_v4.

Rationale:
- `tokenizer.encode("tall")[0]` (plain) and `tokenizer.encode(" tall")[0]`
  (with leading space) give DIFFERENT token IDs in SentencePiece / BPE
  tokenizers. When the prompt ends with "... is" and the next token will be
  a word continuation, the " tall" token is usually the correct one.
- get_first_token tries several spacing variants and prefers a variant that
  encodes to a single token. If all variants are multi-token, it returns
  the first token of the best (fewest-tokens) variant along with a flag.

Use first_token_id() for simple cases, tokens_of_word() when you need the
full list (e.g. for multi-token log-prob scoring or for logging warnings).
"""
from __future__ import annotations


def _spacing_variants(word: str) -> list[str]:
    """Try leading-space first (the natural continuation after 'is ')."""
    return [f" {word}", word, f" {word.capitalize()}", word.capitalize()]


def tokens_of_word(tokenizer, word: str) -> list[int]:
    """Return the token-ID list for the best (fewest-tokens) spacing variant.

    If any variant encodes to a single token, that variant wins.
    Otherwise returns the shortest multi-token encoding.
    """
    best: list[int] | None = None
    for v in _spacing_variants(word):
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            return ids
        if best is None or len(ids) < len(best):
            best = ids
    assert best is not None
    return best


def first_token_id(tokenizer, word: str) -> int:
    """Return the single token ID if the word is single-token under any
    spacing variant, else the FIRST token of the shortest encoding.

    Callers that care about multi-token bias should use tokens_of_word()
    and check len > 1 before calling this.
    """
    return tokens_of_word(tokenizer, word)[0]


def report_tokenization(tokenizer, words: list[str]) -> dict[str, list[int]]:
    """Print tokenization for each word and return {word: token_ids}."""
    table = {}
    for w in words:
        ids = tokens_of_word(tokenizer, w)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        flag = "" if len(ids) == 1 else f"  [MULTI-TOKEN: {len(ids)} pieces]"
        print(f"  '{w}': ids={ids} tokens={tokens}{flag}", flush=True)
        table[w] = ids
    return table
