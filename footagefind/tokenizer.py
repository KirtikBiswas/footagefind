"""Torch-free CLIP BPE tokenizer.

Adapted from open_clip's ``SimpleTokenizer`` (itself copied from
https://github.com/openai/CLIP, MIT License, Copyright (c) 2021 OpenAI).
Vendored so the search path needs only numpy + onnxruntime, which matters on
Windows-on-ARM where PyTorch wheels are not a given.
"""
from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import numpy as np
import regex as re

try:  # ftfy is optional; open_clip uses it for mojibake repair
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

BPE_PATH = Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
CONTEXT_LENGTH = 77


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: tuple) -> set:
    return {(a, b) for a, b in zip(word[:-1], word[1:])}


def _clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return " ".join(text.split()).strip().lower()


class SimpleTokenizer:
    def __init__(self, bpe_path: Path = BPE_PATH, context_length: int = CONTEXT_LENGTH):
        self.byte_encoder = bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = [tuple(m.split()) for m in merges[1 : 49152 - 256 - 2 + 1]]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        vocab.extend("".join(m) for m in merges)
        special = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special)
        self.encoder = {tok: i for i, tok in enumerate(vocab)}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.cache = {t: t for t in special}
        self.pat = re.compile(
            "|".join(special) + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in re.findall(self.pat, _clean(text)):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return ids

    def __call__(self, texts: str | list[str]) -> np.ndarray:
        """Return int32 token ids of shape [len(texts), context_length]."""
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.context_length), dtype=np.int32)
        for i, text in enumerate(texts):
            toks = [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            if len(toks) > self.context_length:
                toks = toks[: self.context_length]
                toks[-1] = self.eot_token_id
            out[i, : len(toks)] = toks
        return out


@lru_cache(maxsize=1)
def get_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()
