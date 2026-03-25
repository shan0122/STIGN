"""
Vocabulary utility: maps words ↔ integer ids.

Special tokens:
  <pad> = 0
  <sos> = 1
  <eos> = 2
  <unk> = 3
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def _tokenize(sentence: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer (lowercase)."""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\w\s']", " ", sentence)
    return sentence.split()


class Vocabulary:
    """Word ↔ index bidirectional mapping.

    Args:
        min_freq (int): Minimum word frequency to include in vocabulary.
    """

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._counter: Counter = Counter()
        self._built = False

        for tok in _SPECIAL_TOKENS:
            self._add(tok)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _add(self, word: str) -> None:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def count_sentence(self, sentence: str) -> None:
        """Count words in a sentence (call before :meth:`build`)."""
        self._counter.update(_tokenize(sentence))

    def build(self) -> None:
        """Finalise vocabulary from counted sentences."""
        for word, freq in self._counter.items():
            if freq >= self.min_freq:
                self._add(word)
        self._built = True

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, sentence: str, add_special: bool = True) -> list[int]:
        """Convert a sentence string to a list of token ids.

        Args:
            sentence:    Input string.
            add_special: Wrap with <sos> and <eos> when True.
        """
        tokens = _tokenize(sentence)
        ids = [self.word2idx.get(t, UNK_ID) for t in tokens]
        if add_special:
            ids = [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Convert a list of token ids back to a sentence string."""
        words = []
        for i in ids:
            w = self.idx2word.get(i, "<unk>")
            if skip_special and w in _SPECIAL_TOKENS:
                continue
            words.append(w)
        return " ".join(words)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from a previously saved JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = cls(min_freq=payload["min_freq"])
        vocab.word2idx = {w: int(i) for w, i in payload["word2idx"].items()}
        vocab.idx2word = {int(i): w for w, i in payload["word2idx"].items()}
        vocab._built = True
        return vocab

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def sos_id(self) -> int:
        return SOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID
