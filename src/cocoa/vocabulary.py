#!/usr/bin/env python3

"""
provides a generic class for creating and maintaining a map from a vocabulary
of strings to unique integers;
The Vocabulary class has two modes: training and not-training.
- During training, new items are added to the lookup table.
- After training, the vocabulary is frozen and new items are all mapped to UNK
"""

import collections
import typing
import warnings

from omegaconf import OmegaConf

Hashable: typing.TypeAlias = collections.abc.Hashable


class Vocabulary:
    """
    maintains a dictionary `lookup` mapping words -> tokens,
    a dictionary `reverse` inverting the lookup
    """

    def __init__(
        self, words: tuple = ("UNK", "BOS", "EOS"), *, is_training: bool = True
    ):
        assert len(set(words)) == len(words)
        self.lookup = {v: i for i, v in enumerate(words)}
        self.reverse = dict(enumerate(words))
        self.is_training = is_training

    def __call__(self, word: Hashable) -> int | None:
        try:
            return self.lookup[word]
        except KeyError:
            if self.is_training:
                self.lookup[word], self.reverse[n] = (n := len(self.lookup)), word
                return n
            else:
                warnings.warn(
                    "Encountered previously unseen token: {} {}".format(
                        word, type(word)
                    )
                )
                return self.lookup["UNK"] if "UNK" in self.lookup else None

    def __contains__(self, word: Hashable) -> bool:
        return word in self.lookup

    def __str__(self):
        return "{sp} of {sz} words {md}".format(
            sp=self.__class__,
            sz=len(self),
            md="in training mode" if self.is_training else "(frozen)",
        )

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, word):
        return self.__call__(word)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(
            {
                "words": [self.reverse[i] for i in range(len(self))],
                "is_training": self.is_training,
            }
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Vocabulary":
        cfg = OmegaConf.create(yaml_str)
        return cls(tuple(cfg.words), is_training=cfg.is_training)
