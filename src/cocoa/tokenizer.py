#!/usr/bin/env python3

"""
tokenizes collated data into integer sequences, creating a vocabulary and bins
"""

import collections
import gzip
import pathlib
import pickle
import typing

import polars as pl
from omegaconf import OmegaConf

from cocoa.vocabulary import Vocabulary

Hashable: typing.TypeAlias = collections.abc.Hashable


class Tokenizer:
    def __init__(self, is_training: bool = True, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        tokenization_cfg = OmegaConf.load(
            pathlib.Path(main_cfg.tokenization_config).expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, tokenization_cfg, kwargs)
        self.data_home = pathlib.Path(self.cfg.data_home).expanduser().resolve()
        self.vocab = Vocabulary(is_training=is_training)
        self.bins = None
        self.subject_splits = None
        self._is_training = is_training

    def get_data(self) -> pl.LazyFrame:
        self.subject_splits = pl.scan_parquet(
            pathlib.Path(self.cfg.subject_splits).expanduser().resolve()
        )
        return pl.concat(
            [
                pl.scan_parquet(pathlib.Path(f).expanduser().resolve())
                for f in self.cfg.collated_inputs
            ],
            how="diagonal",
        )

    @staticmethod
    def add_ends(df: pl.LazyFrame) -> pl.LazyFrame:
        # add BOS / EOS codes at appropriate places
        return pl.concat(
            [
                df,
                df.group_by("subject_id")
                .agg(pl.col("time").min())
                .with_columns(code=pl.lit("BOS")),
                df.group_by("subject_id")
                .agg(pl.col("time").max())
                .with_columns(code=pl.lit("EOS")),
            ],
            how="diagonal",
        )

    def get_bins(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.bins is None and self.is_training:
            self.bins = (
                df.join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .filter(pl.col("numeric_value").is_not_null())
                .group_by("code")
                .agg(
                    [
                        pl.col("numeric_value")
                        .quantile(i / self.cfg.n_bins)
                        .alias(f"break_{i}")
                        for i in range(1, self.cfg.n_bins)
                    ]
                )
            )
        return self.bins

    def bin_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.join(self.get_bins(df), on="code", how="left")
            .with_columns(
                pl.when(pl.col("numeric_value").is_not_null())
                .then(
                    pl.concat_str(
                        pl.lit("Q"),
                        pl.sum_horizontal(
                            [
                                pl.col(f"break_{i}") <= pl.col("numeric_value")
                                for i in range(1, self.cfg.n_bins)
                            ]
                        ).cast(pl.String),
                    )
                )
                .otherwise(None)
                .alias("binned_value")
            )
            .drop([f"break_{i}" for i in range(1, self.cfg.n_bins)])
        )

    def get_tokens_from_binned_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.cfg.fused:
            return df.with_columns(
                pl.concat_list(
                    pl.concat_str(
                        pl.col("code"),
                        pl.col("binned_value"),
                        pl.col("text_value"),
                        separator="_",
                        ignore_nulls=True,
                    ).map_elements(self.vocab, return_dtype=pl.Int64, skip_nulls=True)
                ).alias("tokens")
            )
        else:
            return df.with_columns(
                pl.concat_list(
                    pl.col("code").map_elements(
                        self.vocab, return_dtype=pl.Int64, skip_nulls=True
                    ),
                    pl.col("binned_value").map_elements(
                        self.vocab, return_dtype=pl.Int64, skip_nulls=True
                    ),
                    pl.col("text_value").map_elements(
                        self.vocab, return_dtype=pl.Int64, skip_nulls=True
                    ),
                )
                .list.drop_nulls()
                .alias("tokens")
            )

    def aggregate_timelines_from_tokens(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.sort(
                "time",
                pl.col("code")
                .str.split("//")
                .list[0]
                .replace(dict(enumerate(self.cfg.ordering))),
            )
            .explode("tokens")
            .group_by("subject_id", maintain_order=True)
            .agg("tokens", pl.col("time").alias("times"))
        )

    def get_all(self) -> pl.LazyFrame:
        df = self.get_data()  # load data
        df = self.add_ends(df)  # add BOS/EOS tokens
        df = self.bin_data(df)  # create bins from training data and bin numeric values
        df = df.join(self.subject_splits, on="subject_id", validate="m:1")
        df_tr = self.get_tokens_from_binned_data(
            df.filter(pl.col("split") == "train")
        ).collect()  # collect to force vocab population before freezing
        self.is_training = False  # freeze tokenizer after processing training data
        df_tu_ho = self.get_tokens_from_binned_data(
            df.filter(pl.col("split") != "train")
        )  # converts binned data to tokens for tuning and held-out data
        df = self.aggregate_timelines_from_tokens(
            pl.concat([df_tr.lazy(), df_tu_ho])
        )  # collect tokens into timelines
        return df

    def save_all(self, path: pathlib.Path = None):
        df = self.get_all()

        to_folder = (
            pathlib.Path(path if path is not None else self.cfg.processed_data_home)
            .expanduser()
            .resolve()
        )
        to_folder.mkdir(parents=True, exist_ok=True)
        df.sink_parquet(to_folder / "tokens_times.parquet")
        with gzip.open(to_folder / "tkzr.pkl.gz", "wb") as f:
            self.is_training = False
            pickle.dump(self, f)

    def __contains__(self, word: Hashable) -> bool:
        return word in self.vocab

    def __str__(self):
        return "{sp} of {sz} words {md}".format(
            sp=self.__class__,
            sz=len(self),
            md="in training mode" if self.is_training else "(frozen)",
        )

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return len(self.vocab)

    def to_yaml(self) -> str:
        bins = self.bins.collect().to_dicts() if self.bins is not None else None
        return OmegaConf.to_yaml(
            {
                "vocab": OmegaConf.create(self.vocab.to_yaml()),
                "bins": bins,
                "is_training": self.is_training,
                "cfg": OmegaConf.to_container(self.cfg),
            }
        )

    @classmethod
    def from_yaml(cls, yaml_str: str, done_training=True) -> "Tokenizer":
        data = OmegaConf.create(yaml_str)
        cfg = OmegaConf.to_container(data.cfg)
        tkzr = cls(is_training=data.is_training, **cfg)
        tkzr.vocab = Vocabulary.from_yaml(OmegaConf.to_yaml(data.vocab))
        if data.bins is not None:
            tkzr.bins = pl.DataFrame(OmegaConf.to_container(data.bins)).lazy()
        if done_training:
            tkzr.is_training = False
        return tkzr

    def save(self, path: pathlib.Path):
        to_file = pathlib.Path(path).expanduser().resolve()
        to_file.parent.mkdir(parents=True, exist_ok=True)
        with open(to_file, "w") as f:
            f.write(self.to_yaml())

    def load(self, path: pathlib.Path, done_training=True):
        from_file = pathlib.Path(path).expanduser().resolve()
        with open(from_file, "r") as f:
            yaml_str = f.read()
        return self.from_yaml(yaml_str, done_training=done_training)

    @property
    def is_training(self) -> bool:
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value
        self.vocab.is_training = value


if __name__ == "__main__":
    tkzr = Tokenizer()
    tkzr.save_all()
    tkzr.save("~/Downloads/tkzr.yaml")
    breakpoint()
