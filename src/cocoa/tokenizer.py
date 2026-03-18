#!/usr/bin/env python3

"""
tokenizes collated data into integer sequences, creating bins & a lookup table
"""

import collections
import datetime
import pathlib
import typing
import zoneinfo

import polars as pl
from omegaconf import OmegaConf

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
        self.bins = None
        self.subject_splits = None
        self.lookup = None
        self.is_training = is_training
        self.created_dttm = (
            datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
            .replace(microsecond=0)
            .isoformat()
        )

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

    def get_pretokenized(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            pl.concat_list(
                pl.concat_str(
                    pl.col("code"),
                    pl.col("binned_value"),
                    pl.col("text_value"),
                    separator="_",
                    ignore_nulls=True,
                )
                if self.cfg.fused
                else pl.concat_list(
                    "code", "binned_value", "text_value"
                ).list.drop_nulls()
            ).alias("to_tokenize")
        )

    def get_lookup(self, df: pl.LazyFrame) -> dict:
        if self.lookup is None and self.is_training:
            self.lookup = {"UNK": 0} | dict(
                self.get_pretokenized(df)
                .join(  # restrict to training data
                    self.subject_splits.filter(pl.col("split") == "train"),
                    on="subject_id",
                    validate="m:1",
                )
                .explode("to_tokenize")
                .select(pl.col("to_tokenize").unique().sort())
                .with_row_index("token", offset=1)  # UNK is zero
                .select("to_tokenize", "token")
                .collect()
                .iter_rows()
            )
        return self.lookup

    def tokenize_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return self.get_pretokenized(df).with_columns(
            tokens=pl.col("to_tokenize").list.eval(
                pl.element().replace_strict(old=self.get_lookup(df), default=0)
            )
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
        df = self.tokenize_data(df)  # create lookup table and run tokenization
        df = self.aggregate_timelines_from_tokens(df)  # collect tokens into timelines
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
        self.save(to_folder / "tokenizer.yaml")

    def __contains__(self, word: Hashable) -> bool:
        return word in self.lookup.keys()

    def __str__(self):
        return "{sp} of {sz} words {md}".format(
            sp=self.__class__,
            sz=len(self),
            md="in training mode" if self.is_training else "(frozen)",
        )

    def __repr__(self):
        return str(self) + ", created {dttm}".format(dttm=self.created_dttm)

    def __len__(self) -> int:
        return len(self.lookup.keys())

    def to_yaml(self) -> str:
        bins = self.bins.collect().to_dicts() if self.bins is not None else None
        return OmegaConf.to_yaml(
            {
                "lookup": self.lookup,
                "bins": bins,
                "is_training": self.is_training,
                "cfg": OmegaConf.to_container(self.cfg),
                "created_dttm": self.created_dttm,
            }
        )

    @classmethod
    def from_yaml(cls, yaml_str: str, done_training=True) -> "Tokenizer":
        data = OmegaConf.create(yaml_str)
        cfg = OmegaConf.to_container(data.cfg)
        tkzr = cls(is_training=data.is_training, **cfg)
        # `data.lookup` may be an OmegaConf mapping; convert to plain dict
        tkzr.lookup = (
            dict(OmegaConf.to_container(data.lookup))
            if data.lookup is not None
            else None
        )
        tkzr.created_dttm = data.created_dttm
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


if __name__ == "__main__":
    tkzr = Tokenizer()
    tkzr.save_all()
    # breakpoint()
