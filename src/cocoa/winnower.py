#!/usr/bin/env python3

"""
prepares held-out data for evaluation,
adding flags to disqualify certain subjects from evaluation
"""

import pathlib

import polars as pl
from omegaconf import OmegaConf

from cocoa.reporter import Logger


class Winnower:
    """
    filters held-out timelines for evaluation;
    assigns flags to disqualify certain subjects from evaluation,
    e.g. those whose timelines ends prior to the outcome horizon
    """

    def __init__(self, **kwargs):
        main_cfg = OmegaConf.load(
            pathlib.Path("./config/main.yaml").expanduser().resolve()
        )
        winnowing_cfg = OmegaConf.load(
            pathlib.Path(main_cfg.winnowing_config).expanduser().resolve()
        )
        self.cfg = OmegaConf.merge(main_cfg, winnowing_cfg, kwargs)
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")

    def prepare_winnowed_frame(self) -> pl.LazyFrame:
        """
        loads held-out data, splits at time threshold, and prepares labels
        """
        return (
            pl.scan_parquet(self.processed_data_home / "tokens_times.parquet")
            .join(
                pl.scan_parquet(self.processed_data_home / "subject_splits.parquet"),
                on="subject_id",
            )
            .filter(pl.col("split") == "held_out")
            .drop("split")
            .with_columns(
                s_elapsed=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                )
            )
            .with_columns(
                s_total_duration=pl.col("s_elapsed").list.last(),
                n_past=pl.col("s_elapsed")
                .list.eval(pl.element() < self.cfg.horizon_s)
                .list.sum(),
            )
            .filter(pl.col("s_total_duration") > self.cfg.horizon_s)
            .with_columns(
                tokens_past=pl.col("tokens").list.head("n_past"),
                tokens_future=pl.col("tokens").list.tail(
                    pl.col("tokens").list.len() - pl.col("n_past")
                ),
            )
            .with_columns(
                **{
                    f"{t}_{tense}": pl.col(f"tokens_{tense}").list.contains(
                        self.tkzr_cfg.lookup[t]
                    )
                    for t in self.cfg.outcome_tokens
                    for tense in ("past", "future")
                }
            )
        )

    def save_all(self, path: pathlib.Path = None, verbose: bool = False):
        """
        grabs winnowed frame, prints summary stats if requested, and saves it
        """
        df = self.prepare_winnowed_frame()
        to_folder = (
            pathlib.Path(path if path is not None else self.cfg.processed_data_home)
            .expanduser()
            .resolve()
        )
        to_folder.mkdir(parents=True, exist_ok=True)
        df.sink_parquet(
            to_folder / "held_out_for_inference.parquet", engine="streaming"
        )
        if verbose:
            logger = Logger()
            logger.info(
                (
                    df.select(
                        [
                            pl.col(f"{t}_{s}").mean().alias(f"{t}_{s}")
                            for t in self.cfg.outcome_tokens
                            for s in ("past", "future")
                        ]
                    )
                    .collect()
                    .transpose(
                        include_header=True, header_name="event", column_names=("rate",)
                    )
                    .with_columns(
                        token=pl.col("event").str.replace(r"_(past|future)$", ""),
                        tense=pl.col("event").str.extract(r"(past|future)$"),
                    )
                    .pivot(values="rate", index="token", on="tense")
                )
            )


if __name__ == "__main__":
    self = Winnower()
    self.save_all(verbose=True)
    # breakpoint()
