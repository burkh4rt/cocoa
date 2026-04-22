#!/usr/bin/env python3

"""
prepares held-out data for evaluation,
adding flags to disqualify certain subjects from evaluation
"""

import pathlib

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from cocoa.logger import Logger


class Winnower:
    """
    filters held-out timelines for evaluation;
    assigns flags to disqualify certain subjects from evaluation,
    e.g. those whose timelines ends prior to the outcome horizon
    """

    def __init__(
        self,
        main_cfg: pathlib.Path | str = None,
        winnowing_cfg: pathlib.Path | str = None,
        **kwargs,
    ):
        main_cfg = OmegaConf.load(
            pathlib.Path(main_cfg if main_cfg is not None else "./config/main.yaml")
            .expanduser()
            .resolve()
        )
        winnowing_cfg = OmegaConf.load(
            pathlib.Path(
                winnowing_cfg
                if winnowing_cfg is not None
                else main_cfg.winnowing_config
            )
            .expanduser()
            .resolve()
        )
        self.cfg = OmegaConf.merge(
            main_cfg, winnowing_cfg, {k: v for k, v in kwargs.items() if v is not None}
        )
        self.processed_data_home = (
            pathlib.Path(self.cfg.processed_data_home).expanduser().resolve()
        )
        self.tkzr_cfg = OmegaConf.load(self.processed_data_home / "tokenizer.yaml")
        self.rng = np.random.default_rng(seed=42)

    def load_frame(self) -> pl.LazyFrame:
        """
        loads held_out timelines, and performs some preliminary calculations;
        these are lazily evaluated, so only completed if used
        """
        return (
            pl.scan_parquet(self.processed_data_home / "tokens_times.parquet")
            .join(
                pl.scan_parquet(self.processed_data_home / "subject_splits.parquet"),
                on="subject_id",
                validate="1:1",
            )
            .filter(pl.col("split") == "held_out")
            .drop("split")
            .with_columns(
                s_elapsed=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                )
            )
            .with_columns(s_total_duration=pl.col("s_elapsed").list.last())
        )

    def run_thresholding(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        evaluates configurable criteria for establishing a cut-point "last_valid";
        drops timelines that do not reach that point
        """
        if "horizon_s" in self.cfg or "duration_s" in self.cfg.get("threshold", {}):
            # run duration-based thresholding
            horizon_s = self.cfg.get("horizon_s", self.cfg.threshold.duration_s)
            return df.filter(pl.col("s_total_duration") > horizon_s).with_columns(
                last_valid=pl.col("s_elapsed")
                .list.eval(pl.element() < horizon_s)
                .list.sum()
            )
        elif "first_occurrence" in self.cfg.get("threshold", {}):
            # run first-occurrence-based thresholding
            toi = self.tkzr_cfg.lookup[self.cfg.threshold.first_occurrence]
            return df.filter(pl.col("tokens").list.contains(toi)).with_columns(
                last_valid=pl.col("tokens")
                .list.eval(pl.element() == toi)
                .list.arg_max()
                + pl.lit(1)
                # place the triggering token into the past; it is known
            )
        elif (
            "uniform_random" in self.cfg.get("threshold", {})
            and self.cfg.threshold.uniform_random
        ):
            # set the threshold uniformly at random over the duration of each stay
            return df.with_columns(
                sampled_duration=pl.col("s_total_duration").map_elements(
                    lambda x: x * self.rng.random()
                )
            ).with_columns(
                last_valid=pl.struct(["s_elapsed", "sampled_duration"]).map_elements(
                    lambda row: sum(
                        x < row["sampled_duration"] for x in row["s_elapsed"]
                    )
                )
            )
        else:
            raise NotImplementedError("Please check the thresholding configuration.")

    def add_outcome_flags(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        adds boolean flags for each outcome token and tense,
        e.g. DSCG//expired_past, DSCG//expired_future
        """
        df = df.with_columns(
            tokens_past=pl.col("tokens").list.head("last_valid"),
            tokens_future=pl.col("tokens").list.tail(
                pl.col("tokens").list.len() - pl.col("last_valid")
            ),
        )  # split into past and future
        if "horizon_after_threshold_s" in self.cfg:
            df = (
                df.with_columns(
                    s_elapsed_thresh=pl.col("times")
                    .list.tail(
                        pl.col("tokens_future").list.len() + 1
                    )  # include threshold time
                    .list.eval((pl.element() - pl.element().first()).dt.total_seconds())
                )
                .with_columns(
                    valid_future_count=pl.col("s_elapsed_thresh")
                    .list.eval(pl.element() <= self.cfg.horizon_after_threshold_s)
                    .list.sum()
                    - pl.lit(1)  # threshold token was counted, drop it
                )
                .with_columns(
                    tokens_future=pl.col("tokens_future").list.head(
                        "valid_future_count"
                    )
                )
            )
        return df.select(
            "subject_id", "tokens", "times", "tokens_past", "tokens_future"
        ).with_columns(
            **{
                f"{t}_{tense}": pl.col(f"tokens_{tense}").list.contains(
                    self.tkzr_cfg.lookup[t]
                )
                for t in self.cfg.outcome_tokens
                for tense in ("past", "future")
            }
        )

    def prepare_winnowed_frame(self) -> pl.LazyFrame:
        """loads held-out data, splits at time threshold, and prepares labels"""
        return (
            self.load_frame().pipe(self.run_thresholding).pipe(self.add_outcome_flags)
        )

    def save_all(self, verbose: bool = False):
        """grabs winnowed frame, prints summary stats if requested, and saves it"""
        df = self.prepare_winnowed_frame()
        df.sink_parquet(
            self.processed_data_home / "held_out_for_inference.parquet",
            engine="streaming",
        )
        if verbose:
            logger = Logger()
            logger.summarize_thresholded(df, self.cfg.outcome_tokens)


if __name__ == "__main__":
    self = Winnower()
    self.save_all(verbose=True)
    # breakpoint()
