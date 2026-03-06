import datetime
import tempfile

import polars as pl
import pytest
from omegaconf import OmegaConf

from cocoa.collator import Collator


@pytest.fixture()
def collator():
    """Create a Collator with two fake tables written to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = __import__("pathlib").Path(tmp_dir)

        # -- reference table: 4 hospitalizations across 2 patients --
        ref = pl.DataFrame(
            {
                "hosp_id": [100, 101, 102, 103],
                "patient_id": ["P1", "P1", "P2", "P2"],
                "admit_time": [
                    datetime.datetime(2025, 1, 1),
                    datetime.datetime(2025, 3, 1),
                    datetime.datetime(2025, 2, 1),
                    datetime.datetime(2025, 4, 1),
                ],
                "discharge_time": [
                    datetime.datetime(2025, 1, 10),
                    datetime.datetime(2025, 3, 10),
                    datetime.datetime(2025, 2, 10),
                    datetime.datetime(2025, 4, 10),
                ],
                "sex": ["male", "male", "female", "female"],
            }
        )
        ref.write_parquet(tmp_path / "admissions.parquet")

        # -- event table: labs for these hospitalizations --
        labs = pl.DataFrame(
            {
                "hosp_id": [100, 100, 101, 102, 103, 103],
                "lab_name": [
                    "glucose",
                    "creatinine",
                    "glucose",
                    "hemoglobin",
                    "glucose",
                    "hemoglobin",
                ],
                "lab_value": [120.0, 1.2, 95.0, 13.5, 110.0, 11.0],
                "lab_time": [
                    datetime.datetime(2025, 1, 2),
                    datetime.datetime(2025, 1, 3),
                    datetime.datetime(2025, 3, 2),
                    datetime.datetime(2025, 2, 5),
                    datetime.datetime(2025, 4, 2),
                    datetime.datetime(2025, 4, 3),
                ],
            }
        )
        labs.write_parquet(tmp_path / "labs.parquet")

        # -- build config that mirrors the real structure --
        cfg = OmegaConf.create(
            {
                "data_home": str(tmp_path),
                "processed_data_home": str(tmp_path / "out"),
                "subject_id": "hosp_id",
                "group_id": "patient_id",
                "subject_splits": {"train_frac": 0.5, "tuning_frac": 0.25},
                "reference": {
                    "table": "admissions",
                    "start_time": "admit_time",
                    "end_time": "discharge_time",
                },
                "entries": [
                    {
                        "table": "REFERENCE",
                        "prefix": "SEX",
                        "code": "sex",
                        "time": "admit_time",
                    },
                    {
                        "table": "labs",
                        "prefix": "LAB",
                        "code": "lab_name",
                        "numeric_value": "lab_value",
                        "time": "lab_time",
                    },
                ],
            }
        )

        c = object.__new__(Collator)
        c.cfg = cfg
        c.data_home = tmp_path
        c.reference_frame = None
        yield c


def test_get_entry_from_reference(collator):
    """Static entries pulled from the reference table have the right shape."""
    result = collator.get_entry(
        table="REFERENCE", prefix="SEX", code="sex", time="admit_time"
    ).collect()
    assert result.columns == [
        "subject_id",
        "time",
        "code",
        "numeric_value",
        "text_value",
    ]
    assert len(result) == 4
    assert set(result["code"]) == {"SEX//male", "SEX//female"}
    assert result["numeric_value"].is_null().all()


def test_get_entry_from_external_table(collator):
    """Event entries from an external table carry numeric values through."""
    result = collator.get_entry(
        table="labs",
        prefix="LAB",
        code="lab_name",
        numeric_value="lab_value",
        time="lab_time",
    ).collect()
    assert len(result) == 6
    assert result["numeric_value"].is_null().sum() == 0
    assert "LAB//glucose" in result["code"].to_list()


def test_get_entry_with_filter(collator):
    """filter_expr restricts rows before extraction."""
    result = collator.get_entry(
        table="labs",
        prefix="LAB",
        code="lab_name",
        numeric_value="lab_value",
        time="lab_time",
        filter_expr='pl.col("lab_name") == "glucose"',
    ).collect()
    assert set(result["code"]) == {"LAB//glucose"}
    assert len(result) == 3


def test_get_entry_with_col_expr(collator):
    """with_col_expr adds a computed column usable as the code."""
    result = collator.get_entry(
        table="labs",
        prefix="LAB",
        code="flag",
        time="lab_time",
        with_col_expr='pl.lit("high_value").alias("flag")',
        filter_expr='pl.col("lab_value") > 100',
    ).collect()
    assert set(result["code"]) == {"LAB//high_value"}
    assert len(result) == 2  # glucose 120 and glucose 110


def test_get_all_concatenates_entries(collator):
    """get_all produces the union of all configured entries."""
    result = collator.get_all().collect()
    # 4 sex rows + 6 lab rows
    assert len(result) == 10
    prefixes = result["code"].str.split("//").list[0].unique().sort()
    assert prefixes.to_list() == ["LAB", "SEX"]


def test_subject_splits_covers_all_subjects(collator):
    """Every subject gets exactly one split label."""
    df_all = collator.get_all()
    splits = collator.get_subject_splits(df_all)
    assert set(splits.columns) == {"subject_id", "split"}
    assert len(splits) == 4
    assert set(splits["split"]) <= {"train", "tuning", "held_out"}
    # no duplicates
    assert splits["subject_id"].n_unique() == len(splits)


def test_subject_splits_respects_fractions(collator):
    """Split fractions roughly match the configured ratios."""
    df_all = collator.get_all()
    splits = collator.get_subject_splits(df_all)
    counts = splits.group_by("split").len().sort("split")
    split_map = dict(zip(counts["split"].to_list(), counts["len"].to_list()))
    # 4 subjects: 50% train → 2, 25% tuning → 1, rest held_out → 1
    assert split_map["train"] == 2
    assert split_map["tuning"] == 1
    assert split_map["held_out"] == 1


def test_save_all_writes_files(collator):
    """save_all produces the expected output files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = __import__("pathlib").Path(tmp_dir) / "output"
        collator.save_all(path=out)
        assert (out / "meds.parquet").exists()
        assert (out / "subject_splits.parquet").exists()
        meds = pl.read_parquet(out / "meds.parquet")
        assert len(meds) == 10
