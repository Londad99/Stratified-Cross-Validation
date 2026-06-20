"""Tests de la verificación de integridad."""

import numpy as np
import pandas as pd

from skfold_kge import StratifiedPartitioner
from skfold_kge.partition import FoldSet
from skfold_kge.verify import build_integrity_report


def _toy_triples():
    rows = []
    for r in ("REL_A", "REL_B"):
        for i in range(10):
            rows.append({"s": f"s{i}", "r": r, "o": f"o{i}"})
    return pd.DataFrame(rows)


def test_clean_partition_passes():
    df = _toy_triples()
    folds = StratifiedPartitioner(k=5, stratify_by="r", seed=42).fit_transform(df)
    report = folds.verify()
    assert report.passed
    assert report.overlap_count == 0
    assert report.data["checks"]["full_coverage"]


def test_detects_injected_overlap():
    df = _toy_triples().reset_index(drop=True)
    # Construye manualmente folds con la fila 0 repetida en dos folds.
    bad_folds = [[0, 1, 2], [0, 3, 4]]  # posición 0 en ambos
    fs = FoldSet(
        clean_df=df,
        folds=bad_folds,
        stratify_by="r",
        k=2,
        seed=0,
        triple_columns=("s", "r", "o"),
    )
    report = build_integrity_report(fs)
    assert report.overlap_count >= 1
    assert not report.passed


def test_detects_nan_stratum():
    df = _toy_triples()
    df.loc[0, "r"] = np.nan  # introduce un estrato faltante
    folds = StratifiedPartitioner(
        k=5, stratify_by="r", seed=42, dropna_stratum=False
    ).fit_transform(df)
    report = folds.verify()
    assert report.data["has_na_stratum"]
    assert report.data["na_count"] == 1
    assert any("NaN" in w for w in report.warnings)


def test_dropna_stratum_removes_na():
    df = _toy_triples()
    df.loc[0, "r"] = np.nan
    folds = StratifiedPartitioner(
        k=5, stratify_by="r", seed=42, dropna_stratum=True
    ).fit_transform(df)
    report = folds.verify()
    assert not report.data["has_na_stratum"]
    assert folds.n_na_removed == 1


def test_entity_overlap_present_in_triple_mode():
    df = _toy_triples()
    folds = StratifiedPartitioner(k=5, stratify_by="r", seed=42).fit_transform(df)
    report = folds.verify()
    assert report.data["is_triples"]
    assert report.data["entity_overlap"] is not None
    assert len(report.data["entity_overlap"]["pairs"]) == 10  # C(5,2)


def test_distribution_has_std_and_cv():
    df = _toy_triples()
    folds = StratifiedPartitioner(k=5, stratify_by="r", seed=42).fit_transform(df)
    report = folds.verify()
    for item in report.data["distribution"]:
        assert "std" in item and "cv" in item and "total" in item
        assert item["total"] == sum(item["per_fold"])
