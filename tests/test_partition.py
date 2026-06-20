"""Tests del particionado estratificado."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skfold_kge import StratifiedPartitioner

REPO_ROOT = Path(__file__).resolve().parent.parent
GOT_CSV = REPO_ROOT / "datasets" / "GoT.csv"


def _toy_df(n_per_class=20, classes=("A", "B", "C"), seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c in classes:
        for i in range(n_per_class):
            rows.append({"subj": f"{c}{i}", "rel": c, "obj": f"o{rng.randint(100)}"})
    return pd.DataFrame(rows)


def test_no_overlap_and_full_coverage():
    df = _toy_df()
    folds = StratifiedPartitioner(k=5, stratify_by="rel", seed=42).fit_transform(df)

    positions = [p for f in folds.folds for p in f]
    # Cada posición aparece exactamente una vez.
    assert len(positions) == len(set(positions)) == len(folds.clean_df)
    # Los folds son disjuntos.
    for i in range(folds.k):
        for j in range(i + 1, folds.k):
            assert set(folds.folds[i]).isdisjoint(folds.folds[j])


def test_reproducible_with_seed():
    df = _toy_df()
    a = StratifiedPartitioner(k=5, stratify_by="rel", seed=7).fit_transform(df)
    b = StratifiedPartitioner(k=5, stratify_by="rel", seed=7).fit_transform(df)
    assert a.folds == b.folds


def test_different_seed_changes_partition():
    df = _toy_df()
    a = StratifiedPartitioner(k=5, stratify_by="rel", seed=1).fit_transform(df)
    b = StratifiedPartitioner(k=5, stratify_by="rel", seed=2).fit_transform(df)
    assert a.folds != b.folds


def test_distribution_is_proportional():
    df = _toy_df(n_per_class=20)  # 20 por clase, k=5 -> 4 por fold por clase
    folds = StratifiedPartitioner(k=5, stratify_by="rel", seed=42).fit_transform(df)
    for i in range(folds.k):
        sub = folds.fold_frame(i)
        counts = sub["rel"].value_counts()
        for cls in ("A", "B", "C"):
            assert counts.get(cls, 0) == 4


def test_dedup_removes_duplicates():
    df = _toy_df(n_per_class=5)
    dup = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    folds = StratifiedPartitioner(k=2, stratify_by="rel", dedup=True).fit_transform(dup)
    assert folds.n_dedup_removed == 3
    assert len(folds.clean_df) == len(df)


def test_sizes_sum_to_total():
    df = _toy_df(n_per_class=23)  # no divisible entre 5
    folds = StratifiedPartitioner(k=5, stratify_by="rel", seed=42).fit_transform(df)
    assert sum(folds.sizes()) == len(folds.clean_df)
    assert max(folds.sizes()) - min(folds.sizes()) <= 3  # 3 clases, resto <= 1 c/u


def test_invalid_args():
    df = _toy_df()
    with pytest.raises(ValueError):
        StratifiedPartitioner(k=1, stratify_by="rel")
    with pytest.raises(ValueError):
        StratifiedPartitioner(k=5, stratify_by=None)
    with pytest.raises(KeyError):
        StratifiedPartitioner(k=5, stratify_by="inexistente").fit_transform(df)


@pytest.mark.skipif(not GOT_CSV.exists(), reason="datasets/GoT.csv no disponible")
def test_got_reproduces_notebook_sizes():
    df = pd.read_csv(GOT_CSV, sep=";")
    folds = StratifiedPartitioner(k=5, stratify_by="Column2", seed=42).fit_transform(df)
    assert folds.sizes() == [640, 639, 635, 632, 630]
    report = folds.verify()
    assert report.passed
    assert report.overlap_count == 0
