from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    pass


def _require_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, precision_score, recall_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cross_validate_text requiere el extra de texto. Instale con:\n"
            "    pip install skfold-kge[text]\n"
            f"(causa: {exc})"
        ) from exc
    return TfidfVectorizer, LogisticRegression, f1_score, precision_score, recall_score


def cross_validate_text(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    k: int = 5,
    seed: int = 42,
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
    verbose: bool = True,
) -> Dict[str, object]:

    from ..partition import StratifiedPartitioner

    (
        TfidfVectorizer,
        LogisticRegression,
        f1_score,
        precision_score,
        recall_score,
    ) = _require_sklearn()

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Se requieren columnas {text_col!r} y {label_col!r}.")

    
    foldset = StratifiedPartitioner(
        k=k, stratify_by=label_col, seed=seed, dedup=False
    ).fit_transform(df)

    per_fold: List[Dict[str, float]] = []
    for i, train, test in foldset.iter_train_test():
        vec = TfidfVectorizer(
            max_features=max_features, sublinear_tf=True, ngram_range=ngram_range
        )
        x_train = vec.fit_transform(train[text_col].astype(str))
        x_test = vec.transform(test[text_col].astype(str))
        y_train = train[label_col].to_numpy()
        y_test = test[label_col].to_numpy()

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
        clf.fit(x_train, y_train)
        y_hat = clf.predict(x_test)

        rec = {
            "F1": float(f1_score(y_test, y_hat, average="macro")),
            "Precision": float(precision_score(y_test, y_hat, average="macro", zero_division=0)),
            "Recall": float(recall_score(y_test, y_hat, average="macro", zero_division=0)),
            "Accuracy": float((y_hat == y_test).mean()),
        }
        per_fold.append(rec)
        if verbose:
            print(
                f"  Fold {i + 1}: F1={rec['F1']:.4f}  Prec={rec['Precision']:.4f}  "
                f"Rec={rec['Recall']:.4f}  Acc={rec['Accuracy']:.4f}"
            )

    keys = ["F1", "Precision", "Recall", "Accuracy"]
    summary = {}
    for key in keys:
        vals = [r[key] for r in per_fold]
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        summary[key] = {
            "mean": round(mean, 4),
            "std": round(sd, 4),
            "cv": round(sd / mean * 100, 4) if mean else 0.0,
        }

    vc = df[label_col].value_counts().sort_index()
    class_distribution = {str(c): int(n) for c, n in vc.items()}

    if verbose:
        print("\nResumen (F1 macro es la métrica primaria):")
        for key in keys:
            s = summary[key]
            print(f"  {key:10s}: {s['mean']:.4f} ± {s['std']:.4f}  (CV={s['cv']:.1f}%)")

    return {
        "per_fold": per_fold,
        "summary": summary,
        "class_distribution": class_distribution,
        "k": k,
    }

# Loaders de datasets públicos de noticias falsas
def load_isot(
    fake_path: str = "isot/Fake.csv", real_path: str = "isot/True.csv"
) -> pd.DataFrame:
    """Carga el dataset ISOT (Kaggle) y devuelve columnas ``text`` y ``label``.

    Descargar de:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    (``label``: 0 = fake, 1 = real).
    """
    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 0
    df_real = pd.read_csv(real_path)
    df_real["label"] = 1
    df = pd.concat([df_fake, df_real], ignore_index=True)
    title = df["title"].fillna("") if "title" in df.columns else ""
    body = df["text"].fillna("") if "text" in df.columns else ""
    df["text"] = (title + " " + body).str.strip()
    return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)


_LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
_LIAR_FAKE_LABELS = {"pants-fire", "false", "barely-true"}
_LIAR_REAL_LABELS = {"half-true", "mostly-true", "true"}


def load_liar(cache_dir: str = ".cache/liar") -> pd.DataFrame:
    import os
    import urllib.request
    import zipfile

    os.makedirs(cache_dir, exist_ok=True)
    extract_dir = os.path.join(cache_dir, "liar_dataset")
    train_tsv = os.path.join(extract_dir, "train.tsv")

    if not os.path.exists(train_tsv):
        zip_path = os.path.join(cache_dir, "liar_dataset.zip")
        urllib.request.urlretrieve(_LIAR_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    all_labels = _LIAR_FAKE_LABELS | _LIAR_REAL_LABELS
    frames = []
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        part = pd.read_csv(
            os.path.join(extract_dir, split), sep="\t", header=None, usecols=[1, 2]
        )
        part.columns = ["label", "statement"]
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(all_labels)].copy()
    df["text"] = df["statement"].astype(str)
    df["label"] = df["label"].map(lambda lbl: 0 if lbl in _LIAR_FAKE_LABELS else 1)
    return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)


def load_welfake(path: str = "WELFake_Dataset.csv") -> pd.DataFrame:
    """Carga WELFake (Kaggle) y devuelve columnas ``text`` y ``label``.

    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
    """
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
