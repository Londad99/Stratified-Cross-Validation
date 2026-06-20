"""Quickstart de detección de noticias falsas con validación cruzada estratificada.

Requiere el extra de texto::

    pip install -e ".[text]"
    python examples/quickstart_fakenews.py

Por defecto usa un dataset sintético desbalanceado (25% fake / 75% real) para
demostrar por qué el F1 macro es la métrica adecuada. Para datos reales,
reemplaza `build_demo()` por `load_isot(...)`, `load_liar()` o `load_welfake(...)`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from skfold_kge.evaluate import cross_validate_text


def build_demo(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fake = [
        f"Shocking: politician accused of {rng.choice(['fraud', 'corruption', 'scandal'])} "
        f"in {rng.choice(['2023', '2024', '2025'])} fake report {i}"
        for i in range(500)
    ]
    real = [
        f"{rng.choice(['Reuters', 'AP', 'BBC'])} reports: "
        f"{rng.choice(['GDP growth', 'climate deal', 'tech summit'])} update {i}"
        for i in range(1500)
    ]
    df = pd.DataFrame(
        {"text": fake + real, "label": [0] * 500 + [1] * 1500}
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = build_demo()
    print(f"Dataset: {len(df):,} noticias | "
          f"fake={int((df.label == 0).sum())} (25%) real={int((df.label == 1).sum())} (75%)")

    res = cross_validate_text(df, text_col="text", label_col="label", k=5, seed=42)

    print("\nDistribución de clases:", res["class_distribution"])
    print("F1 macro (métrica primaria):", res["summary"]["F1"])
