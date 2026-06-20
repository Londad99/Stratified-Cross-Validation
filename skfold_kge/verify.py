"""Verificación de integridad de una partición estratificada.

Produce un :class:`IntegrityReport` con las comprobaciones que el documento
exige para el entregable:

* **Sin solapamiento:** ninguna fila aparece en más de un fold.
* **Cobertura total:** cada fila del dataset limpio cae en exactamente un fold.
* **Distribución estratificada:** recuento por estrato y fold, con
  **desviación estándar (Std)** y **coeficiente de variación (CV)** que
  sustentan la cantidad de datos elegida por escenario.
* **Balance de folds:** Std/CV del tamaño de los folds.
* **Solapamiento de entidades:** (solo grafos) entidades compartidas entre
  folds — esperado en KGs e informativo, no un error.
* **Calidad del dato:** detección de estratos faltantes (NaN).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .partition import FoldSet


def _is_na(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _display(value: object) -> str:
    return "NaN" if _is_na(value) else str(value)


class IntegrityReport:
    """Resultado de la verificación de una partición.

    Los datos calculados quedan en el atributo :attr:`data` (estructura
    serializable a JSON). Use :meth:`to_text`, :meth:`to_json` y
    :meth:`to_html` para renderizarlo.
    """

    def __init__(self, data: Dict[str, Any], foldset: "FoldSet") -> None:
        self.data = data
        self._foldset = foldset

    # ------------------------------------------------------------------ #
    # Atajos de lectura
    # ------------------------------------------------------------------ #
    @property
    def passed(self) -> bool:
        """``True`` si no hay solapamiento y la cobertura es total."""
        return bool(self.data["checks"]["no_overlap"] and self.data["checks"]["full_coverage"])

    @property
    def overlap_count(self) -> int:
        return int(self.data["overlap_count"])

    @property
    def warnings(self) -> List[str]:
        return list(self.data["warnings"])

    # ------------------------------------------------------------------ #
    # Renderizado (delegado a report.py)
    # ------------------------------------------------------------------ #
    def to_text(self) -> str:
        """Devuelve el reporte como texto plano legible en consola."""
        from .report import render_text

        return render_text(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serializa el reporte a JSON. Si ``path`` se da, también lo escribe."""
        from .report import render_json

        return render_json(self, path=path, indent=indent)

    def to_html(self, path: Optional[str] = None, metrics: Optional[dict] = None) -> str:
        """Genera el dashboard HTML estático. Si ``path`` se da, lo escribe.

        Parameters
        ----------
        path : str, optional
            Ruta de salida del ``.html``.
        metrics : dict, optional
            Resultados de evaluación de modelos para añadir una sección de
            métricas (ver :mod:`skfold_kge.evaluate`).
        """
        from .report import render_html

        return render_html(self, path=path, metrics=metrics)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"<IntegrityReport {status} k={self.data['k']} "
            f"strata={self.data['n_strata']} overlap={self.overlap_count} "
            f"warnings={len(self.warnings)}>"
        )


def build_integrity_report(foldset: "FoldSet") -> IntegrityReport:
    """Calcula todas las comprobaciones de integridad de ``foldset``."""
    df = foldset.clean_df
    folds = foldset.folds
    k = foldset.k
    stratify_by = foldset.stratify_by
    n_clean = len(df)

    # --- Cobertura y unicidad de índices -------------------------------- #
    all_positions = [p for f in folds for p in f]
    unique_positions = set(all_positions)
    full_coverage = len(all_positions) == len(unique_positions) == n_clean

    # --- Solapamiento exacto de contenido entre folds ------------------- #
    seen: Dict[tuple, int] = {}
    overlap_count = 0
    for i, fold in enumerate(folds):
        for row in df.iloc[fold].itertuples(index=False, name=None):
            if row in seen and seen[row] != i:
                overlap_count += 1
            else:
                seen.setdefault(row, i)

    # --- Tamaño de folds ------------------------------------------------- #
    fold_sizes = [len(f) for f in folds]
    size_mean = float(np.mean(fold_sizes))
    size_std = float(np.std(fold_sizes))
    size_cv = float(size_std / size_mean * 100) if size_mean else 0.0

    # --- Distribución por estrato y fold (Std / CV) --------------------- #
    fold_strata: List[Counter] = []
    for fold in folds:
        vals = df[stratify_by].iloc[fold]
        fold_strata.append(Counter(_display(v) for v in vals))

    strata_keys = sorted(set().union(*[set(c) for c in fold_strata]) if fold_strata else set())
    distribution: List[Dict[str, Any]] = []
    for s in strata_keys:
        counts = [int(fold_strata[i].get(s, 0)) for i in range(k)]
        total = int(sum(counts))
        mean = float(np.mean(counts))
        std = float(np.std(counts))
        cv = float(std / mean * 100) if mean else 0.0
        distribution.append(
            {
                "stratum": s,
                "total": total,
                "per_fold": counts,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cv": round(cv, 4),
            }
        )

    n_strata = len(strata_keys)
    has_na_stratum = "NaN" in strata_keys
    na_count = int(next((d["total"] for d in distribution if d["stratum"] == "NaN"), 0))

    # --- Solapamiento de entidades (solo tripletas) --------------------- #
    entity_overlap: Optional[Dict[str, Any]] = None
    if foldset.triple_columns is not None:
        s_col, _, o_col = foldset.triple_columns
        fold_entities: List[set] = []
        for fold in folds:
            sub = df.iloc[fold]
            ents = set(sub[s_col].astype(str)) | set(sub[o_col].astype(str))
            fold_entities.append(ents)
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                common = len(fold_entities[i] & fold_entities[j])
                pairs.append({"pair": f"{i + 1}-{j + 1}", "common": int(common)})
        entity_overlap = {
            "entities_per_fold": [int(len(e)) for e in fold_entities],
            "pairs": pairs,
        }

    # --- Avisos ---------------------------------------------------------- #
    warnings: List[str] = []
    if has_na_stratum:
        warnings.append(
            f"{na_count} fila(s) con estrato faltante (NaN) agrupadas en un "
            f"estrato propio. Considere limpiarlas o usar dropna_stratum=True."
        )
    sparse = [d["stratum"] for d in distribution if 0 < d["total"] < k]
    if sparse:
        warnings.append(
            f"Estratos con menos de k={k} ejemplos (no presentes en todos los "
            f"folds): {', '.join(sparse)}."
        )
    if size_cv > 5:
        warnings.append(
            f"CV del tamaño de folds = {size_cv:.1f}% (> 5%): folds desbalanceados."
        )

    data: Dict[str, Any] = {
        "dataset_rows_input": int(foldset.n_input),
        "dedup_removed": int(foldset.n_dedup_removed),
        "na_removed": int(foldset.n_na_removed),
        "dataset_rows_clean": int(n_clean),
        "k": int(k),
        "seed": int(foldset.seed),
        "stratify_by": str(stratify_by),
        "n_strata": int(n_strata),
        "has_na_stratum": bool(has_na_stratum),
        "na_count": na_count,
        "fold_sizes": fold_sizes,
        "fold_size_stats": {
            "mean": round(size_mean, 4),
            "std": round(size_std, 4),
            "cv": round(size_cv, 4),
        },
        "train_test_split": f"{(k - 1) * 100 // k}/{100 // k}",
        "overlap_count": int(overlap_count),
        "checks": {
            "no_overlap": overlap_count == 0,
            "full_coverage": bool(full_coverage),
        },
        "distribution": distribution,
        "entity_overlap": entity_overlap,
        "warnings": warnings,
        "is_triples": foldset.triple_columns is not None,
        "triple_columns": list(foldset.triple_columns) if foldset.triple_columns else None,
    }
    return IntegrityReport(data, foldset)
