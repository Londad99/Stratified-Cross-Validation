"""skfold-kge — Validación cruzada k-fold estratificada.

Particionado estratificado (Round-Robin) con verificación de integridad y
reportes, para grafos de conocimiento (estratificando por relación) y para
clasificación de texto / detección de noticias falsas (estratificando por
clase).

Ejemplo mínimo
--------------
>>> import pandas as pd
>>> from skfold_kge import StratifiedPartitioner
>>> df = pd.read_csv("datasets/GoT.csv", sep=";")
>>> folds = StratifiedPartitioner(k=5, stratify_by="Column2", seed=42).fit_transform(df)
>>> report = folds.verify()
>>> report.passed
True
>>> report.to_html("outputs/integrity_report.html")  # doctest: +SKIP

Los extras de evaluación viven en :mod:`skfold_kge.evaluate`.
"""

from __future__ import annotations

from .io import export_folds_csv_dir, export_folds_excel, load_triples
from .metrics import compute_filtered_ranks, compute_metrics_from_ranks, f1_pairwise
from .partition import FoldSet, StratifiedPartitioner, partition
from .verify import IntegrityReport, build_integrity_report

__version__ = "0.1.0"

__all__ = [
    "StratifiedPartitioner",
    "FoldSet",
    "partition",
    "IntegrityReport",
    "build_integrity_report",
    "load_triples",
    "export_folds_excel",
    "export_folds_csv_dir",
    "compute_filtered_ranks",
    "compute_metrics_from_ranks",
    "f1_pairwise",
    "__version__",
]
