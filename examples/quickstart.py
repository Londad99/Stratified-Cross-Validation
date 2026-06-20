"""Quickstart de skfold-kge: particionar, verificar y exportar un KG.

Ejecutar desde la raíz del repositorio::

    python examples/quickstart.py
"""

import os
import sys

# Permite ejecutar sin instalar el paquete.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from skfold_kge import StratifiedPartitioner

# 1) Cargar el dataset de tripletas (sujeto; relación; objeto).
df = pd.read_csv("datasets/GoT.csv", sep=";")
print(f"Dataset: {len(df):,} filas, columnas {list(df.columns)}")

# 2) Particionar estratificando por la relación (Column2).
folds = StratifiedPartitioner(k=5, stratify_by="Column2", seed=42).fit_transform(df)
print("Tamaño de folds:", folds.sizes())  # [640, 639, 635, 632, 630]

# 3) Verificar integridad (sin solapamiento, cobertura total, Std/CV).
report = folds.verify()
print(report.to_text())

# 4) Iterar pares entrenamiento/prueba.
for i, train, test in folds.iter_train_test():
    print(f"Fold {i + 1}: train={len(train)}  test={len(test)}")

# 5) Exportar el entregable.
os.makedirs("outputs", exist_ok=True)
folds.to_csv_dir("outputs/folds")
folds.to_excel("outputs/folds_partitions.xlsx")
report.to_html("outputs/integrity_report.html")
print("\nListo. Abre outputs/integrity_report.html en el navegador.")
