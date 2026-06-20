"""Genera el entregable: dataset particionado + verificado + reportes.

Lee ``datasets/GoT.csv``, crea 5 folds estratificados por relación (Column2),
verifica la integridad y escribe en ``outputs/``:

* ``folds/Fold_1.csv`` … ``Fold_5.csv``  (un CSV por fold)
* ``folds_partitions.xlsx``              (un Excel con una hoja por fold)
* ``integrity_report.txt`` / ``.json`` / ``.html``  (reportes de integridad)

Uso::

    python scripts/build_deliverable.py
    python scripts/build_deliverable.py --k 10 --input datasets/GoT.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Permite ejecutar el script sin instalar el paquete (añade la raíz al path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from skfold_kge import StratifiedPartitioner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/GoT.csv", help="CSV de entrada.")
    parser.add_argument("--by", default="Column2", help="Columna de estrato (relación).")
    parser.add_argument("--k", type=int, default=5, help="Número de folds.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla.")
    parser.add_argument("--sep", default=";", help="Separador del CSV.")
    parser.add_argument("--out", default="outputs", help="Carpeta de salida.")
    args = parser.parse_args()

    print(f"Cargando {args.input} …")
    df = pd.read_csv(args.input, sep=args.sep)
    print(f"  {len(df):,} filas, columnas: {list(df.columns)}")

    folds = StratifiedPartitioner(
        k=args.k, stratify_by=args.by, seed=args.seed
    ).fit_transform(df)
    report = folds.verify()

    # Nombres legibles para las tripletas del KG.
    rename = None
    if folds.triple_columns and len(folds.triple_columns) == 3:
        rename = dict(zip(folds.triple_columns, ("Subject", "Relation", "Object")))

    os.makedirs(args.out, exist_ok=True)
    folds_dir = os.path.join(args.out, "folds")
    csv_paths = folds.to_csv_dir(folds_dir, rename=rename)
    xlsx_path = folds.to_excel(os.path.join(args.out, "folds_partitions.xlsx"), rename=rename)

    txt_path = os.path.join(args.out, "integrity_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_text())
    json_path = os.path.join(args.out, "integrity_report.json")
    report.to_json(json_path)
    html_path = os.path.join(args.out, "integrity_report.html")
    report.to_html(html_path)

    print(report.to_text())
    print("\n" + "=" * 60)
    print("ENTREGABLE GENERADO en", os.path.abspath(args.out))
    print("=" * 60)
    print(f"  Folds CSV : {folds_dir}  ({len(csv_paths)} archivos)")
    print(f"  Excel     : {xlsx_path}")
    print(f"  Reportes  : integrity_report.txt / .json / .html")
    print(f"\nResultado de integridad: {'PASS' if report.passed else 'FAIL'} "
          f"(overlap={report.overlap_count})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
