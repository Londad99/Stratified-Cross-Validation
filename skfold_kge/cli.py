"""Interfaz de línea de comandos de skfold-kge.

Ejemplos
--------
Particionar y generar el entregable completo (folds + reportes)::

    python -m skfold_kge partition datasets/GoT.csv --by Column2 --k 5 \\
        --sep ";" --out outputs --triple-names Subject Relation Object

Verificar/imprimir solo el reporte de integridad::

    python -m skfold_kge partition datasets/GoT.csv --by Column2 --report-only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .io import load_triples
from .partition import StratifiedPartitioner


def _cmd_partition(args: argparse.Namespace) -> int:
    df = load_triples(args.input, sep=args.sep)
    triple_names = tuple(args.triple_names) if args.triple_names else None

    part = StratifiedPartitioner(
        k=args.k,
        stratify_by=args.by,
        seed=args.seed,
        dedup=not args.no_dedup,
        dropna_stratum=args.dropna,
    )
    folds = part.fit_transform(df)
    report = folds.verify()

    print(report.to_text())

    if args.report_only:
        return 0 if report.passed else 1

    os.makedirs(args.out, exist_ok=True)
    folds_dir = os.path.join(args.out, "folds")

    rename = None
    if triple_names and folds.triple_columns:
        rename = dict(zip(folds.triple_columns, triple_names))

    csv_paths = folds.to_csv_dir(folds_dir, **({"rename": rename} if rename else {}))
    xlsx_path = os.path.join(args.out, "folds_partitions.xlsx")
    folds.to_excel(xlsx_path, **({"rename": rename} if rename else {}))

    txt_path = os.path.join(args.out, "integrity_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_text())
    json_path = os.path.join(args.out, "integrity_report.json")
    report.to_json(json_path)
    html_path = os.path.join(args.out, "integrity_report.html")
    report.to_html(html_path)

    print("\nEntregable generado:")
    print(f"  Folds CSV : {folds_dir}  ({len(csv_paths)} archivos)")
    print(f"  Excel     : {xlsx_path}")
    print(f"  Reportes  : {txt_path}")
    print(f"              {json_path}")
    print(f"              {html_path}")
    return 0 if report.passed else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    df = load_triples(args.input, sep=args.sep)
    if args.task == "kge":
        from .evaluate import cross_validate_kge

        folds = StratifiedPartitioner(
            k=args.k, stratify_by=args.by, seed=args.seed
        ).fit_transform(df)
        cross_validate_kge(folds, num_epochs=args.epochs, embedding_dim=args.dim)
    else:  # text
        from .evaluate import cross_validate_text

        cross_validate_text(
            df, text_col=args.text_col, label_col=args.by, k=args.k, seed=args.seed
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skfold-kge",
        description="Validación cruzada k-fold estratificada (KGE / noticias falsas).",
    )
    parser.add_argument("--version", action="version", version=f"skfold-kge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # partition
    p = sub.add_parser("partition", help="Particionar y generar el entregable.")
    p.add_argument("input", help="Ruta o URL del CSV.")
    p.add_argument("--by", required=True, help="Columna de estrato (p. ej. relación o clase).")
    p.add_argument("--k", type=int, default=5, help="Número de folds (def. 5).")
    p.add_argument("--seed", type=int, default=42, help="Semilla (def. 42).")
    p.add_argument("--sep", default=";", help="Separador del CSV (def. ';').")
    p.add_argument("--out", default="outputs", help="Carpeta de salida (def. 'outputs').")
    p.add_argument("--no-dedup", action="store_true", help="No deduplicar filas.")
    p.add_argument("--dropna", action="store_true", help="Descartar filas con estrato NaN.")
    p.add_argument(
        "--triple-names",
        nargs=3,
        metavar=("SUBJECT", "RELATION", "OBJECT"),
        help="Renombrar las 3 columnas en la exportación (modo tripletas).",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Solo imprimir el reporte, sin escribir archivos.",
    )
    p.set_defaults(func=_cmd_partition)

    # evaluate (requiere extras)
    e = sub.add_parser("evaluate", help="Evaluar modelos (requiere extras [kge]/[text]).")
    e.add_argument("input", help="Ruta o URL del CSV.")
    e.add_argument("--task", choices=["kge", "text"], default="kge")
    e.add_argument("--by", required=True, help="Columna de estrato (relación o clase).")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--sep", default=";")
    e.add_argument("--epochs", type=int, default=200, help="(kge) epochs de entrenamiento.")
    e.add_argument("--dim", type=int, default=100, help="(kge) dimensión de embedding.")
    e.add_argument("--text-col", default="text", help="(text) columna de texto.")
    e.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
