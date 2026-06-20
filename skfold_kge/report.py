"""Renderizado del :class:`~skfold_kge.verify.IntegrityReport`.

Tres formatos:

* :func:`render_text`  — texto plano para consola / logs.
* :func:`render_json`  — JSON serializable (para integraciones).
* :func:`render_html`  — **dashboard HTML estático y autocontenido** (CSS
  embebido, sin peticiones externas): se abre directamente en el navegador.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .verify import IntegrityReport


# ====================================================================== #
# Utilidades de tabla ASCII (sin dependencias externas)
# ====================================================================== #
def _ascii_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cols = [str(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(c) for c in cols]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(cols), sep]
    out.extend(fmt(r) for r in str_rows)
    return "\n".join(out)


# ====================================================================== #
# Texto
# ====================================================================== #
def render_text(report: "IntegrityReport") -> str:
    d = report.data
    lines: List[str] = []
    add = lines.append

    add("=" * 70)
    add("  REPORTE DE INTEGRIDAD — Validación Cruzada Estratificada")
    add("=" * 70)
    add(f"Columna de estrato : {d['stratify_by']}")
    add(f"Folds (k)          : {d['k']}      Semilla: {d['seed']}")
    add(f"Filas de entrada   : {d['dataset_rows_input']:,}")
    add(f"  Duplicados elim. : {d['dedup_removed']:,}")
    if d["na_removed"]:
        add(f"  NaN eliminados   : {d['na_removed']:,}")
    add(f"Filas limpias      : {d['dataset_rows_clean']:,}")
    add(f"Estratos           : {d['n_strata']}")
    add(f"Proporción train/test: {d['train_test_split']}")
    add("")

    status = "[PASS]" if report.passed else "[FAIL]"
    add(f"RESULTADO GLOBAL: {status}")
    add(f"  [{'OK' if d['checks']['no_overlap'] else 'XX'}] Sin solapamiento entre folds (overlap={d['overlap_count']})")
    add(f"  [{'OK' if d['checks']['full_coverage'] else 'XX'}] Cobertura total (cada fila en exactamente un fold)")
    add("")

    add("Tamaño de folds:")
    sizes = d["fold_sizes"]
    add("  " + "  ".join(f"F{i + 1}={s}" for i, s in enumerate(sizes)))
    st = d["fold_size_stats"]
    add(f"  Media={st['mean']}  Std={st['std']}  CV={st['cv']}%")
    add("")

    add("Distribución por estrato (recuento por fold, con Std y CV):")
    headers = ["Estrato", "Total"] + [f"F{i + 1}" for i in range(d["k"])] + ["Media", "Std", "CV%"]
    rows = []
    for item in d["distribution"]:
        rows.append(
            [item["stratum"], item["total"]]
            + item["per_fold"]
            + [item["mean"], item["std"], item["cv"]]
        )
    add(_ascii_table(headers, rows))
    add("")

    if d["entity_overlap"]:
        add("Entidades por fold (modo tripletas):")
        epf = d["entity_overlap"]["entities_per_fold"]
        add("  " + "  ".join(f"F{i + 1}={n}" for i, n in enumerate(epf)))
        add("  Entidades compartidas entre pares (esperado en KGs):")
        for p in d["entity_overlap"]["pairs"]:
            add(f"    Par {p['pair']}: {p['common']}")
        add("")

    if d["warnings"]:
        add("AVISOS:")
        for w in d["warnings"]:
            add(f"  [!] {w}")
    else:
        add("Sin avisos.")
    add("=" * 70)
    return "\n".join(lines)


# ====================================================================== #
# JSON
# ====================================================================== #
def render_json(report: "IntegrityReport", path: Optional[str] = None, indent: int = 2) -> str:
    text = json.dumps(report.data, indent=indent, ensure_ascii=False)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


# ====================================================================== #
# HTML (dashboard estático)
# ====================================================================== #
_CSS = """
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--muted:#94a3b8;
--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;--accent:#60a5fa;--line:#334155;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 64px;}
h1{font-size:1.6rem;margin:0 0 4px;}
h2{font-size:1.15rem;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px;}
.sub{color:var(--muted);margin:0 0 24px;font-size:.92rem;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;}
.card .v{font-size:1.5rem;font-weight:700;}
.card .l{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;}
.badges{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;}
.badge{display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;
font-weight:600;font-size:.9rem;border:1px solid var(--line);background:var(--card);}
.badge.ok{color:var(--ok);} .badge.bad{color:var(--bad);} .badge.warn{color:var(--warn);}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.dot.ok{background:var(--ok);} .dot.bad{background:var(--bad);} .dot.warn{background:var(--warn);}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:8px;}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);}
th:first-child,td:first-child{text-align:left;}
thead th{color:var(--muted);font-weight:600;border-bottom:2px solid var(--line);}
tbody tr:hover{background:rgba(96,165,250,.06);}
.cv-ok{color:var(--ok);} .cv-warn{color:var(--warn);} .cv-bad{color:var(--bad);}
.bar-row{display:flex;align-items:center;gap:12px;margin:6px 0;}
.bar-row .name{width:56px;color:var(--muted);font-size:.85rem;}
.bar{flex:1;background:#0b1220;border-radius:6px;overflow:hidden;height:22px;border:1px solid var(--line);}
.bar > span{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.bar-row .val{width:54px;text-align:right;font-variant-numeric:tabular-nums;}
.warnbox{background:rgba(245,158,11,.08);border:1px solid var(--warn);border-radius:10px;
padding:12px 16px;margin-top:8px;}
.warnbox ul{margin:6px 0 0;padding-left:20px;} .warnbox li{margin:4px 0;}
.okbox{color:var(--ok);}
footer{color:var(--muted);font-size:.8rem;margin-top:40px;text-align:center;}
"""


def _cv_class(cv: float) -> str:
    if cv < 5:
        return "cv-ok"
    if cv < 15:
        return "cv-warn"
    return "cv-bad"


def _bars(items: List[tuple]) -> str:
    """``items`` = [(nombre, valor, max)] → filas de barras CSS."""
    out = []
    for name, val, mx in items:
        pct = (val / mx * 100) if mx else 0
        out.append(
            f'<div class="bar-row"><span class="name">{html.escape(str(name))}</span>'
            f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'
            f'<span class="val">{val}</span></div>'
        )
    return "\n".join(out)


def _metrics_section(metrics: Dict[str, Any]) -> str:
    models = metrics.get("models", [])
    keys = metrics.get("metric_keys", [])
    avg = metrics.get("avg", {})
    std = metrics.get("std", {})
    if not models or not keys:
        return ""
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = []
    for m in models:
        cells = "".join(
            f"<td>{avg.get(m, {}).get(k, float('nan')):.4f} "
            f"<span style='color:#94a3b8'>± {std.get(m, {}).get(k, 0):.3f}</span></td>"
            for k in keys
        )
        body.append(f"<tr><td>{html.escape(str(m))}</td>{cells}</tr>")
    note = ""
    if metrics.get("epochs"):
        note = (
            f"<p class='sub'>k={metrics.get('k', '?')} folds · "
            f"{metrics.get('epochs')} epochs · dim={metrics.get('dim', '?')}. "
            f"Valor = promedio ± desviación estándar entre folds.</p>"
        )
    return (
        "<h2>Métricas de evaluación (promedio k-fold ± Std)</h2>"
        + note
        + f"<table><thead><tr><th>Modelo</th>{head}</tr></thead>"
        + f"<tbody>{''.join(body)}</tbody></table>"
    )


def render_html(
    report: "IntegrityReport",
    path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    d = report.data
    esc = html.escape

    # Tarjetas resumen
    cards = [
        ("Filas limpias", f"{d['dataset_rows_clean']:,}"),
        ("Folds (k)", d["k"]),
        ("Estratos", d["n_strata"]),
        ("Duplicados elim.", f"{d['dedup_removed']:,}"),
        ("Train / Test", d["train_test_split"]),
        ("CV tamaño folds", f"{d['fold_size_stats']['cv']}%"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="v">{esc(str(v))}</div>'
        f'<div class="l">{esc(l)}</div></div>'
        for l, v in cards
    )

    # Badges de chequeos
    def badge(ok: bool, label: str) -> str:
        cls = "ok" if ok else "bad"
        dot = "ok" if ok else "bad"
        return f'<span class="badge {cls}"><span class="dot {dot}"></span>{esc(label)}</span>'

    badges = (
        badge(d["checks"]["no_overlap"], f"Sin solapamiento (overlap={d['overlap_count']})")
        + badge(d["checks"]["full_coverage"], "Cobertura total")
    )
    if d["has_na_stratum"]:
        badges += (
            f'<span class="badge warn"><span class="dot warn"></span>'
            f'Estrato NaN: {d["na_count"]}</span>'
        )

    # Tabla de distribución
    dist_head = (
        "<th>Estrato</th><th>Total</th>"
        + "".join(f"<th>F{i + 1}</th>" for i in range(d["k"]))
        + "<th>Media</th><th>Std</th><th>CV</th>"
    )
    dist_rows = []
    for item in d["distribution"]:
        per_fold = "".join(f"<td>{c}</td>" for c in item["per_fold"])
        cv_cls = _cv_class(item["cv"])
        dist_rows.append(
            f"<tr><td>{esc(str(item['stratum']))}</td><td>{item['total']}</td>{per_fold}"
            f"<td>{item['mean']}</td><td>{item['std']}</td>"
            f"<td class='{cv_cls}'>{item['cv']}%</td></tr>"
        )
    dist_table = (
        f"<table><thead><tr>{dist_head}</tr></thead>"
        f"<tbody>{''.join(dist_rows)}</tbody></table>"
    )

    # Barras de tamaño de fold
    sizes = d["fold_sizes"]
    mx = max(sizes) if sizes else 1
    size_bars = _bars([(f"Fold {i + 1}", s, mx) for i, s in enumerate(sizes)])

    # Entidades (tripletas)
    entity_html = ""
    if d["entity_overlap"]:
        epf = d["entity_overlap"]["entities_per_fold"]
        pair_rows = "".join(
            f"<tr><td>Par {esc(p['pair'])}</td><td>{p['common']}</td></tr>"
            for p in d["entity_overlap"]["pairs"]
        )
        ent_cards = "  ".join(f"F{i + 1}={n}" for i, n in enumerate(epf))
        entity_html = (
            "<h2>Solapamiento de entidades (modo tripletas)</h2>"
            f"<p class='sub'>Entidades únicas por fold: {esc(ent_cards)}. "
            "El solapamiento de entidades es esperado e informativo en grafos de conocimiento.</p>"
            "<table><thead><tr><th>Par de folds</th><th>Entidades comunes</th></tr></thead>"
            f"<tbody>{pair_rows}</tbody></table>"
        )

    # Avisos
    if d["warnings"]:
        items = "".join(f"<li>{esc(w)}</li>" for w in d["warnings"])
        warn_html = f'<div class="warnbox"><strong>⚠ Avisos</strong><ul>{items}</ul></div>'
    else:
        warn_html = '<p class="okbox">✔ Sin avisos: partición limpia.</p>'

    metrics_html = _metrics_section(metrics) if metrics else ""

    doc = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte de Integridad — {esc(str(d['stratify_by']))}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Reporte de Integridad de la Partición</h1>
<p class="sub">Validación cruzada estratificada por <strong>{esc(str(d['stratify_by']))}</strong>
· k={d['k']} · semilla={d['seed']}</p>

<div class="cards">{cards_html}</div>

<h2>Chequeos de integridad</h2>
<div class="badges">{badges}</div>

<h2>Distribución por estrato (Std / CV sustentan la cantidad de datos)</h2>
<p class="sub">CV &lt; 5% (verde) = estratificación casi perfecta · 5–15% (ámbar) ·
&gt; 15% (rojo) = revisar. La columna CV justifica que cada escenario reciba
datos proporcionales en todos los folds.</p>
{dist_table}

<h2>Balance de tamaño de folds</h2>
<p class="sub">Media={d['fold_size_stats']['mean']} · Std={d['fold_size_stats']['std']} ·
CV={d['fold_size_stats']['cv']}%</p>
{size_bars}

{entity_html}

{metrics_html}

<h2>Avisos</h2>
{warn_html}

<footer>Generado por skfold-kge · dashboard estático autocontenido</footer>
</div></body></html>"""

    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    return doc
