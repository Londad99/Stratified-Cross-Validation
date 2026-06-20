# skfold-kge

**Validación cruzada k-fold estratificada** para grafos de conocimiento (KGE) y
clasificación de texto / detección de noticias falsas — con **verificación de
integridad** y un **dashboard HTML** autocontenido.

[![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-pytest-green)](#testing)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#licencia)

---

## ¿Qué hace?

Particiona un dataset en `k` folds **estratificados por una columna categórica**,
de forma que cada fold conserva la distribución del conjunto original:

| Caso de uso | `stratify_by` | Resultado |
|---|---|---|
| **Grafos de conocimiento (KGE)** | columna de **relación** | Cada fold mantiene la proporción de cada tipo de relación |
| **Noticias falsas / clasificación** | columna de **clase** (`label`) | Cada fold mantiene la proporción de clases |

Además:

- ✔ **Verifica** que no haya solapamiento entre folds y que la cobertura sea total.
- 📊 **Sustenta la cantidad de datos** con desviación estándar (Std) y coeficiente
  de variación (CV) por estrato.
- 📄 Genera reportes de integridad en **texto, JSON y HTML** (dashboard estático).
- 📈 Evalúa modelos KGE (TransE/ComplEx/RotatE) y clasificadores de texto con
  **F1 Score**, MRR y Hits@K (extras opcionales).

---

## Índice

1. [Instalación](#instalación)
2. [Tutorial paso a paso](#tutorial-paso-a-paso)
3. [Parámetros de `StratifiedPartitioner`](#parámetros-de-stratifiedpartitioner)
4. [Cómo leer el reporte de integridad](#cómo-leer-el-reporte-de-integridad)
5. [Generar el entregable completo (CLI)](#generar-el-entregable-completo)
6. [Referencia completa de la CLI](#referencia-completa-de-la-cli)
7. [Metodología](#metodología)
8. [Detección de noticias falsas](#detección-de-noticias-falsas)
9. [Evaluación de modelos KGE](#evaluación-de-modelos-kge)
10. [API](#api)
11. [Preguntas frecuentes / Troubleshooting](#preguntas-frecuentes--troubleshooting)
12. [Testing](#testing)

---

## Instalación

```bash
# Núcleo (particionar + verificar + reportar + exportar): pandas, numpy, openpyxl
pip install -e .

# Extras opcionales
pip install -e ".[kge]"    # evaluación KGE: pykeen + torch (~2 GB)
pip install -e ".[text]"   # clasificación de texto: scikit-learn
pip install -e ".[liar]"   # loader del dataset LIAR (HuggingFace datasets)
pip install -e ".[all]"    # todo
pip install -e ".[dev]"    # pytest
```

> Distribución: `skfold-kge` · Import: `import skfold_kge`

Verifica la instalación:

```bash
python -c "import skfold_kge; print(skfold_kge.__version__)"
skfold-kge --version
```

---

## Tutorial paso a paso

Esta sección explica, de cero, cómo particionar y verificar **tu propio dataset**
(no solo el GoT de ejemplo). Cada paso explica el *por qué*, no solo el *cómo*.

### Paso 1 — Prepara tu dataset

Necesitas un `DataFrame` de pandas con **una columna que sirva de estrato**:
el valor por el que quieres que la proporción se mantenga igual en todos los
folds. Dos casos típicos:

- **Grafo de conocimiento**: 3 columnas `(sujeto, relación, objeto)` → estrato
  = columna de relación.
- **Clasificación** (p. ej. noticias falsas): columna de texto + columna de
  clase → estrato = columna de clase.

```python
import pandas as pd
df = pd.read_csv("mi_dataset.csv", sep=";")   # o sep="," según tu archivo
print(df.head())
print(df.shape)
```

> **Importante:** si tu CSV usa coma como separador, no pases `sep=";"`. Si no
> sabes el separador, abre el archivo con un editor de texto y mira la primera línea.

### Paso 2 — Elige `k` y la columna de estrato

```python
from skfold_kge import StratifiedPartitioner

part = StratifiedPartitioner(
    k=5,                     # número de folds (5 es el estándar: 80% train / 20% test)
    stratify_by="Column2",   # nombre EXACTO de tu columna de clase/relación
    seed=42,                 # cualquier entero fijo => resultados reproducibles
)
```

¿Cómo elegir `k`? Más folds = más datos de entrenamiento por iteración pero
folds de prueba más pequeños (más varianza al medir). `k=5` o `k=10` son los
valores estándar en la literatura; usa `k` más grande solo si tu dataset es
pequeño y necesitas exprimir cada ejemplo.

### Paso 3 — Particiona

```python
folds = part.fit_transform(df)
print(folds.sizes())   # p. ej. [640, 639, 635, 632, 630]
```

`fit_transform` hace, en este orden: (1) elimina duplicados, (2) agrupa filas
por el valor del estrato, (3) baraja cada grupo con la semilla dada, (4)
reparte Round-Robin entre los `k` folds. El resultado es un objeto `FoldSet`
que **no modifica tu `DataFrame` original**.

### Paso 4 — Verifica la integridad (obligatorio antes de usar los folds)

```python
report = folds.verify()
print(report.passed)        # True / False — ¿la partición es válida?
print(report.to_text())     # reporte humano completo
```

`report.passed` es `False` si hay solapamiento entre folds o si falta
cobertura (alguna fila no quedó en ningún fold). Si es `False`, **no
continúes** con el entrenamiento — revisa tu dataset (duplicados raros,
columna de estrato mal elegida, etc.).

### Paso 5 — Itera entrenamiento/prueba

```python
for i, train, test in folds.iter_train_test():
    print(f"Fold {i + 1}: train={len(train)} filas, test={len(test)} filas")
    # entrena tu modelo con `train`, evalúa con `test`
```

En cada iteración, `train` son los `k-1` folds restantes concatenados y `test`
es el fold `i`. Así obtienes `k` corridas de entrenamiento/evaluación, una por
fold.

### Paso 6 — Exporta el resultado (el "entregable")

```python
import os
os.makedirs("outputs", exist_ok=True)

folds.to_csv_dir("outputs/folds")                       # un CSV por fold
folds.to_excel("outputs/folds_partitions.xlsx")          # un Excel, una hoja por fold
report.to_text()                                         # string en memoria
report.to_json("outputs/integrity_report.json")          # para integraciones/CI
report.to_html("outputs/integrity_report.html")          # ábrelo en el navegador
```

Abre `outputs/integrity_report.html` con doble clic (o `start outputs/integrity_report.html`
en Windows / `open ...` en Mac / `xdg-open ...` en Linux) para ver el dashboard.

> Todo este flujo (pasos 1–6) ya está automatizado en
> [`examples/quickstart.py`](examples/quickstart.py) — cópialo y adáptalo a tu
> dataset en vez de escribir el código desde cero.

---

## Parámetros de `StratifiedPartitioner`

| Parámetro | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `k` | `int` | — (requerido) | Número de folds. Mínimo 2. |
| `stratify_by` | `str` | — (requerido) | Nombre de la columna usada como estrato. |
| `seed` | `int` | `42` | Semilla del barajado. Misma semilla + mismos datos = misma partición siempre. |
| `dedup` | `bool` | `True` | Elimina filas 100% duplicadas antes de particionar. Pon `False` solo si las repeticiones son intencionales (p. ej. ya son un peso/frecuencia). |
| `dropna_stratum` | `bool` | `False` | Si `True`, descarta filas cuyo estrato es `NaN`. Si `False` (recomendado), las agrupa en un estrato propio y las **reporta como aviso**, para que no pasen inadvertidas. |
| `triple_columns` | `tuple[str, str, str]` o `None` | `None` (autodetecta si hay 3 columnas) | Indica explícitamente `(sujeto, relación, objeto)` para habilitar el reporte de solapamiento de entidades. |

---

## Cómo leer el reporte de integridad

El objeto `IntegrityReport` (`folds.verify()`) trae estos campos clave —
disponibles igual en texto, JSON y HTML:

| Campo / sección | Significa |
|---|---|
| **`passed` / badge verde "Sin solapamiento"** | Ninguna fila cayó en dos folds a la vez. Si esto falla, hay un bug grave de particionamiento. |
| **`overlap_count`** | Número de filas duplicadas entre folds. Debe ser `0`. |
| **Cobertura total** | Cada fila del dataset limpio aparece en exactamente un fold (ni se perdió ninguna, ni se repitió). |
| **Tabla "Distribución por estrato"** | Para cada valor de estrato (relación o clase): cuántos casos hay en cada fold, su media, su **Std** (desviación estándar) y su **CV** (coeficiente de variación = Std/Media en %). |
| **CV bajo (verde, <5%)** | El estrato está repartido casi exactamente igual en todos los folds — la cantidad de datos por fold para esa clase/relación es confiable. |
| **CV alto (rojo, >15%)** | Ese estrato tiene pocos ejemplos o se reparte de forma desigual; con pocos datos, considera bajar `k` o agregar más ejemplos de esa clase. |
| **Avisos (`warnings`)** | Texto explicando estratos con `NaN`, estratos con menos ejemplos que folds (`< k`), o folds desbalanceados (CV global > 5%). |
| **Sección "Entidades" (solo grafos)** | Cuántas entidades únicas (nodos) hay por fold y cuántas se repiten entre cada par de folds. Esto es **esperado e informativo** en grafos de conocimiento (las entidades sí se repiten; las tripletas no).|

El **dashboard HTML** muestra exactamente esta misma información con tarjetas,
colores y barras — pensado para compartir con alguien que no use Python.

---

## Generar el entregable completo

Produce, en una sola corrida, los folds + Excel + los tres reportes de integridad:

```bash
python scripts/build_deliverable.py            # usa datasets/GoT.csv, k=5
# o vía CLI:
python -m skfold_kge partition datasets/GoT.csv --by Column2 --k 5 --sep ";" \
    --out outputs --triple-names Subject Relation Object
```

Salida en `outputs/`:

```
outputs/
├── folds/Fold_1.csv … Fold_5.csv      # un CSV por fold (Subject,Relation,Object,label)
├── folds_partitions.xlsx              # un Excel, una hoja por fold
├── integrity_report.txt               # reporte en texto
├── integrity_report.json              # reporte serializado
└── integrity_report.html              # dashboard estático (abrir en navegador)
```

---

## Referencia completa de la CLI

```bash
python -m skfold_kge partition <input> --by <columna> [opciones]
```

| Flag | Descripción |
|---|---|
| `input` (posicional) | Ruta o URL del CSV. |
| `--by COLUMNA` | **Requerido.** Columna de estrato. |
| `--k N` | Número de folds (def. `5`). |
| `--seed N` | Semilla (def. `42`). |
| `--sep "S"` | Separador del CSV (def. `";"`). Usa `--sep ","` para CSV estándar. |
| `--out DIR` | Carpeta de salida (def. `outputs`). |
| `--no-dedup` | No eliminar filas duplicadas. |
| `--dropna` | Descartar filas con estrato `NaN` (en vez de reportarlas). |
| `--triple-names S R O` | Renombra las 3 columnas exportadas a estos nombres (modo grafo). |
| `--report-only` | Solo imprime el reporte por consola; no escribe archivos. Útil para validar rápido un dataset nuevo. |

Ejemplos:

```bash
# Dataset propio con coma como separador, sin generar archivos (solo ver el reporte)
python -m skfold_kge partition mi_dataset.csv --by clase --sep "," --report-only

# Grafo con k=10 y nombres de columnas personalizados
python -m skfold_kge partition mi_kg.csv --by relacion --k 10 \
    --triple-names sujeto relacion objeto --out salida/
```

También existe `evaluate` (requiere los extras `[kge]` o `[text]`):

```bash
python -m skfold_kge evaluate datasets/GoT.csv --task kge  --by Column2 --epochs 50
python -m skfold_kge evaluate noticias.csv      --task text --by label --text-col texto
```

---

## Metodología

### Estratificación Round-Robin

Las filas se agrupan por el valor del estrato; cada grupo se baraja de forma
reproducible (semilla) y se reparte entre los `k` folds en orden Round-Robin.
Así cada fold recibe ≈ `1/k` de cada estrato.

### Sustento de la cantidad de datos (Std / CV)

El reporte cuantifica, por estrato, el recuento en cada fold con su **Std** y
**CV**. La regla práctica:

| CV | Interpretación |
|---|---|
| **< 5 %** | Estratificación casi perfecta (verde) |
| 5 – 15 % | Aceptable; revisar estratos pequeños (ámbar) |
| **> 15 %** | Desbalance: aumentar datos o reducir `k` (rojo) |

Con `k=5` la proporción entrenamiento/prueba es **80/20**, estándar en benchmarks
KGE como FB15k-237 y WN18RR.

### ¿Por qué F1 Score?

- **KGE:** `f1_pairwise` mide si el modelo puntúa una tripleta verdadera por
  encima de una negativa muestreada 1:1 (clasificación binaria balanceada).
- **Noticias falsas:** **F1 macro** no se infla con la clase mayoritaria en
  datasets desbalanceados — penaliza por igual errores en *fake* y *real*.

---

## Detección de noticias falsas

```python
from skfold_kge.evaluate import cross_validate_text, load_isot

df = load_isot("isot/Fake.csv", "isot/True.csv")   # 0=fake, 1=real
res = cross_validate_text(df, text_col="text", label_col="label", k=5)
print(res["summary"]["F1"])     # {'mean':…, 'std':…, 'cv':…}
```

Loaders incluidos: `load_isot`, `load_liar`, `load_welfake`. Datasets sugeridos:

| Dataset | Tamaño | Acceso |
|---|---|---|
| ISOT | 23,503 | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| LIAR | 12,836 | HuggingFace `datasets` |
| WELFake | 72,134 | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |

---

## Evaluación de modelos KGE

```python
from skfold_kge import StratifiedPartitioner
from skfold_kge.evaluate import cross_validate_kge

folds = StratifiedPartitioner(k=5, stratify_by="Column2").fit_transform(df)
results = cross_validate_kge(folds, models=["TransE", "ComplEx", "RotatE"],
                             num_epochs=200, embedding_dim=100)

# Añade la sección de métricas (MRR/Hits/F1 ± Std) al dashboard:
folds.verify().to_html("outputs/integrity_report.html", metrics=results)
```

---

## API

| Símbolo | Descripción |
|---|---|
| `StratifiedPartitioner(k, stratify_by, seed, dedup, dropna_stratum, triple_columns)` | Particionador principal. `.fit_transform(df) → FoldSet` |
| `FoldSet` | `.sizes() .fold_frame(i) .train_test(i) .iter_train_test() .verify() .to_excel() .to_csv_dir()` |
| `IntegrityReport` | `.passed .overlap_count .warnings .to_text() .to_json() .to_html()` |
| `partition(df, …)` | Atajo funcional |
| `compute_filtered_ranks`, `compute_metrics_from_ranks`, `f1_pairwise` | Métricas KGE sin dependencias pesadas |
| `evaluate.cross_validate_kge` | Comparación KGE (extra `[kge]`) |
| `evaluate.cross_validate_text` | Clasificación de texto (extra `[text]`) |

---

## Estructura del proyecto

```
skfold_kge/            # paquete
  partition.py         # StratifiedPartitioner + FoldSet
  verify.py            # IntegrityReport
  report.py            # texto / JSON / HTML (dashboard)
  io.py                # carga y exportación
  metrics.py           # MRR, Hits@K, f1_pairwise
  cli.py               # interfaz de línea de comandos
  evaluate/            # extras opcionales (kge, classification)
scripts/build_deliverable.py
examples/quickstart.py
tests/                 # pytest
datasets/GoT.csv
outputs/               # entregable generado
```

---

## Preguntas frecuentes / Troubleshooting

**`KeyError: La columna de estrato '...' no está en el DataFrame`**
El nombre pasado en `stratify_by` (o `--by`) no coincide exactamente con una
columna del CSV. Imprime `df.columns` para ver los nombres reales (cuidado con
espacios o mayúsculas).

**`ValueError: k debe ser >= 2`**
`k=1` no es validación cruzada (no quedaría fold de prueba). Usa `k>=2`.

**El reporte muestra `has_na_stratum: true`**
Hay filas cuya columna de estrato es `NaN`/vacía. Por defecto se agrupan en un
estrato propio (no se pierden), pero conviene revisar el dato fuente. Si
prefieres descartarlas, usa `dropna_stratum=True` (o `--dropna` en la CLI).

**Avisos de "Estratos con menos de k ejemplos"**
Alguna clase/relación tiene menos casos que folds — no puede repartirse en
todos. No es un error, pero indica que ese estrato es muy minoritario; revisa
si es ruido o si necesitas más datos para esa clase.

**`ImportError` al usar `cross_validate_kge` o `cross_validate_text`**
Faltan los extras opcionales. Instala con `pip install -e ".[kge]"` o
`pip install -e ".[text]"` según el caso (ver [Instalación](#instalación)).

**¿Por qué mis tamaños de fold no son exactamente iguales?**
Es normal: si el total de un estrato no es múltiplo exacto de `k`, algunos
folds reciben un elemento más que otros (diferencia máxima de 1 por estrato).
Esto se ve reflejado en el CV de esa fila en la tabla de distribución.

**¿Cómo sé qué separador (`--sep`) usar?**
Abre el CSV con un editor de texto simple y mira el carácter entre columnas en
la primera línea (`;`, `,` o `\t`). `pandas.read_csv` también falla con un
error claro si el separador es incorrecto (verás una sola columna gigante).

**El dashboard HTML se ve sin estilos al abrirlo**
No debería pasar: el CSS está embebido en el mismo archivo (no depende de
internet ni de archivos externos). Si ocurre, asegúrate de abrir el `.html`
completo y no solo un fragmento, y revisa que no haya sido truncado al copiarlo.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Cubre: ausencia de solapamiento, cobertura total, reproducibilidad por semilla,
distribución proporcional, deduplicación, detección de estrato NaN, solapamiento
de entidades y las métricas (MRR/Hits/F1).

---

## Reproducibilidad

El notebook `Validación_Cruzada_Estratificada.ipynb` es el artefacto exploratorio
original. Esta librería reproduce **exactamente** sus tamaños de fold
(`640, 639, 635, 632, 630`) con `seed=42`.

## Licencia

MIT.
