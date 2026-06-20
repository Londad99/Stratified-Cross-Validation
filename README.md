# skfold-kge

Librería de validación cruzada k-fold estratificada para grafos de conocimiento
(KGE) y para clasificación de texto, con verificación de integridad de los
folds y un reporte HTML autocontenido.

[![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-pytest-green)](#testing)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#licencia)

---

## Descripción

La librería particiona un dataset en `k` folds estratificados por una columna
categórica, de modo que cada fold conserva la distribución del conjunto
original respecto a esa columna.

| Caso de uso | `stratify_by` | Resultado |
|---|---|---|
| Grafos de conocimiento (KGE) | columna de relación | cada fold mantiene la proporción de cada tipo de relación |
| Clasificación de texto (p. ej. noticias falsas) | columna de clase (`label`) | cada fold mantiene la proporción de clases |

Funciones principales:

- Verificación de que no haya solapamiento entre folds y de que la cobertura sea total.
- Cálculo de desviación estándar (Std) y coeficiente de variación (CV) por
  estrato, como sustento cuantitativo de la cantidad de datos asignada a cada
  fold.
- Reportes de integridad en texto, JSON y HTML.
- Evaluación opcional de modelos KGE (TransE, ComplEx, RotatE) y de
  clasificadores de texto, con F1 Score, MRR y Hits@K.

---

## Índice

1. [Instalación](#instalación)
2. [Uso básico](#uso-básico)
3. [Parámetros de StratifiedPartitioner](#parámetros-de-stratifiedpartitioner)
4. [Estructura del reporte de integridad](#estructura-del-reporte-de-integridad)
5. [Generación del entregable](#generación-del-entregable)
6. [Referencia de la CLI](#referencia-de-la-cli)
7. [Metodología](#metodología)
8. [Clasificación de texto / noticias falsas](#clasificación-de-texto--noticias-falsas)
9. [Evaluación de modelos KGE](#evaluación-de-modelos-kge)
10. [API](#api)
11. [Resolución de problemas](#resolución-de-problemas)
12. [Testing](#testing)
13. [Reproducibilidad](#reproducibilidad)

---

## Instalación

```bash
# Núcleo: particionar, verificar, reportar, exportar. Dependencias: pandas, numpy, openpyxl.
pip install -e .

# Extras opcionales
pip install -e ".[kge]"    # evaluación KGE: pykeen + torch (~2 GB)
pip install -e ".[text]"   # clasificación de texto: scikit-learn
pip install -e ".[liar]"   # loader del dataset LIAR (HuggingFace datasets)
pip install -e ".[all]"    # todos los extras
pip install -e ".[dev]"    # pytest
```

Nombre de distribución: `skfold-kge`. Nombre de import: `skfold_kge`.

Verificación de la instalación:

```bash
python -c "import skfold_kge; print(skfold_kge.__version__)"
skfold-kge --version
```

---

## Uso básico

El dataset de entrada debe ser un `DataFrame` de pandas con una columna que
sirva de estrato: el valor que debe mantener su proporción en todos los
folds. En un grafo de conocimiento, normalmente es la columna de relación
de la tripleta `(sujeto, relación, objeto)`. En un dataset de clasificación,
es la columna de clase.

```python
import pandas as pd
from skfold_kge import StratifiedPartitioner

df = pd.read_csv("mi_dataset.csv", sep=";")

part = StratifiedPartitioner(k=5, stratify_by="Column2", seed=42)
folds = part.fit_transform(df)

print(folds.sizes())   # ej.: [640, 639, 635, 632, 630]
```

`fit_transform` ejecuta, en orden: eliminación de duplicados, agrupación de
filas por valor del estrato, barajado de cada grupo con la semilla dada, y
reparto Round-Robin entre los `k` folds. El `DataFrame` original no se
modifica.

Antes de usar los folds para entrenar, se debe verificar la integridad de la
partición:

```python
report = folds.verify()
assert report.passed   # False si hay solapamiento o cobertura incompleta
print(report.to_text())
```

Iteración sobre los pares entrenamiento/prueba:

```python
for i, train, test in folds.iter_train_test():
    print(f"Fold {i + 1}: train={len(train)} filas, test={len(test)} filas")
```

En cada iteración, `train` corresponde a los `k-1` folds restantes
concatenados y `test` al fold `i`. Esto produce `k` corridas de
entrenamiento/evaluación, una por fold.

Exportación de resultados:

```python
import os
os.makedirs("outputs", exist_ok=True)

folds.to_csv_dir("outputs/folds")               # un CSV por fold
folds.to_excel("outputs/folds_partitions.xlsx") # un Excel, una hoja por fold
report.to_json("outputs/integrity_report.json")
report.to_html("outputs/integrity_report.html") # reporte HTML
```

El flujo completo está implementado en [`examples/quickstart.py`](examples/quickstart.py).

---

## Parámetros de StratifiedPartitioner

| Parámetro | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `k` | `int` | requerido | Número de folds. Mínimo 2. Con `k=5`, la proporción entrenamiento/prueba es 80/20. |
| `stratify_by` | `str` | requerido | Nombre de la columna usada como estrato. |
| `seed` | `int` | `42` | Semilla del barajado. La misma semilla sobre los mismos datos produce la misma partición. |
| `dedup` | `bool` | `True` | Elimina filas duplicadas antes de particionar. Desactivar solo si la repetición de filas es intencional (por ejemplo, representa un peso o frecuencia). |
| `dropna_stratum` | `bool` | `False` | Si es `True`, descarta filas cuyo estrato es `NaN`. Si es `False`, las agrupa en un estrato propio y las incluye como aviso en el reporte. |
| `triple_columns` | `tuple[str, str, str]` o `None` | `None` (se infiere si hay 3 columnas) | Define explícitamente `(sujeto, relación, objeto)` para habilitar el reporte de solapamiento de entidades. |

---

## Estructura del reporte de integridad

El objeto `IntegrityReport`, devuelto por `folds.verify()`, expone los mismos
datos en texto, JSON y HTML:

| Campo o sección | Contenido |
|---|---|
| `passed` | `True` si no hay solapamiento entre folds y la cobertura es total. |
| `overlap_count` | Número de filas que aparecen en más de un fold. Debe ser 0. |
| Cobertura total | Cada fila del dataset limpio aparece en exactamente un fold. |
| Distribución por estrato | Recuento por fold de cada valor de estrato, con media, desviación estándar (Std) y coeficiente de variación (CV). |
| CV por estrato | Menor a 5%: distribución equilibrada entre folds. Entre 5% y 15%: aceptable, revisar estratos pequeños. Mayor a 15%: desbalance; considerar reducir `k` o aumentar los datos de ese estrato. |
| `warnings` | Lista de avisos: estratos con valores `NaN`, estratos con menos ejemplos que folds, o desbalance global del tamaño de los folds. |
| Solapamiento de entidades (modo grafo) | Entidades únicas por fold y entidades compartidas entre pares de folds. Es un valor informativo, no un error: las entidades pueden repetirse entre folds aunque las tripletas no se repitan. |

El reporte HTML presenta la misma información con tarjetas, una tabla y
barras de tamaño de fold, sin dependencias externas ni conexión a internet.

---

## Generación del entregable

```bash
python scripts/build_deliverable.py
# equivalente vía CLI:
python -m skfold_kge partition datasets/GoT.csv --by Column2 --k 5 --sep ";" \
    --out outputs --triple-names Subject Relation Object
```

Salida:

```
outputs/
├── folds/Fold_1.csv ... Fold_5.csv    # un CSV por fold (Subject,Relation,Object,label)
├── folds_partitions.xlsx              # un Excel, una hoja por fold
├── integrity_report.txt
├── integrity_report.json
└── integrity_report.html
```

---

## Referencia de la CLI

```bash
python -m skfold_kge partition <input> --by <columna> [opciones]
```

| Flag | Descripción |
|---|---|
| `input` (posicional) | Ruta o URL del CSV. |
| `--by COLUMNA` | Requerido. Columna de estrato. |
| `--k N` | Número de folds. Por defecto 5. |
| `--seed N` | Semilla. Por defecto 42. |
| `--sep "S"` | Separador del CSV. Por defecto `;`. |
| `--out DIR` | Carpeta de salida. Por defecto `outputs`. |
| `--no-dedup` | No elimina filas duplicadas. |
| `--dropna` | Descarta filas con estrato `NaN` en lugar de reportarlas. |
| `--triple-names S R O` | Renombra las tres columnas en la exportación (modo grafo). |
| `--report-only` | Imprime el reporte sin escribir archivos. |

```bash
python -m skfold_kge partition mi_dataset.csv --by clase --sep "," --report-only

python -m skfold_kge partition mi_kg.csv --by relacion --k 10 \
    --triple-names sujeto relacion objeto --out salida/
```

El subcomando `evaluate` requiere los extras `[kge]` o `[text]`:

```bash
python -m skfold_kge evaluate datasets/GoT.csv --task kge  --by Column2 --epochs 50
python -m skfold_kge evaluate noticias.csv      --task text --by label --text-col texto
```

---

## Metodología

### Estratificación Round-Robin

Las filas se agrupan por el valor del estrato. Cada grupo se baraja de forma
reproducible (semilla fija) y se reparte entre los `k` folds en orden
Round-Robin. Cada fold recibe aproximadamente `1/k` de cada estrato.

### Sustento de la cantidad de datos

El reporte cuantifica, por estrato, el recuento en cada fold junto con su
desviación estándar y su coeficiente de variación:

| CV | Interpretación |
|---|---|
| Menor a 5% | Distribución casi idéntica entre folds. |
| Entre 5% y 15% | Aceptable; revisar los estratos con menos ejemplos. |
| Mayor a 15% | Desbalance; reducir `k` o ampliar los datos de ese estrato. |

Con `k=5`, la proporción entrenamiento/prueba es 80/20, consistente con
benchmarks de referencia en KGE como FB15k-237 y WN18RR.

### Uso de F1 Score

En KGE, `f1_pairwise` evalúa si el modelo puntúa una tripleta verdadera por
encima de una negativa muestreada en proporción 1:1, lo que produce una
clasificación binaria balanceada. En clasificación de texto, se usa F1 macro,
que no se infla con la clase mayoritaria en datasets desbalanceados y
penaliza por igual los errores en cada clase.

---

## Clasificación de texto / noticias falsas

```python
from skfold_kge.evaluate import cross_validate_text, load_isot

df = load_isot("isot/Fake.csv", "isot/True.csv")   # 0 = fake, 1 = real
res = cross_validate_text(df, text_col="text", label_col="label", k=5)
print(res["summary"]["F1"])
```

Loaders incluidos: `load_isot`, `load_liar`, `load_welfake`.

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
results = cross_validate_kge(
    folds, models=["TransE", "ComplEx", "RotatE"],
    num_epochs=200, embedding_dim=100,
)

# Incorpora la sección de métricas (MRR, Hits, F1, con desviación estándar) al reporte HTML.
folds.verify().to_html("outputs/integrity_report.html", metrics=results)
```

---

## API

| Símbolo | Descripción |
|---|---|
| `StratifiedPartitioner(k, stratify_by, seed, dedup, dropna_stratum, triple_columns)` | Particionador principal. `.fit_transform(df)` devuelve un `FoldSet`. |
| `FoldSet` | `.sizes()`, `.fold_frame(i)`, `.train_test(i)`, `.iter_train_test()`, `.verify()`, `.to_excel()`, `.to_csv_dir()`. |
| `IntegrityReport` | `.passed`, `.overlap_count`, `.warnings`, `.to_text()`, `.to_json()`, `.to_html()`. |
| `partition(df, ...)` | Atajo funcional equivalente a instanciar `StratifiedPartitioner` y llamar `fit_transform`. |
| `compute_filtered_ranks`, `compute_metrics_from_ranks`, `f1_pairwise` | Métricas de evaluación KGE, sin dependencias pesadas. |
| `evaluate.cross_validate_kge` | Comparación de modelos KGE. Requiere el extra `[kge]`. |
| `evaluate.cross_validate_text` | Clasificación de texto. Requiere el extra `[text]`. |

---

## Estructura del proyecto

```
skfold_kge/            paquete principal
  partition.py         StratifiedPartitioner, FoldSet
  verify.py            IntegrityReport
  report.py            renderizado a texto, JSON y HTML
  io.py                carga y exportación de datos
  metrics.py           MRR, Hits@K, f1_pairwise
  cli.py               interfaz de línea de comandos
  evaluate/            extras opcionales: kge.py, classification.py
scripts/build_deliverable.py
examples/quickstart.py
tests/
datasets/GoT.csv
outputs/               entregable generado
```

---

## Resolución de problemas

`KeyError` indicando que la columna de estrato no está en el DataFrame: el
valor pasado en `stratify_by` (o `--by`) no coincide con el nombre exacto de
una columna del CSV. Verificar `df.columns`, incluyendo mayúsculas y espacios.

`ValueError: k debe ser >= 2`: con `k=1` no queda fold de prueba, por lo que
no constituye validación cruzada.

`has_na_stratum` en `True`: existen filas con valor de estrato `NaN`. Por
defecto se agrupan en un estrato propio y se reportan como aviso. Para
descartarlas, usar `dropna_stratum=True` o el flag `--dropna`.

Aviso de "estratos con menos de k ejemplos": alguna clase o relación tiene
menos casos que folds y no puede repartirse en todos ellos. No es un error;
indica un estrato minoritario que conviene revisar.

`ImportError` al usar `cross_validate_kge` o `cross_validate_text`: faltan
los extras opcionales correspondientes. Instalar con `pip install -e ".[kge]"`
o `pip install -e ".[text]"`.

Tamaños de fold ligeramente distintos entre sí: es el comportamiento
esperado cuando el total de un estrato no es múltiplo exacto de `k`; la
diferencia máxima por estrato es de un elemento. Se refleja en el CV de la
tabla de distribución.

Separador del CSV: si no se conoce, inspeccionar la primera línea del
archivo con un editor de texto. `pandas.read_csv` falla con un error
identificable si el separador es incorrecto (se interpreta una sola columna).

Reporte HTML sin estilos: el CSS está embebido en el mismo archivo y no
depende de recursos externos. Si esto ocurre, verificar que el archivo se
abrió completo y no fue truncado al copiarlo.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

La suite cubre: ausencia de solapamiento, cobertura total, reproducibilidad
por semilla, distribución proporcional, deduplicación, detección de estrato
`NaN`, solapamiento de entidades y las métricas de evaluación (MRR, Hits, F1).

---

## Reproducibilidad

El notebook `Validación_Cruzada_Estratificada.ipynb` es el artefacto
exploratorio original del proyecto. Esta librería reproduce sus tamaños de
fold (`640, 639, 635, 632, 630`) con `seed=42` sobre el dataset GoT.

## Licencia

MIT.
