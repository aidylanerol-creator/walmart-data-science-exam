# Pronóstico de demanda a 14 días — Retail México

Solución a la prueba técnica de Data Scientist (enunciado original en [`INSTRUCCIONES.md`](INSTRUCCIONES.md)).

## Problema de negocio

**Pronosticar las unidades vendidas (`units_sold`) por tienda y categoría con 14 días de anticipación**, para alimentar la reposición de inventario con cobertura de un ciclo de quincena completo.

Por qué este problema:

- La reposición es la decisión operativa diaria más sensible al error de pronóstico en retail: sobre-stock inmoviliza capital y genera merma; sub-stock pierde venta.
- El dataset lo pide a gritos: calendario con quincenas y eventos mexicanos, 80 tiendas × 6 categorías × 425 días, y una `replenishment_signal` existente que resulta estar mal construida (ver hallazgos).
- El EDA reveló que la señal de reposición actual **usa la demanda del mismo día** (leakage): un pronóstico honesto a 14 días es exactamente lo que ese sistema necesita y no tiene.

## Resultados

Validación con 4 folds de origen rodante (ventanas de prueba de 14 días, ene–feb 2024):

| Método | WMAPE promedio |
|---|---|
| Naive estacional (14 días) | 58.4% |
| Media móvil 28 días | 61.1% |
| **LightGBM** | **30.2%** |

El modelo **reduce el error de pronóstico un 48%** frente al mejor baseline, con mejora consistente en los 4 folds y desempeño homogéneo por categoría (WMAPE 28–32%). Valuado a precio mediano por categoría, equivale a ~18 M MXN/día de desalineamiento inventario–demanda evitado en las 80 tiendas (ver interpretación y caveats en el notebook de modelado, sección 8).

## Estructura del repositorio

```
├── INSTRUCCIONES.md        # Enunciado original de la prueba
├── PROCESS.md              # Proceso de trabajo, decisiones y uso de AI
├── data/                   # Datos fuente (transactions, stores, calendar)
├── data_dictionary.md      # Diccionario de datos original
├── notebooks/
│   ├── 01_eda.ipynb        # EDA: calidad de datos, patrones, auditoría de leakage
│   └── 02_modeling.ipynb   # Baselines, LightGBM, validación, valor de negocio
├── src/
│   ├── features.py         # Pipeline de features sin leakage
│   ├── validation.py       # Validación temporal con origen rodante
│   └── metrics.py          # WMAPE, MAE, sesgo
└── requirements.txt
```

## Reproducibilidad

```bash
# 1. Crear entorno e instalar dependencias (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Ejecutar los notebooks en orden
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

Los notebooks se ejecutan de inicio a fin sin intervención (semilla fija, rutas relativas). `02_modeling.ipynb` tarda unos minutos por el entrenamiento de 4 modelos LightGBM.

## Decisiones clave

1. **`replenishment_signal` excluida del modelado.** Correlación ~0.8 con la demanda de los mismos días y nulos solo en los 2 primeros días del periodo: es una ventana móvil que incluye el día corriente. Usarla sería data leakage (detalle en `01_eda.ipynb`, sección 10).
2. **Cero información del futuro.** Todos los lags y ventanas móviles están desplazados ≥ 14 días (el horizonte); la validación es exclusivamente temporal — nunca split aleatorio.
3. **Nulos manejados de forma explícita.** El target se interpola solo para calcular lags; las filas con target originalmente nulo se excluyen del entrenamiento y la evaluación.
4. **Baselines primero.** Sin un naive estacional como referencia no se puede demostrar que un modelo aporta valor.
