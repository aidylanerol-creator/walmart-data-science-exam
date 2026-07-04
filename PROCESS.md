# PROCESS.md — Proceso de trabajo

## Resumen

Trabajé el problema en cuatro etapas: exploración de los datos antes de decidir nada, definición del problema de negocio con base en la evidencia del EDA, modelado con validación temporal estricta, y documentación. Usé AI (Claude, de Anthropic) como copiloto durante todo el proceso; el detalle está al final.

## Etapa 1 — EDA antes que todo (`notebooks/01_eda.ipynb`)

El enunciado es deliberadamente abierto, así que la primera decisión fue no decidir: explorar estructura, calidad y patrones de los datos antes de elegir el problema. Hallazgos que determinaron el resto del trabajo:

- **Panel casi completo** (203,958 de 204,000 combinaciones fecha × tienda × categoría), sin duplicados. Los huecos son 42 filas en 3 tiendas.
- **Nulos manejables y bien caracterizados**: ~6% en columnas de efectivo (repartido uniformemente entre tiendas, consistente con fallas aleatorias de POS), 3% en el target `units_sold`.
- **`replenishment_signal` es una trampa de leakage.** El diccionario dice que sus nulos están "en los últimos días del periodo"; en realidad están en los 2 primeros. Su correlación con la demanda es ~0.8 en lags 0 a 3. Conclusión: es una ventana móvil que incluye el día corriente — inutilizable como feature de pronóstico.
- **Inconsistencia contable**: `cash + card < total_transactions` en 20% de las filas (nunca mayor) — transacciones sin método de pago clasificado. Los montos sí cuadran exacto.
- **Promociones sin uplift aparente** (−0.6% a +0.8%): descarté construir la solución alrededor de promos.
- Estacionalidad clara: semanal, quincenas, Buen Fin, Navidad.

## Etapa 2 — Definición del problema

Elegí **pronóstico de demanda (`units_sold`) por tienda–categoría a 14 días** porque conecta con la decisión operativa más valiosa (reposición), aprovecha el calendario, y responde directamente a la debilidad encontrada: el sistema de reposición existente se calcula con información del mismo día.

Decisiones de alcance: target `units_sold` (no monto: la reposición se decide en unidades), horizonte 14 días (ciclo de quincena), evaluación con WMAPE (estándar en retail, robusto a días de venta baja), MAE y sesgo (distingue riesgo de sobre-stock vs quiebre).

## Etapa 3 — Modelado (`notebooks/02_modeling.ipynb`, `src/`)

- **Features** (`src/features.py`): lags de 14/21/28 días, medias y desviaciones móviles de 7/28 días desplazadas 14 días, calendario completo, atributos de tienda. Regla dura: para predecir el día *t* solo se usa información disponible en *t − 14*.
- **Validación** (`src/validation.py`): 4 folds de origen rodante con ventanas de prueba de 14 días (ene–feb 2024). Nunca split aleatorio.
- **Baselines primero**: naive estacional (lag 14) y media móvil de 28 días.
- **Modelo**: LightGBM con objetivo L1. Sin tuning exhaustivo — preferí invertir el tiempo en el diseño anti-leakage y la validación, que es donde se gana o pierde la credibilidad del resultado.
- **Análisis de errores**: el fold post-navideño es el más difícil (la demanda cae y los lags aún reflejan diciembre) y el modelo sobre-pronostica en ene–feb (+10–17%). Ambos documentados con siguientes pasos concretos en las conclusiones.

Resultado: WMAPE 30.2% vs 58.4% del mejor baseline (−48%).

## Etapa 4 — Documentación y entrega

README con problema, resultados y reproducibilidad; este PROCESS.md; historial de Git por etapas.

## Herramientas utilizadas

- **Python**: pandas, numpy, matplotlib, seaborn, LightGBM, scikit-learn, Jupyter.
- **Git/GitHub** para control de versiones y entrega.
- **AI: Claude (Anthropic), vía la app de escritorio en modo agente.** Uso declarado conforme lo permite y pide el enunciado:
  - Claude propuso la estructura del EDA y del pipeline, escribió borradores del código de los notebooks y módulos de `src/`, y ejecutó verificaciones de reproducibilidad.
  - La detección de leakage en `replenishment_signal` surgió de una hipótesis planteada al leer el diccionario de datos, que validamos juntos con el análisis de correlaciones por rezago.
  - Todas las decisiones (problema a resolver, target, horizonte, métricas, exclusión de la señal, esquema de validación) las tomé yo, validando cada paso en la conversación antes de proceder.
  - Todo el código fue revisado y ejecutado por mí en mi máquina antes de integrarse al repositorio.
