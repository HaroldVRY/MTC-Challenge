# SMART CHALLENGE 2026: IA para la Movilidad del Perú

Este repositorio contiene mi solución al reto:

## Overview

# SMART CHALLENGE 2026: IA para la Movilidad del Perú
El **Ministerio de Transportes y Comunicaciones (MTC)** de la República del Perú convoca al Concurso Nacional **“SMART Challenge 2026: IA para la Movilidad del Perú”**. Esta competencia busca identificar y desarrollar modelos de visión computacional especializados en la detección y clasificación precisa de vehículos en el contexto urbano peruano.

## El Reto
El objetivo principal es desarrollar una solución basada en inteligencia artificial capaz de detectar y clasificar vehículos en intersecciones urbanas a partir de clips de video.

Cada clip corresponde a un video de aproximadamente **5 segundos**, muestreado a **10 FPS**, es decir, alrededor de **50 fotogramas por clip**. Aunque la evaluación se realiza a nivel de fotograma, los participantes pueden utilizar la información temporal del clip completo para mejorar sus predicciones, por ejemplo mediante tracking, postprocesamiento temporal o modelos que aprovechen la secuencia de frames.

El reto se enfoca en la detección vehicular mediante **oriented bounding boxes (OBB)**, es decir, cajas delimitadoras con orientación. Esta representación permite describir mejor la geometría de los vehículos en tomas aéreas o con perspectiva, donde las cajas horizontales tradicionales pueden ser insuficientes.

La motivación final del reto está relacionada con el análisis de vehículos en movimiento en intersecciones urbanas. En esta competencia, la tarea evaluada corresponde a detección y clasificación vehicular por fotograma. Las detecciones generadas por los modelos constituyen un componente base para futuros sistemas de seguimiento vehicular, conteo de flujos, análisis de trayectorias, matrices de giro y estudios de tráfico automatizados.

## Marco Normativo y Propósito
El concurso surge como una iniciativa para facilitar la generación de información técnica útil para la planificación vial y la evaluación del impacto vial. Actualmente, muchos conteos vehiculares y estudios de movimientos en intersecciones se realizan de forma manual, lo que demanda tiempo, recursos y está sujeto a errores humanos.

Con esta competencia, se busca que el talento nacional desarrolle soluciones tecnológicas que puedan servir como base para herramientas prácticas, escalables y de bajo costo, orientadas a apoyar a gobiernos locales en el análisis del tránsito urbano.

## Contenido del repositorio
- `MTC_Challengue.ipynb`: Notebook principal con el flujo de trabajo, preprocesamiento, entrenamiento e inferencia.
- `requirements.txt`: Lista de dependencias necesarias para reproducir el entorno.
- `data/`: (no incluida) Espacio para colocar datasets descargados o preparados.
- `models/`: Carpeta para guardar pesos y modelos entrenados.
- `notebooks/`: Notebooks adicionales (experimentos, análisis EDA).

## Cómo reproducir el entorno
1. Crear y activar el entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# o en cmd: .\.venv\Scripts\activate
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar el notebook principal `MTC_Challengue.ipynb`.

## Requisitos y Dossier Técnico
Al final de la competencia entregaré el Dossier Técnico Digital que incluye código, documentación técnica, manual de operación y análisis de resultados, siguiendo las bases del concurso (GPL v3).

Nota de entorno: `pandas-profiling` se excluyó de la instalación base porque no es compatible con Python 3.13 en este entorno. Si más adelante se necesita un perfilado exploratorio, se evaluará una alternativa compatible.

---

(Contenido adaptado del enunciado del reto y completado para documentar esta solución.)
