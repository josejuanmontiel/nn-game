# ⚙️ Input Data Engine (Generador de Datos N-D)

El **Input Data Engine** es el responsable de generar conjuntos de datos sintéticos complejos en espacios de múltiples dimensiones para desafiar la capacidad de aprendizaje de la red.

## 1. Generadores Geométricos
El motor soporta la creación de formas en $D \ge 2$ dimensiones:

- **Hyper-Sphere / Ring**: Genera puntos distribuidos uniformemente sobre la superficie de una hiperesfera. Ideal para probar la capacidad de la red para "encerrar" clases.
- **Swiss Roll (Espiral)**: Una estructura 2D enrollada en un espacio de mayor dimensión. Es la prueba definitiva de "manifold unfolding".
- **Clusters Separables**: Nubes de puntos gaussianas separadas por una distancia paramétrica.
- **XOR Extendido**: Distribuye las clases en cuadrantes opuestos de las primeras dos dimensiones, requiriendo plegados no lineales para su resolución.

## 2. Proyección de "Nivel Cero"
Dado que el simulador es visualmente 3D, el motor debe proyectar los datos de entrada antes de que la red los procese:
- **Si $D \le 3$**: Mapeo directo a los ejes cartesianos.
- **Si $D > 3$**: Se aplica **PCA** para generar un "Mapa de Referencia" estético que preserve la mayor varianza posible de los datos originales.

## 3. Integración con el Pipeline
Cada punto generado mantiene su identidad a través del simulador:
- **ID Único**: Permite rastrear puntos individuales durante la deformación.
- **Etiqueta de Clase**: Define el color y el objetivo de clasificación.
- **Dualidad de Datos**: El motor entrega el tensor original de N dimensiones para el entrenamiento de la red y las coordenadas $(x, y, z)$ proyectadas para el visualizador.

---
**Configuración**: La dimensionalidad se define en `levels.json` mediante el campo `"InputDims"`.
