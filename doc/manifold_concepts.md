# Conceptos de Dimensionalidad y Manifold Folding

Este documento explora la relación entre los datos de entrada, el generador de niveles y la representación visual en la "manta" (manifold) del simulador.

## 1. Dimensionalidad de los Datos

Aunque la visualización ocurre en un espacio que parece 3D, el flujo de datos sigue una progresión dimensional específica:

1.  **Espacio de Entrada (2D):** Cada punto (estrella) se genera originalmente como un par de coordenadas `(x, y)` en un plano normalizado de `[-1, 1]`. Este es nuestro "suelo" o espacio base.
2.  **Generadores de Niveles:** Los generadores (Gaussian, Spiral, etc.) definen la *distribución de probabilidad* en este plano 2D. Clasificar consiste en encontrar fronteras que separen estas distribuciones.
3.  **Espacio Latente (Representación de la Red):** A medida que los datos pasan por las capas ocultas, la red neuronal aplica transformaciones afines (rotaciones, traslaciones, escalados) seguidas de no-linealidades (Tanh, ReLU). Esto es lo que "dobla" el espacio.
4.  **Espacio de Clasificación (N-Dimensiones):** La última capa proyecta los datos a un espacio con tantas dimensiones como clases existan (`NumClasses`).

## 2. La "Manta" y el Manifold 3D

La visualización 3D que ves es una **proyección de las primeras 3 neuronas** de la red (o de la capa seleccionada).

*   **¿Qué es la "Manta"?** Es un "mapeo" (mapping) continuo del plano 2D original hacia el espacio 3D. Cada vértice de la rejilla representa una posición `(x, y)` que ha sido transformada por la red.
*   **Dimensionalidad vs. Cardinalidad:** La cardinalidad de los objetos a clasificar (el número de clases) dicta cuántos "puntos de destino" o "atractores" existen en el espacio 3D.
    *   Si hay 3 clases, el objetivo de la red es "doblar" la manta para que todos los puntos de la Clase A acaben cerca del vértice A en 3D, los de la B cerca del B, etc.

## 3. Dinámica del Movimiento

Conforme la red entrena (`Evolving...`):

1.  **Rotación:** Los pesos de la red actúan como matrices de rotación en hyper-espacios. Visualmente, esto se traduce en que la manta gira para alinear las características de entrada con los ejes de salida.
2.  **Traslación:** Los `Biases` desplazan la manta por el espacio, permitiendo que la red "centre" o "centre de masa" las nubes de puntos.
3.  **Folding (Plegado):** Es el corazón del Manifold Learning. La red intenta "desenrollar" distribuciones complejas (como espirales) para que sean linealmente separables en el espacio final.

> [!IMPORTANT]
> Lo que ves no es solo un dibujo, es el **comportamiento topológico** de la red. Si la manta se cruza consigo misma, significa que la red está encontrando una frontera de decisión compleja.

## 4. El Objetivo del Juego

El objetivo es configurar generadores de niveles que desafíen la capacidad de la red para "doblar" el espacio. Cuanto más entrelazadas estén las clases en 2D (ej: Triple Spiral), más capas y "pliegues" necesitará la red para separarlos en la manta 3D.
