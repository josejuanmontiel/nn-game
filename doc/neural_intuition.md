# Intuición de la Red: Del Plano al Hiperespacio

Este documento explica cómo el simulador traduce las matemáticas de las matrices en la deformación visual de la "manta" que ves en pantalla.

## 1. El Salto al Hiperespacio (2D → 5D)

Cuando un punto de los generadores entra en la red, ocurre una expansión dimensional.

*   **Punto de Origen (2D)**: Un "insecto" moviéndose en un plano $(x, y)$.
*   **Capa Oculta (5D)**: Cada una de las 5 neuronas actúa como un "observador" independiente. Cada una calcula su propia versión de la realidad multiplicando las coordenadas por sus pesos ($W$) y sumando un sesgo ($B$).
*   **Intuición**: Tu punto no ha cambiado de identidad, pero ahora tiene 5 "características" o coordenadas que lo describen. Ha pasado de un plano a un hiper-volumen.

## 2. La Ventana al Mundo (Visualización 5D → 3D)

Como humanos, no podemos ver en 5 dimensiones. El simulador aplica una **proyección directa**:

1.  Coge el vector resultante de 5 elementos.
2.  Ignora los últimos 2 elementos.
3.  Usa los primeros 3 como coordenadas $(x, y, z)$ en el espacio 3D.

**¿Por qué vemos curvas?** Porque la red está "doblando" la superficie 2D dentro del espacio 5D. Al mirar solo 3 de esas dimensiones, vemos las sombras y pliegues de ese movimiento.

## 3. La Física de la Manta

Los controles del HUD que ves a la izquierda afectan físicamente a la manta:

*   **Bias ($B$) - Traslación Rígida**: Al sumar un valor constante a una neurona, toda la manta se desplaza en esa dirección. Es como empujar la sábana por encima de la mesa.
*   **Weights ($W$) - Estiramiento y Plegado**: Cambian la "importancia" de cada eje. Esto estira, encoge o retuerce la manta. Es la herramienta que usa la red para separar puntos que están muy mezclados.

## 4. El Poder de la Profundidad (Múltiples Capas)

¿Qué pasa cuando añades más capas? Cada capa es como una persona distinta haciendo un doblez de origami y pasándoselo a la siguiente.

*   **Capa 1**: Hace el primer doblez (proyecta el plano a 5D).
*   **Capas Posteriores**: Cogen ese papel ya doblado y lo vuelven a plegar. 
*   **Resultado**: Cuantas más capas, más "nudos" y formas complejas puede crear la red para atrapar los puntos objetivo.

---
> [!TIP]
> **Diferencia con UMAP**: Mientras que algoritmos como UMAP buscan una representación "fiel" de datos estáticos, este simulador muestra la **distorsión del espacio** que la red *elige* crear para resolver el problema de clasificación.
