# 📊 Backends de Visualización Avanzada

El simulador cuenta con un motor de visualización modular que permite inspeccionar la red neuronal y el espacio latente desde múltiples perspectivas.

## 1. PCA (Análisis de Componentes Principales)
Este backend realiza una reducción de dimensionalidad en tiempo real sobre los puntos de la "manta".
- **Función**: Calcula los 3 componentes principales de mayor varianza.
- **Uso**: Útil para orientar automáticamente la cámara hacia la vista que mejor explica la separación de los datos en espacios de alta dimensión.

## 2. Coordenadas Paralelas (Parallel Coordinates)
Visualiza cada vértice de la malla 5D (o de la capa activa) como una línea quebrada.
- **Ejes**: 5 ejes verticales representan las 5 neuronas de la capa oculta.
- **Interacción**: La posición actual del jugador se resalta en **Amarillo Brillante**, permitiendo ver cómo su ubicación en el espacio 3D se correlaciona con los valores específicos de cada neurona.

## 3. SPLOM (Scatter Plot Matrix)
Una matriz de mini-gráficos de dispersión que muestra todas las combinaciones posibles de proyecciones entre neuronas.
- **Utilidad**: Identifica correlaciones directas o agrupamientos entre pares específicos de dimensiones (ej: Neurona 1 vs Neurona 4).

## 4. Glifos Estructurales (Chernoff-inspired)
Añade profundidad informativa a cada nodo de la malla 3D principal.
- **Dimensión 4**: Controla el **tamaño** de la esfera en el vértice.
- **Dimensión 5**: Controla la **opacidad/rugosidad** visual.
- **Resultado**: Permite "ver" 5 dimensiones simultáneamente en el visor 3D.

## 5. Diagrama de Schlegel
Una proyección alámbrica de un hipercubo de 5 dimensiones.
- **Estructura**: Visualiza la topología del espacio de búsqueda de la red.
- **Simetría**: Muestra cómo los pesos de la red están organizados jerárquicamente.

---
**Control**: Pulsa la tecla **[X]** para ciclar entre los diferentes modos de visualización.
