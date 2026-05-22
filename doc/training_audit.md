# 🔍 Auditoría de Entrenamiento y Dinámica de Red

Este documento detalla la auditoría técnica realizada sobre el simulador para asegurar que la visualización del manifold y el proceso de aprendizaje sean coherentes y matemáticamente robustos.

## 1. Desacoplamiento de Tensores (The Detach Check)
En implementaciones de PyTorch, es crucial usar `.detach()` para evitar que la lógica de visualización interfiera con el grafo de computación. 
- **Estado en Go**: Gracias a la semántica de copia de Go y el uso de búferes independientes en los motores (`Engine`), el desacoplamiento es **implícito**. 
- **Conclusión**: El dibujo del manifold no contamina los gradientes ni el estado de entrenamiento.

## 2. Dinámica de la Activación (Post-Nonlinearity)
Para visualizar el "plegado" real del espacio, es vital capturar las activaciones después de la función de transferencia ($f$).
- **Validación**: El simulador utiliza las salidas (`Outputs`) ya transformadas por `Tanh` o `Sigmoid` para proyectar la rejilla 3D.
- **Rango**: La normalización del visor está alineada con el rango de la función activa (ej: $[-1, 1]$ para Tanh), garantizando que la visibilidad de los pliegues sea óptima.

## 3. Inicialización Geométrica (W)
La forma inicial de la "manta" depende directamente de la varianza de los pesos iniciales.
- **Observación**: Una inicialización puramente aleatoria puede colapsar la manta prematuramente.
- **Sugerencia**: Se recomienda el uso de **Kaiming (He) Initialization** para mejorar el estiramiento inicial, especialmente en arquitecturas profundas o con muchas dimensiones de entrada (como el Nivel 5 de 10D).

## 4. Sincronización Frame-to-Epoch
El entrenamiento es órdenes de magnitud más rápido que el renderizado gráfico (60 FPS).
- **Mecánica Actual**: El usuario controla el paso de tiempo (`Sim Speed`).
- **Mejora Propuesta**: Implementar un sub-bucle que procese $N$ épocas por cada frame de renderizado para asegurar una animación fluida sin cuellos de botella de CPU/GPU.

## 5. Métrica de Estrés Geométrico
La pérdida (Loss) no es solo un número; en este simulador representa la **tensión estructural** de la manta.
- **Métrica**: El error de clasificación (MSE o Cross-Entropy) se visualiza como la incapacidad de la manta para alcanzar los "atractores" (clases) en el espacio 3D.
- **Futuro**: Se planea añadir un indicador visual de "Estrés" basado en la magnitud del gradiente acumulado en cada vértice de la rejilla.

---
> [!NOTE]
> Esta auditoría confirma que la arquitectura actual es capaz de representar manifolds complejos de forma fiel a la teoría propugnada en el curso de Deep Learning de NYU.
