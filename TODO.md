
### 📚 Referencia Rápida
- **[Índice de Documentación](doc/README.md)** (Arquitectura, Manifolds, Config)
- **[Manual del Motor (README)](README.md)**


## 🧠 I. Cerebro y Algoritmos (Optimización)
- [ ] **Optimización Avanzada (Adam/RMSprop)**: Implementar optimizadores con tasa de aprendizaje adaptativa para una convergencia más rápida.
- [ ] **Regularización (L2/Dropout)**: Añadir técnicas para evitar el overfitting y suavizar las deformaciones del espacio.
- [ ] **Clasificación Multiclase (Softmax)**: Implementar la función Softmax y pérdida de Entropía Cruzada para mejorar la precisión en niveles con 3 o más colores.
- [ ] **Inicialización Inteligente**: Implementar métodos como Xavier o Kaiming (He) Initialization para asegurar un estiramiento equilibrado del manifold en altas dimensiones.
- [ ] **Cálculo de Pérdida Directa (Geometric Stress)**: Mostrar el valor de la función de coste (Loss) en el HUD para correlacionar el error con la tensión de la manta.
- [ ] **Sincronización Frame-to-Epoch**: Optimizar el bucle de entrenamiento para procesar lotes de épocas por cada frame, mejorando la fluidez de la animación.
- [ ] **Proyección de Atención (Transformers)**: Visualizar cómo la atención "mueve" los puntos en base al contexto, integrando con `Cybertron` o `Hugot`.

## 🎨 II. Interacción y Visualización 3D
- [ ] **Edición Dinámica de Neuronas**: Permitir añadir/quitar neuronas de una capa específica con el ratón/teclado en tiempo real.
- [ ] **Efecto Neón y Constelaciones Vertex**: Añadir brillo (bloom) a la rejilla y puntos estelares en las intersecciones para un look más "premium".
- [ ] **Coloración Dinámica por Tensión**: Cambiar el color de la manta según la curvatura o tensión (ej. cian a magenta en zonas de alto estrés).
- [ ] **Mapas de Calor de Pesos**: Visualizar la intensidad de las conexiones (sinapsis) como colores sobre el modelo 3D.
- [ ] **Modo Comparativa (A/B Test)**: Pantalla dividida para comparar dos motores (ej. CPU vs GPU) o dos arquitecturas simultáneamente.
- [ ] **Ondas de Gravedad Visuales**: Efectos de distorsión en la rejilla cuando los pesos cambian drásticamente durante el aprendizaje.
- [ ] **Animaciones Pedagógicas (Estilo Manim)**: Integrar conceptos de [Manim](https://github.com/3b1b/manim/) para crear explicaciones visuales matemáticas sobre el flujo de tensores.
- [ ] **Evolución del Escaneo CNN**: Refinar la idea del Sliding Window para que sea aún más "real" y conectada con los kernels internos.

## ⚙️ III. Infraestructura Técnica (Motores)
- [ ] **Persistencia Total USM (Level Zero)**: Eliminar copias de memoria (`memcpy`) manteniendo pesos y activaciones en la GPU permanentemente.
- [x] **Motor WebAssembly (WASM/SIMD)**: Portado el motor de cálculo a WASM para ejecución en navegadores con edición en vivo.
- [ ] **Exportación Universal (ONNX)**: Permitir guardar el modelo en formato ONNX para su uso en herramientas externas (Python/PyTorch/C++).
- [ ] **Especialización WASM (SIMD)**: Crear `nn/wasm` para aprovechar las extensiones SIMD de WebAssembly y optimizar el cálculo en navegador.
- [ ] **Optimización de la Pizarra**: Mejorar el layout de las operaciones matemáticas para redes muy profundas (>10 capas).
- [ ] **Refactorización de Logs y Persistencia**:
    - Borrar la pantalla popup (`ebiten.KeyL`) ya que la visualización de operaciones en la pizarra la hace redundante.
    - Generar un archivo de log persistente (`.log` o `.txt`) al final del entrenamiento junto con la matriz de pesos.
    - Implementar un sistema de "Carga de Estado" que use estos logs y pesos para restaurar niveles ya entrenados.
- [ ] **Backend Gonum**: Implementar un motor que utilice [gonum/mat](https://github.com/gonum/gonum) para operaciones de álgebra lineal de alto rendimiento.
- [ ] **Backend Gonum**: Implementar un motor que utilice [gonum/mat](https://github.com/gonum/gonum) para operaciones de álgebra lineal de alto rendimiento.
- [ ] **Backend GoMLX**: Integrar un motor basado en [GoMLX](https://github.com/gomlx/gomlx) (TensorFlow/XLA) para aprovechar aceleración por hardware avanzada y grafos de computación.
- [ ] **Integración Transformers**: Motores de inferencia para modelos Transformer (BERT/GPT) pre-entrenados trabajando sobre el manifold, integrando [Cybertron](https://github.com/nlpodyssey/cybertron) o [Hugot](https://github.com/knights-analytics/hugot).

## 🛡️ V. DevOps e Infraestructura
- [ ] **GitHub Actions (CI/CD)**: Crear worker para compilar automáticamente a WASM y publicar en GitHub Pages en cada push.
- [ ] **Testing Multi-plataforma**: Automatizar pruebas unitarias para todos los motores (Parallel, AVX, GPU, WASM).

---

## ✅ Tareas Completadas
- [x] **Pizarra de Cálculo**: Visualización en tiempo real de Forward y Backward.
- [x] **Sistema de Motores (Multiplexación)**: Implementación de engines Standard, Parallel, AVX, oneAPI y Level Zero.
- [x] **Configuración Externa**: Control de niveles y motores vía `levels.json`.
- [x] **Evolución Dinámica**: Sistema de generadores componentes (Gaussian, Ring, Spiral, Box).
- [x] **Documentación de Dimensionalidad**: Guía técnica sobre el plegado del manifold.
- [x] **Lanzador de Sondas**: Pruebas de inferencia dinámica con estrellas persistentes.
- [x] **Persistencia de Red**: Guardado automático de matrices al alcanzar el objetivo de precisión.
- [x] **Optimización de Persistencia GPU**: Contexto persistente en Level Zero para mayor eficiencia.
