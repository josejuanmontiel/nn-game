# WASM & Web Integration (Ebitengine)

Este documento detalla cómo el simulador neuronal se exporta a la web utilizando WebAssembly (WASM) y el motor Ebitengine, permitiendo una experiencia interactiva sin instalación.

## Arquitectura de Exportación

El simulador utiliza un sistema de compilación cruzada de Go para generar un binario ejecutable por el navegador.

### Componentes Clave

1.  **game.wasm**: El binario principal que contiene toda la lógica del juego, la red neuronal y el motor de dibujo.
2.  **wasm_exec.js**: El puente de runtime proporcionado por Go para permitir que el código WASM interactúe con el DOM y las APIs del navegador.
3.  **index.html**: El host que inicializa el entorno WASM y proporciona la interfaz de usuario (editor JSON).

## El Puente Go/JavaScript (JS Bridge)

Para permitir que la página web interactúe con el simulador en tiempo real, hemos implementado un puente utilizando el paquete `syscall/js`.

### js_bridge.go
Este archivo expone funciones de Go al entorno global de JavaScript:
- `updateLevelConfig(jsonStr)`: Permite inyectar una nueva configuración de niveles al vuelo.

`js_bridge_stub.go` asegura que el código siga compilando para escritorio (Linux/Windows/macOS) ignorando estas llamadas web.

## Compilación y Ejecución

Hemos automatizado el proceso a través del `Makefile` incluido en la raíz.

### 1. Compilar el binario WASM
```bash
make build-wasm
```
*Esto genera `web/game.wasm` estableciendo `GOOS=js` y `GOARCH=wasm`.*

### 2. Levantar el Servidor
Los navegadores no permiten ejecutar WASM desde `file://` por razones de seguridad. Es necesario un servidor web.
```bash
make serve
```
*Por defecto, esto levanta un servidor en `http://localhost:8080`.*

## Editor de Niveles "Live"

La integración web incluye un editor lateral que permite modificar `levels.json`. Al pulsar **"Apply Config"**:
1. JavaScript valida el JSON.
2. Se llama a la función de Go vinculada en el puente.
3. El simulador reinicia el nivel actual con los nuevos parámetros (nodos, probabilidad, tipos de generadores) sin necesidad de recargar la página.

## Optimización Técnica (SIMD)

A partir de la versión 1.21 de Go y mejorado en la 1.25, el compilador es capaz de generar instrucciones SIMD (`v128`) para WebAssembly de forma automática si el código está estructurado para ello.

### Estrategias de Optimización en `nn/wasm`

1.  **Loop Unrolling (4x)**:
    En los núcleos de cálculo (`Forward`, `Backward`, `Update`), hemos desenrollado los bucles críticos. Esto permite que el procesador trate múltiples datos en un solo ciclo de instrucción.
    ```go
    for ; j <= n-4; j += 4 {
        sum += inputs[j] * weights[j]
        sum += inputs[j+1] * weights[j+1]
        // ...
    }
    ```

2.  **BCE (Bounds Check Elimination)**:
    Hemos reorganizado el acceso a los *slices* para asegurar que el compilador de Go pueda omitir las comprobaciones de límites de array en cada iteración. Al garantizarle al compilador que los índices son válidos una sola vez al inicio del bucle, el código generado es mucho más limpio y rápido.

## Rendimiento

- **SIMD Nativo**: Gracias a las optimizaciones anteriores, el motor web aprovecha la aceleración vectorial de los navegadores modernos (Chrome, Firefox, Safari).
- **Estabilidad**: Se observa una mayor estabilidad en los FPS y un menor consumo de CPU durante los pasos de entrenamiento intensivo comparado con la implementación secuencial estándar.

## Limitaciones Técnicas

- **CGO**: WASM en Go no soporta CGO nativamente de forma sencilla. Por ello, las aceleraciones de hardware específicas (como AVX o GPU local) se desactivan automáticamente mediante *build tags* (`//go:build !js`) a favor de motores de cálculo puramente en Go o WebGL a través de Ebitengine.
- **Rendimiento**: Aunque WASM es rápido, para redes neuronales masivas se recomienda el uso de backends optimizados que estamos planificando (como WebGL o WebNN).
