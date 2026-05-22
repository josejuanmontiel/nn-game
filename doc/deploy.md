# Guía de Despliegue en GitHub Pages

Este proyecto está preparado para desplegarse automáticamente en GitHub Pages utilizando GitHub Actions.

## Configuración Automática (CI/CD)

He configurado un workflow en `.github/workflows/deploy.yml` que realiza los siguientes pasos en cada `push` a la rama `main`:
1. Instala Go 1.22.
2. Compila el simulador a WebAssembly (`game.wasm`).
3. Prepara la carpeta `web/` con todos los recursos necesarios.
4. Publica el contenido en la rama de GitHub Pages.

## Configuración en el Repositorio

Para que el despliegue funcione correctamente en tu repo de GitHub:
1. Ve a **Settings** > **Pages**.
2. En **Build and deployment**, asegúrate de que **Source** esté en `GitHub Actions`.
3. Una vez hagas push, el worker se encargará del resto.

## Motor WASM Especializado

Ahora existe una carpeta `nn/wasm/` dedicada a optimizaciones específicas para la web:
- **`nn/wasm/wasm.go`**: Contiene el motor que se compila exclusivamente para `js/wasm`. Aquí se pueden añadir optimizaciones SIMD en el futuro.
- **`nn/wasm/wasm_stub.go`**: Permite que el proyecto siga compilando correctamente en Linux/Windows/macOS realizando un fallback al motor estándar.

---
> [!IMPORTANT]
> Recuerda que para que el simulador funcione en la web, el archivo `wasm_exec.js` debe estar presente en la carpeta `web/`. El workflow de GitHub lo copia automáticamente por ti.
