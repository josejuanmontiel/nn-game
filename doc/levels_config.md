# Guía de Configuración de Niveles

El sistema de niveles es ahora 100% dinámico y se configura a través de `levels.json`. Cada nivel define una serie de "Generadores" que inyectan datos en el espacio 2D.

## Estructura de un Generador

```json
{
  "Type": "tipo_de_generador",
  "Class": 0,
  "Count": 50,
  "Settings": { ... }
}
```

### Tipos de Generadores Disponibles

1. **Gaussian**: Crea una nube de puntos con distribución normal.
   - `cx`, `cy`: Centro de la nube.
   - `std`: Desviación estándar (dispersión).

2. **Ring**: Crea un anillo de puntos.
   - `cx`, `cy`: Centro del anillo.
   - `r1`: Radio interno.
   - `r2`: Radio externo.

3. **Spiral**: Genera un brazo de espiral.
   - `cx`, `cy`: Centro de la espiral.
   - `radiusScale`: Factor de crecimiento del radio.
   - `rotations`: Número de vueltas (en múltiplos de PI).
   - `width`: Ancho del brazo de la espiral.
   - `phase`: Desplazamiento angular (útil para múltiples brazos).

4. **Box**: Crea un área rectangular sólida.
   - `x1`, `y1`: Esquina superior izquierda.
   - `x2`, `y2`: Esquina inferior derecha.

## Ejemplo: Nivel de 3 Brazos
Para crear una espiral triple, define tres generadores de tipo `spiral` con fases separadas por `2.09` (aproximadamente `2*PI/3`).
