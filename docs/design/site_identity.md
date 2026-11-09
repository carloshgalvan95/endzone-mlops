# Site identity

The page in `site/` is English. Two strings differ from the Spanish draft below. The honest banner reads: "Open experiment. Model log loss 0.6377 vs market 0.6120 (the market wins). Not betting picks." The disagreement badge reads "DISAGREES". The example flag reads "EXAMPLE".

The model always picks one team. Home wins when `model_home_wp` is at least 0.5. Otherwise the away team wins. The box uses `--pick-win` and `--pick-win-label` (kit yellow, `#EDB01A`), fill `--pick-win-fill`, stroke `--pick-stroke` (1.5 px), radius `--pick-radius`, and padding `--pick-pad` (4 px), so the 3 px team-color border sits inside the outline. On a disagreement row the box keeps the dark fill and a 2 px halo in that same fill. The disagreement badge stays in the Diff. column. The light theme sets the outline and the WINS label to `--black` on a cream fill. The Model legend and column tooltip say the percentage is the home team's win probability.

The estimated score sits under that box. Away points, then home points, rounded to integers for display. When ADR-007 has not passed, the line is the hidden-reason sentence and the Model accuracy block shows MAE as n/a. When it has passed, the block shows margin and total MAE against the baseline and the market. The copy stays an open experiment, not betting picks. The WINS box does not change.

The track record sits under Model accuracy. It is a table on wide screens and a stack of cards at 760 px and below. Each archived week shows finished games, pick hits, final ties, model and market log loss, and margin and total error. A season to date row follows the weeks. One finished week is labeled a small sample. A week with no finals, a week still in progress, and a build that could not read the archive each have their own empty or partial copy. The method link goes to ADR-008. The numbers are not a betting record.

The rest of this note is the design specification, copied as delivered.

---

# endzone-mlops · Diseño del sitio de pronósticos NFL

Sitio estático (GitHub Pages, HTML/CSS/JS vanilla). Tema oscuro por defecto (regla: base #1B1F23). Variante clara opcional con `prefers-color-scheme` activándola con la clase `auto-theme` en `<html>`. Tokens en `tokens.css` (destino: `site/assets/tokens.css`). Paleta tomada sin cambios de `branding/endzone-mlops/source/brand.json`.

## Colores (oscuro)

| Token | Hex | Uso | Contraste sobre #1B1F23 |
|---|---|---|---|
| `--color-bg` | #1B1F23 | fondo | |
| `--color-surface` | #262B30 | tarjetas, tabla | |
| `--color-border` | #30363C | líneas | |
| `--color-text` | #F4EEDF | texto | 14.32:1 (12.34:1 sobre #262B30) |
| `--color-text-muted` | #A9A396 | texto secundario | 6.60:1 (5.69:1 sobre #262B30) |
| `--color-model` / `--color-positive` | #5A86D6 | modelo, diferencia positiva | 4.59:1 sobre #1B1F23, pero 3.95:1 sobre #262B30: por eso en tabla y tarjetas las cifras van en #F4EEDF y el azul solo en barras, punto indicador y leyenda |
| `--color-market` | #A9A396 | barra y etiqueta mercado | 6.60:1 |
| `--color-accent` / `--color-negative` | #D04A63 | acento, "-mlops", negativo | 3.82:1: solo texto grande (h1) o formas; nunca texto normal |
| `--color-disagreement` | #EDB01A | fondo del badge de discrepancia | texto #151515 sobre #EDB01A 8.55:1 |
| `--logo-disc-bg` | #F4EEDF | disco de logo | abreviatura #151515 15.78:1 |

Claro (`auto-theme`): fondo #F4EEDF, texto #151515 (15.78:1), muted #5E5950 (6.01:1), modelo #22509B (6.75:1), acento #AE2440 (5.80:1).

Negativo: el kit no tiene verde/rojo; cifras en texto normal con punto azul (positivo) o crimson (negativo) (no depende solo del color, siempre lleva signo + / −).

## Tipografía (Google Fonts)
- Display y UI: **Jost** 400/500/700, fallback `"Segoe UI", system-ui, -apple-system, sans-serif`.
- Números, kickers, etiquetas: **IBM Plex Mono** 400/500/600, `tabular-nums`, fallback `ui-monospace, Menlo, Consolas, monospace`.
- Escala: 12, 14, 16, 18, 20, 24, 30, 36 px (`--text-xs` a `--text-4xl`).

## Espaciado y forma
Escala 4/8: 4, 8, 12, 16, 24, 32, 48, 64, 96 px (`--space-1` a `--space-9`). Radios 4, 8, 12, 16, pill, disco 50%. Ancho máximo 1120 px.

## Componentes
- **Header**: logo horizontal oscuro (`assets/logo-horizontal-dark.svg`, alto 36 px, 30 px en móvil), nav con subrayado crimson en la página activa. Sticky.
- **Banner honesto**: superficie elevada, borde izquierdo crimson 4 px, etiqueta "Aviso". Texto fijo: "Experimento abierto. Log loss del modelo 0.6377 vs mercado 0.6120 (el mercado gana). No son picks de apuesta." Va siempre arriba de los datos.
- **Selector de semana y umbral**: `<select>` nativos con etiqueta visible; el umbral (pp) viene de `data.json` (`threshold_pp`) y se puede cambiar.
- **Tabla de partidos**: Partido (visita arriba, local abajo), Kickoff (hora CDMX), Modelo, Mercado, barra, Dif. Cifras en mono.
- **Barra modelo vs mercado**: dos pistas de 8 px, azul = modelo, gris cálido = mercado, línea al 50%. `role="img"` con aria-label.
- **Badge de discrepancia**: "DISCREPA" en #151515 sobre amarillo cuando |dif| >= umbral; la fila recibe tinte amarillo 7% y borde izquierdo amarillo 3 px.
- **Logos de equipo**: disco neutro crema (`--logo-disc-size` 36 px) con borde sutil, para que logos oscuros se vean sobre #1B1F23. `<img>` desde nflverse (`team_logo_espn` o `team_logo_wikipedia`) se carga en runtime; si falla o falta, queda la abreviatura en mono dentro del disco.
- **Colores de equipo**: solo como acento fino, `--team-accent` (de `team_color`) en un borde izquierdo de 3 px por equipo, puesto con `style` en línea y validado como hex. `--team-accent-2` (de `team_color2`) reservado para un segmento pequeño. Nunca como fondo ni color de texto. La semántica modelo/mercado/discrepancia es siempre de nuestra paleta.
- **Footer**: marca sola, aviso legal (no es asesoría de apuestas, logos pertenecen a sus dueños), enlace a GitHub.
- **Sprint 2, marcador estimado**: línea bajo el partido. Si el modelo no pasa ADR-007, el texto dice que el estimado está oculto. El bloque Model accuracy muestra el MAE de margen y de total contra el baseline y contra el mercado.
- **Historial**: tabla (escritorio) o tarjetas (760 px o menos) con la semana archivada, el total de la temporada, y el enlace a ADR-008. Una sola semana con finales se marca como muestra pequeña. Sin archivo, la sección dice que no hay historial en ese build.
- **Datos de ejemplo**: si `data.json` trae `is_example: true`, se muestra la etiqueta EJEMPLO sobre la tabla.

## Responsive
- > 760 px: tabla.
- <= 760 px: tarjetas (kickoff + badge, equipos, barra, 3 cifras); tarjeta en discrepancia con borde amarillo. Nav reduce a la página activa. Selectores a ancho completo.

## Accesibilidad
Contraste >= 4.5:1 en todo texto normal; crimson y amarillo nunca como texto pequeño. Foco visible (anillo azul). Enlace "Saltar al contenido". `lang="es"`, caption en tabla, `scope` en encabezados. Discrepancia comunicada con texto ("DISCREPA"), no solo color. Respeta `prefers-reduced-motion`. Alt vacío en logos de equipo (el nombre ya está en texto).
