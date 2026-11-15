# 06 — DeepSeek V4.1 Flash como maestro/juez y LLM-as-judge para front-end

Fecha: 2026-11-03. Alcance: (a) API DeepSeek, (b) términos y licencia, (c) metodología LLM-as-judge, (d) acuerdo juez-humano y tamaño de muestra, (e) renderizado con Playwright, (f) plantilla de juez por pares.
Convención: `[NV]` = no verificado / no abierto; el resto son datos leídos en la URL indicada.

## Resumen

- Id correcto: **`deepseek-flash`** (versión `DeepSeek-V4.1-Flash-1017`), contexto **1M**, salida máx. **384K**, visión **solo en Flash**, thinking **activado por defecto** con `high`; off-peak por 1M: **$0.003** cache-hit, **$0.15** cache-miss, **$0.60** output, peak duplica (01:00–04:00 y 06:00–10:00 UTC, lun–vie) (https://api-docs.deepseek.com/quick_start/pricing).
- Imágenes: `image_url` con data URL, solo en mensajes `user`, **≤1024 tokens/imagen**, cuerpo **48 MiB**, **600** imágenes/request, `detail:"low"` = 512×512 (https://api-docs.deepseek.com/guides/vision).
- Terms §4.2(3) permiten “training other models (such as model distillation)”; §4.2(2) cede los Outputs al usuario; §4.3 tiene opt-out (https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html). Pesos abiertos: **MIT** (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/LICENSE).
- MT-Bench: GPT-4 juez logra **85%** de acuerdo con humanos (sin empates) vs **81%** humano-humano, pero solo es consistente en **65.0%** de pares al invertir el orden (https://arxiv.org/html/2306.05685v4).
- Arena 2026: los jueces eligen su propia respuesta **58%** vs **34%** de humanos; DeepSeek V4 Flash es de los menos auto-preferentes (**−7.1 pp**) y el más humano al decidir (**71%** vs 68%) (https://arena.ai/blog/llm-judge-self-preference).
- Para UI, DesignBench reporta kappa juez-humano **0.8648/0.8428** y accuracy **95.54%/91.89%**; UI Bench usa 10 criterios ponderados con umbrales ship/iterate/reject (https://arxiv.org/html/2506.06251v3, https://ui-bench.dev/ux-judge/methodology).
- Con **40 pares**, el IC95% de una win-rate del 70% es ±13.7 pp y el de κ≈0.6 es ±0.25; para ±0.10 de κ hacen falta **~250–300 pares** (cálculo propio, §d).
- Playwright permite render determinista con viewport fijo, `offline=True`, `route.abort()`, `document.fonts.ready` y `screenshot(animations="disabled", caret="hide", scale="css")` (https://playwright.dev/python/docs/screenshots, https://playwright.dev/python/docs/network).

## Hechos verificados

### (a) API de DeepSeek

**Modelo y límites** — https://api-docs.deepseek.com/quick_start/pricing · https://api-docs.deepseek.com/api/create-chat-completion

| Dato | Valor |
|---|---|
| `model` | `deepseek-flash` (legacy: `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`) |
| Versión servida | `DeepSeek-V4.1-Flash-1017` |
| Base URL OpenAI / Anthropic | `https://api.deepseek.com` / `https://api.deepseek.com/anthropic` |
| Contexto / salida máx. | 1,000,000 / 384K = 393,216 tokens |
| Visión | solo Flash; `deepseek-v4-pro` no |
| Thinking | activado por defecto, esfuerzo `high` |
| Tamaño | 552B backbone; 8B activos en prefill, 16B en decode (no corre en 12 GB) |
| Input cache-hit / miss / output | $0.003 / $0.15 / $0.60 off-peak; $0.006 / $0.30 / $1.20 peak |

Peak = 01:00–04:00 y 06:00–10:00 UTC, lun–vie, excluyendo festivos chinos; el resto es off-peak (en CET invierno: 02:00–05:00 y 07:00–11:00). Concurrencia Flash: 2,500.

**Imágenes** — https://api-docs.deepseek.com/guides/vision

- Tres formas: data URL base64, URL `http(s)` pública o `file_id` de la Files API. Bloque: `{"type":"image_url","image_url":{"url":"data:image/png;base64,...","detail":"original"}}`; también `{"type":"file","file_id":"file-api-..."}`.
- Formatos JPEG/PNG/GIF/WebP; URL ≤8,192 caracteres; cuerpo ≤48 MiB; inline/URL ≤32 MiB; Files ≤64 MiB; ≤600 imágenes/request; 64 MiB totales sin `file_id` (200 MiB con); lado máximo 8,192 px (4,096 px si ≥15). Cada imagen se reescala a ~1300×1300 y cuesta **≤1024 tokens**; `detail:"low"` = 512×512.
- **Solo en mensajes `user`**; en `system`/`assistant` devuelve 400.

**Thinking mode** — https://api-docs.deepseek.com/guides/thinking_mode

- OpenAI: `extra_body={"thinking":{"type":"enabled"|"disabled"}}`; Anthropic: `{"reasoning":{"effort":"none|low|high|max"}}`. `reasoning_effort`: `low|high|max` (default `high`; `medium`/`xhigh`→high, `minimal`→low, `ultra`→max). En thinking **se ignoran** `temperature`, `presence_penalty` y `frequency_penalty`; en Flash `top_p` solo aplica en thinking (0.95–1.0) y en non-thinking queda fijo en 1.0.
- CoT en `message.reasoning_content`. Sin `tools` no hace falta reenviarlo; **con `tools`, reenviar todo el `reasoning_content` anterior o 400**.
- `max_tokens` por defecto: 8K non-thinking, **64K thinking**, 128K con `max`. Fijarlo siempre.

**Tool calling y JSON** — https://api-docs.deepseek.com/guides/tool_calls · https://api-docs.deepseek.com/guides/json_mode

- `strict:true` es Beta: `base_url="https://api.deepseek.com/beta"`, `strict` en cada function, **todas** las propiedades en `required` y `additionalProperties:false`. Tipos: `object, string, number, integer, boolean, array, enum, anyOf` (+ `$def`/`$ref`); no soporta `minLength`, `maxLength`, `minItems`, `maxItems`.
- En thinking **`tool_choice:"required"` y named tool choice no se soportan** (400); usar `auto` o desactivar thinking.
- JSON Output: `response_format={"type":"json_object"}`, incluir “json” en system/user y un ejemplo; si la salida no se reconoce, la API la sustituye por espacios (`U+0020`) y `content` puede quedar vacío con `finish_reason="stop"`. Validar y reintentar.

**Context caching** — https://api-docs.deepseek.com/guides/kv_cache

- Caché en disco automática, “best effort”; un hit exige que el prefijo **coincida completo** con una cache prefix unit persistida (al final del input/output, por prefijo común o por intervalos fijos).
- `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`; se construye en segundos y se limpia en horas o días. Pon rúbrica e instrucciones estables al principio: las imágenes base64 distintas rompen el prefijo.

**Determinismo** — https://api-docs.deepseek.com/api/create-chat-completion

- La referencia **no documenta `seed`**; no hay forma oficial de fijar la semilla, y `temperature` se ignora en thinking. Para reproducibilidad: versión fijada, mismo prompt, dos órdenes, guardar crudo y repetir una submuestra.

### (b) Términos y licencia

- Terms of Use, actualizados **27-mar-2026** (https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html); §1.1 incluye APIs y SDKs.
- §4.2(2): DeepSeek cede al usuario los derechos sobre los Outputs (paráfrasis). §4.2(3): permite “**training other models (such as model distillation)**”, además de uso personal, investigación y productos derivados.
- §4.3: DeepSeek puede usar Inputs/Outputs para mejorar el servicio con de-identificación; opt-out con **“Improve the model for everyone”**.
- §3.1: al publicar Outputs hay que indicar que son generados por IA y verificar su exactitud. §6.2: no usar marcas DeepSeek sin permiso. Ley aplicable: RP China (§10.1).
- Pesos: **MIT License**, “Copyright (c) 2023 DeepSeek” (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/LICENSE): usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y vender, conservando copyright y permiso; permite derivados y destilación. No concede marca.

### (c) Metodología LLM-as-judge

**MT-Bench / Chatbot Arena (Zheng et al., 2023)** — https://arxiv.org/html/2306.05685v4

- MT-Bench: **80** preguntas multi-turno, 8 categorías × 10, 2 turnos. Tres variantes: pairwise, single-answer grading y reference-guided.
- Acuerdo: GPT-4 juez vs humanos **85%** sin empates (2.º turno) vs **81%** humano-humano; single-answer **85%**.
- Sesgo de posición (Tabla 2): consistencia al invertir orden: GPT-4 **65.0%**, GPT-3.5 **46.2%**, Claude-v1 **23.8%**; “only GPT-4 outputs consistent results in more than 60% of cases”.
- Verbosidad (“repetitive list attack”, 23 respuestas): fallo de GPT-4 **8.7%**; Claude-v1 y GPT-3.5 **91.3%**.
- Auto-preferencia: GPT-4 **+10%** de win rate; Claude-v1 **+25%** (los autores no concluyen causalidad).
- Mitigaciones: (1) **swapping** conservador: victoria solo si gana en ambos órdenes, si no empate; (2) few-shot: consistencia **65.0% → 77.5%**, prompt ~4× más caro; (3) CoT/reference-guided en matemáticas: fallo **14/20 → 6/20 → 3/20** (70% → 15%). Multi-turno: mostrar la conversación completa en un solo prompt funciona mejor que partir turnos.

**Auto-preferencia 2026 (Arena)** — https://arena.ai/blog/llm-judge-self-preference

- 12 modelos, **1,460** batallas de Text Arena (1 jun–23 sep 2026), cada una en los dos órdenes; **34,580** veredictos válidos.
- Media: el juez elige su propia respuesta **58%** vs **34%** de humanos (≈70% más); GPT-6 Astra 88% vs 29%.
- **DeepSeek V4 Flash: −7.1 pp** de auto-preferencia (no se auto-prefiere; MiniMax M3 es el más bajo, −21.8); decide en **71%** de batallas vs **68%** humano (el más humano).
- Los jueces coinciden entre sí **79.4%**, con la persona **56.9%**; en 393 batallas de veredicto unánime de IAs, el humano coincidió solo **60.6%**.
- Implicación: el juez no es neutral; mide auto-preferencia, empates y contraste con un segundo juez de otra familia.

**Design2Code (Si et al., 2024/2025)** — https://arxiv.org/html/2403.03163v3

- 484 páginas reales, screenshot→HTML/CSS. Métricas: CLIP (texto enmascarado con inpainting), Block-Match, Text (Sørensen-Dice), Position, Color (CIEDE2000).
- Evaluación humana: 100 ejemplos, **5 anotadores** por par, mayoría, orden aleatorizado. **49%** de las generadas se consideran intercambiables con la referencia; **64%** mejor diseñadas que la referencia.
- Regresión logística para predecir preferencia humana: accuracy **79.9%**; coeficientes Block-Match **0.7429** (p<0.001), Position **0.7605** (p<0.001), Color **0.3461** (p=0.001), CLIP **0.4929** (p<0.001), **Text −0.3541** (p=0.021). Implicación: los humanos pesan layout/posición/color/bloques; un juez que solo compara texto/OCR se aleja del humano.

**WebDev Arena (2025)** — https://lmarena.ai/blog/webdev-arena/

- Battles de apps web, ranking Bradley-Terry, >80,000 votos al publicar, 11 categorías (Website Design 15.3%, Game Dev 12.1%, Clone Dev 11.6%). Infra: microVMs Firecracker vía E2B; Next.js de un solo archivo; 7 modelos con visión al publicar.
- Empate **26%** (Chatbot Arena 35%); **18%** “both bad”, a menudo por dependencias/compilación/TypeScript.
- Salida estructurada (JSON schema) vs libre: la libre gana con mejoras de Arena Score de **+12.98 a +88.76**. Constreñir al **modelo evaluado** puede penalizarlo; constreñir al **juez** es otra cosa.

**DesignBench (2025, v3 2026)** — https://arxiv.org/html/2506.06251v3

- 900 muestras: 430 generación, 359 edición, 111 reparación; React/Vue/Angular/vanilla; 11 temas.
- Métricas visuales CLIP/SSIM; código CSR, CMLS (Jaccard sobre AST), CMCS (CodeBLEU); MLLM-as-judge con GPT-4o.
- Rúbrica: **0–10**; **0–3** poor, **4–6** partial, **7–8** good, **9–10** excellent, con explicación.
- Validación humana (5 doctorandos, mayoría): kappa **0.8648** (edit) y **0.8428** (repair); accuracy **95.54%** y **91.89%**. Coste medio: **$52/modelo**; ~49 s/muestra en generación.

**UI Bench / UX Judge (2026, no peer-reviewed)** — https://ui-bench.dev/ux-judge/methodology

- 80 landing pages, 6 modelos, 10 variaciones; capturas desktop **1440×900** y mobile **390×844**; juez `Composer 2.5`.
- 10 criterios 0–10 con pesos: Visual Identity 12%, Typography & Hierarchy 10%, Layout & Composition 12%, Color & Contrast 8%, Content & Copy 10%, Interaction & CTAs 10%, Brand Fit 10%, Section Completeness 10%, Polish & Craft 10%, Mobile Readiness 8%. Score 0–100; veredictos **ship ≥82**, **iterate 65–81**, **reject <65**.
- Procedimiento: abrir, esperar red, capturar desktop, snapshot DOM/accesibilidad, scroll completo, capturar mobile; puntuar cada criterio con una nota; suma ponderada; JSON con strengths/weaknesses.

### (d) Acuerdo juez-humano y tamaño de muestra

- Acuerdo porcentual `p_o = acuerdos / N`; kappa `κ = (p_o − p_e) / (1 − p_e)`, con `p_e` esperado por azar según marginales (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html). `cohen_kappa_score` acepta etiquetas `A`, `B`, `tie`.
- Kappa no es “porcentaje de acuerdo”: con clases desbalanceadas puede ser bajo aunque `p_o` sea alto (paradoja de kappa). Reportar `p_o`, κ, IC y matriz de confusión.
- IC de proporción (Wilson; cálculo propio):

| N pares | IC95% si p=0.70 | IC95% si p=0.50 |
|---|---|---|
| 40 | 54.6–81.9 (±13.7 pp) | 35.2–64.8 (±14.8 pp) |
| 100 | 60.4–78.1 (±8.8 pp) | 40.4–59.6 (±9.6 pp) |
| 300 | 64.6–74.9 (±5.2 pp) | 44.4–55.6 (±5.6 pp) |
| 400 | 65.3–74.3 (±4.5 pp) | 45.1–54.9 (±4.9 pp) |

- IC de kappa por simulación (2 clases, marginales 50/50, κ≈0.60, 6,000 réplicas; cálculo propio): N=40 → ±0.25; N=60 → ±0.21; N=100 → ±0.15; N=150 → ±0.13; N=200 → ±0.11; N=300 → ±0.09; N=400 → ±0.08.
- Regla: **40 pares = piloto**; **120–150 = análisis principal** (±0.12–0.13 de κ); **300 = reporte fino** (±0.09). Con varios pares por tarea, bootstrap por tarea (cluster), no por par.
- Tamaño para una proporción: `n ≈ z²·p(1−p)/e²`; con `z=1.96`, `p=0.5`: `e=0.10 → 96`, `e=0.05 → 384`.

### (e) Playwright para Python

- `pip install pytest-playwright` (plugin recomendado) o `pip install playwright`; luego `playwright install`. Python ≥3.8; Windows 11 y WSL soportados (https://playwright.dev/python/docs/intro).
- Viewport/DPR: `new_context(viewport={"width":1440,"height":900}, device_scale_factor=1)`; `page.set_viewport_size(...)` (https://playwright.dev/python/docs/emulation).
- Captura: `page.screenshot(path=..., full_page=True, animations="disabled", caret="hide", scale="css")`; `scale` = `"css"|"device"`; `locator.screenshot()` para un elemento (https://playwright.dev/python/docs/screenshots, https://playwright.dev/python/docs/api/class-page).
- `animations="disabled"` detiene CSS animations, transiciones y Web Animations; las finitas se adelantan y las infinitas se cancelan al estado inicial y se reanudan después.
- Red: `context.route("**/*", lambda r: r.abort())`, `route.continue_()`, `offline=True`, `service_workers="block"` (https://playwright.dev/python/docs/network).
- Carga: `goto()` espera `load`; no hay forma universal de saber que terminó (hidratación, lazy-load). Usar centinela `window.__RENDER_READY__` o esperar selector/fuente (https://playwright.dev/python/docs/navigations).
- Timeout por defecto **30,000 ms**; `set_default_timeout`, `goto(..., timeout=)`, `screenshot(timeout=)`.
- Tiempo: `page.clock.set_fixed_time(datetime(...))`, `install()`, `pause_at`, `fast_forward`, `run_for` (https://playwright.dev/python/docs/clock). Movimiento reducido: `new_context(reduced_motion="reduce")`.

## Know-how

### 1. Render reproducible (HTML/CSS/JS)

Regla de oro: **HTML autocontenido** (CSS, fuentes, imágenes y JS inline o data URIs). Si bloqueas la red, un CDN que falte cambia el resultado.

```python
ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
    color_scheme="light", reduced_motion="reduce", locale="en-US", timezone_id="UTC",
    offline=True, service_workers="block")
ctx.route("**/*", lambda r: r.abort() if r.request.url.startswith(("http", "https")) else r.continue_())
page = ctx.new_page(); page.set_default_timeout(15000)
page.set_content(html, wait_until="load")
page.evaluate("document.fonts.ready")
try: page.wait_for_function("window.__RENDER_READY__ === true", timeout=3000)
except PWTimeout: pass
page.wait_for_timeout(300)
page.screenshot(path="desktop.png", full_page=True, animations="disabled", caret="hide", scale="css")
page.set_viewport_size({"width": 390, "height": 844})
page.wait_for_timeout(300)
page.screenshot(path="mobile.png", full_page=True, animations="disabled", caret="hide", scale="css")
```

Lanza con `p.chromium.launch(headless=True)`; asume `browser`, `html` y `PWTimeout` definidos. Envuelve el bloque en `try/except PWTimeout` y marca `render_ok=false` sin llamar al juez.

`full_page=True` captura toda la página; para el fold usa `full_page=False` con 1440×900. No cambies la altura del viewport para “hacerla caber” (rompe `100vh` y el layout responsive). Si supera ~6,000 px, trocea en tiles de 1440×3000: al reescalar a ~1300×1300 el texto se vuelve ilegible.

### 2. SVG

```python
svg = Path("art.svg").read_text(encoding="utf-8")
html = f"<!doctype html><meta charset='utf-8'><body style='margin:0;background:#fff'>{svg}</body>"
page.set_content(html, wait_until="load")
page.evaluate("document.fonts.ready")
page.evaluate("document.querySelectorAll('animate,animateTransform,animateMotion,set').forEach(e=>e.remove())")
page.locator("svg").screenshot(path="art.png", animations="disabled")
```

Para SVG sin tamaño explícito, añade `width`/`height` o envuélvelo en un `div` fijo. Si usa fuentes del sistema, fíjalas con `@font-face` data URI para que Windows/WSL rendericen igual.

### 3. Llamada al juez (OpenAI SDK)

```python
import base64, json
from openai import OpenAI
client = OpenAI(api_key=..., base_url="https://api.deepseek.com")

def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": [
                  {"type": "text", "text": user_text},
                  {"type": "image_url", "image_url": {"url": data_url(a_desktop), "detail": "original"}},
                  {"type": "image_url", "image_url": {"url": data_url(a_mobile),  "detail": "original"}},
                  {"type": "image_url", "image_url": {"url": data_url(b_desktop), "detail": "original"}},
                  {"type": "image_url", "image_url": {"url": data_url(b_mobile),  "detail": "original"}}]}],
    max_tokens=4096, reasoning_effort="high",
    response_format={"type": "json_object"},
    extra_body={"thinking": {"type": "enabled"}},
)
verdict = json.loads(resp.choices[0].message.content)
```

1,000 pares × 2 órdenes = 2,000 llamadas × ~5.5K input + ~2K output ≈ **$3–5 off-peak**, **$6–10 peak** (cálculo propio). Cabe en $25 con reintentos y un segundo juez.

### 4. Plantilla de prompt de juez por pares

`SYSTEM_PROMPT` (fijo, cacheable):

```text
You are a senior front-end design reviewer. Judge two implementations of the same brief
from screenshots only. You do not see code, file names, or model names, and must not guess
them. Judge only visible evidence. Do not reward length, decoration, animation, or
confident copy. A tie is allowed.

Score each candidate 0-10 (integers):
1. Brief fidelity & content completeness (25): required sections, labels, CTAs and facts
   present; nothing required missing or invented.
2. Visual hierarchy & typography (20): reading order, type scale, line length, weight
   contrast.
3. Layout, spacing & composition (15): alignment, grid, whitespace, balance, no
   collisions/overflow.
4. Color, contrast & accessibility (15): coherent palette; body text >= 4.5:1; visible
   focus/hover; color not the only signal.
5. Craft & consistency (15): icons, radii, shadows, borders, spacing tokens consistent;
   no clipped text, broken images, lorem, default styles.
6. Responsive / mobile readiness (10): at 390x844 layout reflows, nav/CTAs usable, no
   horizontal scroll or overlap.

Anchors: 10 exceptional/shippable; 8 strong/minor; 6 competent/visible issues; 4 weak;
2 mostly broken; 0 absent.
Procedure: verify both rendered; list the brief's hard requirements; score A and B
independently with one sentence of visible evidence per criterion; weighted_total =
sum(score*weight)/10; winner = higher total, tie only if |A-B| < 0.5 and no criterion
differs by >= 2; confidence 0-100 reflects ambiguity; output JSON only, no markdown.
```

`user_text`:

```text
BRIEF: {{brief}}
HARD CONSTRAINTS: desktop 1440x900, mobile 390x844.
Must include: {{required_sections}}; must appear verbatim: {{required_copy}}; forbidden: {{forbidden}}.
CANDIDATE A: images 1 (desktop full) and 2 (mobile full). CANDIDATE B: images 3 and 4.
Return exactly this JSON:
{"render_ok":{"A":true,"B":true},
 "criteria":{"brief_fidelity":{"A":0,"B":0,"note":""},"hierarchy":{"A":0,"B":0,"note":""},
             "layout":{"A":0,"B":0,"note":""},"color_a11y":{"A":0,"B":0,"note":""},
             "craft":{"A":0,"B":0,"note":""},"mobile":{"A":0,"B":0,"note":""}},
 "weighted":{"A":0.0,"B":0.0},"winner":"A|B|tie","confidence":0,
 "decisive_reason":"","flags":["both_bad|mobile_broken|a11y_issue|placeholder_text|render_error"]}
```

Reglas: (i) ciego y sin código; (ii) rúbrica fija antes de ver resultados; (iii) anclas numéricas; (iv) empate explícito; (v) no premiar verbosidad/animación; (vi) JSON validable; (vii) una frase de evidencia por criterio.

### 5. Agregación de los dos órdenes

Para cada par `(X,Y)`: orden 1 A=X/B=Y; orden 2 A=Y/B=X. Mapea el ganador a `X`/`Y`/`tie`.

- **Primaria (conservadora, MT-Bench):** victoria solo si el mismo modelo gana en ambos órdenes; cualquier otra combinación = empate.
- **Secundaria (Arena 2026):** mismo ganador en ambos → victoria; un orden gana y el otro empata → victoria; divididos → 0.5 a cada uno; ambos empatan → empate.
- Reporta win-rate `(wins + 0.5*ties)/N` con bootstrap por par, **tasa de flip** y win-rate por orden. Un flip alto es sesgo de posición, no igualdad.

```python
def aggregate(x, y, strict=True):
    # x = orden 1; y = orden 2 (ya mapeados a X/Y/tie)
    if x == y and x in ("X", "Y"): return x, "win"
    if {x, y} == {"X", "Y"}: return "tie", "split"
    if strict: return "tie", "not_both"
    if "tie" in (x, y): return (x if x != "tie" else y), "win_plus_tie"
    return "tie", "both_tie"
```

### 6. Kappa e IC

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

def agreement(y_j, y_h, B=10000, seed=0):
    y_j, y_h = np.asarray(y_j), np.asarray(y_h)
    po = float((y_j == y_h).mean())
    k = cohen_kappa_score(y_j, y_h)
    rng = np.random.default_rng(seed)
    boot = [cohen_kappa_score(y_j[i], y_h[i]) for i in
            (rng.integers(0, len(y_j), len(y_j)) for _ in range(B))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"p_o": po, "kappa": k, "kappa_ci95": (float(lo), float(hi))}
```

Con 40 pares, reporta el resultado como piloto, no como conclusión. Que el autor califique a mano los mismos 40 pares que ve el juez, ciegos y con orden aleatorizado.

### 7. Protocolo mínimo

Mismo Chromium/viewport/DPR/locale/timezone y hash del HTML; rechaza capturas en blanco (`render_ok=false`); par en orden aleatorio (semilla) y su inverso; juez `deepseek-flash`, thinking `high`, `max_tokens=4096`, JSON mode; guarda request, respuesta cruda, `usage` y `system_fingerprint`; valida con Pydantic y reintenta una vez; agrega con la regla conservadora y reporta la de Arena; repite ≥10% para ruido; califica 120–150 pares a mano; segundo juez de otra familia en 10–20%.

## Trampas

- **Thinking viene activado y `max_tokens` por defecto es 64K.** Una llamada puede costar 20× más si no fijas `max_tokens`; usa 2K–4K.
- **`temperature` no hace nada en thinking.** No sirve para hacer al juez determinista; usa dos órdenes + repetición + versión fijada.
- **No hay `seed` documentada.** Guarda los juicios crudos; no prometas reproducibilidad bit-a-bit.
- **JSON mode puede devolver `content` vacío** (todo espacios) con `finish_reason="stop"`. Valida `content.strip()` y reintenta.
- **`strict` es Beta y exige `/beta`.** Si olvidas `additionalProperties:false` o `required` en todas las propiedades, falla.
- **En thinking no puedes forzar el tool con `tool_choice="required"`** (400). Usa JSON mode o `auto`.
- **Con `tools`, reenviar `reasoning_content` es obligatorio** en todos los turnos; si no, 400.
- **Las imágenes solo van en `user`.** Un `system` con imagen devuelve 400.
- **Cada imagen cuesta hasta 1024 tokens.** Cuatro capturas por par = ~4K tokens.
- **La caché no perdona prefijos parciales.** Si pones imágenes antes de la rúbrica, no habrá hit.
- **El sesgo de posición no desaparece con un prompt “imparcial”.** GPT-4 solo era consistente en 65% de pares casi idénticos.
- **Los jueces evitan el empate.** Humanos empatan/“both bad” en 32%; los modelos deciden casi siempre. Mide la tasa de empate.
- **La auto-preferencia es de familia.** OpenAI +37 pp a modelos OpenAI; DeepSeek fue la excepción (−16 pp entre sí), pero coincide más con otras IAs (79.4%) que con humanos (56.9%). Usa segundo juez.
- **Un juez que solo mira texto/OCR se aleja del humano.** En Design2Code la similitud de texto tuvo coeficiente negativo.
- **Constreñir la salida del modelo evaluado puede empeorar el diseño.** WebDev Arena midió +12.98 a +88.76 para salida libre vs JSON schema.
- **`goto()` espera `load`, no “render terminado”.** Fuentes, hidratación y lazy-load llegan después; usa centinela + `document.fonts.ready` + espera corta.
- **Full-page muy altas se vuelven ilegibles** al reescalar a ~1300×1300. Trocea o usa viewport + tiles.
- **Windows y WSL renderizan distinto** (fuentes, antialiasing). Fija el mismo entorno para todas las variantes y dilo en el paper.
- **Kappa puede ser bajo con acuerdo alto** si los marginales están desbalanceados. Reporta `p_o`, κ, IC y matriz.
- **40 pares no alcanzan para un IC útil.** Es piloto; para ±0.10 de κ necesitas ~250–300.

## Referencias

- https://api-docs.deepseek.com/quick_start/pricing — modelo, versión, contexto, precios, peak, concurrencia.
- https://api-docs.deepseek.com/guides/vision — imágenes, límites, tokens; https://api-docs.deepseek.com/guides/thinking_mode — effort, `reasoning_content`, tools.
- https://api-docs.deepseek.com/guides/tool_calls — strict mode y tipos; https://api-docs.deepseek.com/guides/json_mode — JSON Output y `content` vacío.
- https://api-docs.deepseek.com/guides/kv_cache — cache units; https://api-docs.deepseek.com/api/create-chat-completion — esquema, defaults, sin `seed`.
- https://api-docs.deepseek.com/quick_start/rate_limit — 2,500 concurrentes y `user_id`.
- https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html — Terms §4.2(2)(3), §4.3, §3.1, §6.2.
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash — model card; https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/LICENSE — MIT.
- https://arxiv.org/abs/2306.05685 — MT-Bench, sesgos, acuerdo (HTML: /html/2306.05685v4).
- https://arxiv.org/abs/2403.03163 — Design2Code, métricas y evaluación humana (HTML: /html/2403.03163v3).
- https://arxiv.org/abs/2506.06251 — DesignBench, rúbrica 0–10, kappa (HTML: /html/2506.06251v3).
- https://lmarena.ai/blog/webdev-arena/ — WebDev Arena, ties, “both bad”, structured output.
- https://arena.ai/blog/llm-judge-self-preference — auto-preferencia 2026, 1,460 batallas, dos órdenes.
- https://ui-bench.dev/ux-judge y /ux-judge/methodology — 10 criterios, pesos y umbrales (2026, no peer-reviewed).
- https://playwright.dev/python/docs/intro — instalación, Python ≥3.8, Windows/WSL.
- https://playwright.dev/python/docs/screenshots — full page, buffer, element screenshots; https://playwright.dev/python/docs/api/class-page — `animations`, `caret`, `scale`, timeouts.
- https://playwright.dev/python/docs/network — `route`, `abort`, `offline`; https://playwright.dev/python/docs/navigations — `load` e hidratación.
- https://playwright.dev/python/docs/emulation — viewport, DPR, `reduced_motion`; https://playwright.dev/python/docs/clock — tiempo fijo y fast-forward.
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html — fórmula y API de kappa.

## Sin verificar

- `[NV]` Que la API permita **imágenes + `response_format=json_object` en la misma llamada**: no está prohibido, pero no hay ejemplo conjunto.
- `[NV]` Que `strict` mode funcione con imágenes en la misma llamada: el schema estricto aplica a tools, no a la salida de texto.
- `[NV]` Retención y uso de datos específicos de la **API**: §4.3 es genérica y el toggle es de la plataforma. Revisar privacidad/API antes de enviar datos sensibles.
- `[NV]` Que “DeepSeek V4.1 Flash” sea el mismo checkpoint en todo momento: la versión `1017` y `system_fingerprint` ayudan, pero no hay compromiso de inmutabilidad.
- `[NV]` Umbrales de Landis & Koch (0.41–0.60 moderate, 0.61–0.80 substantial): convención de memoria, no abierta en fuente primaria aquí.
- `[NV]` Que la rúbrica de UI Bench sea representativa: es un sitio público, no un paper revisado; inspiración, no validación.
- `[NV]` El ranking actual de WebDev Arena 2026: la página abierta está actualizada a jun-2025; no uses sus posiciones como referencia de 2026.
- `[NV]` Visión “nativa” de DeepSeek V4.1 Flash: la model card describe vision encoder from scratch y preentrenamiento multimodal; “nativa” es etiqueta de marketing, no un benchmark.
