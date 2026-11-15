# Briefs creativos — formato y congelación

Este documento fija el formato de los **150 briefs de evaluación creativa** de
la Fase 1. Los briefs se congelan por SHA-256 antes de cualquier corrida y no
se editan después de ver resultados. La generación con la API oficial de
DeepSeek **todavía no está implementada**: espera la decisión del autor sobre
proveedor y gasto. `scripts/make_briefs.py` solo valida y hashea un archivo ya
escrito; no llama a ninguna API.

## Mezcla obligatoria: 60 / 25 / 15

| `kind` | Proporción | Briefs de 150 |
|---|---|---|
| `page` | 60 % | 90 |
| `component` | 25 % | 37–38 |
| `svg` | 15 % | 22–23 |

Para 150 briefs el redondeo admite 37 o 38 componentes y 22 o 23 SVG.
`check_mix` acepta una desviación de ±1 punto porcentual respecto de
60/25/15; cualquier otra mezcla invalida el archivo.

## Formato: `data/creative/briefs_eval.jsonl`

Un objeto JSON por línea, sin líneas vacías relevantes. Las seis claves son
obligatorias y no se admiten claves extra (un typo debe fallar, no pasar
silencioso):

```json
{"brief_id": "brief-001", "kind": "page", "brief": "Landing page for a specialty coffee subscription: hero with the value proposition, three-tier pricing, testimonial, FAQ and footer. Calm editorial style, one accent colour.", "required_sections": ["hero", "pricing", "testimonial", "faq", "footer"], "required_copy": ["Start your trial", "$14 / month"], "forbidden": ["lorem ipsum", "carousel", "external fonts"]}
```

| Campo | Tipo | Regla |
|---|---|---|
| `brief_id` | string | no vacío y único en todo el archivo |
| `kind` | string | exactamente `page`, `component` o `svg` |
| `brief` | string | no vacío; el texto que recibe el modelo evaluado |
| `required_sections` | list[string] | secciones que deben existir; cadenas no vacías |
| `required_copy` | list[string] | texto literal que debe aparecer; cadenas no vacías |
| `forbidden` | list[string] | elementos o texto prohibidos; cadenas no vacías |

`required_sections`, `required_copy` y `forbidden` pueden ser listas vacías,
pero no cadenas ni listas con elementos vacíos. El brief no incluye capturas
de referencia, código ni nombres de variantes: el juez y el calificador humano
solo ven la captura renderizada.

## Congelación

1. Completar `briefs_eval.jsonl` (150 líneas) con la mezcla 60/25/15.
2. Validar y obtener el hash del archivo **tal cual** (bytes en disco):

   ```powershell
   uv run python scripts/make_briefs.py --briefs data/creative/briefs_eval.jsonl --check-mix --expect 150
   ```

3. Guardar el valor `sha256:` en el anexo v1.1 del prerregistro, en
   `prereg.yaml` → `creative.briefs_sha256`. Ese valor es el compromiso: si el
   archivo cambia, el hash cambia y hay que declarar una versión nueva del
   anexo y repetir la evaluación.
4. Añadir los briefs al índice de descontaminación (Task 12) antes de
   entrenar. Si se usan semillas de WebSight (`HuggingFaceM4/WebSight`, CC BY
   4.0), hay que declararlo en el paper y en la model card por su cláusula
   adicional.

El hash es el de los bytes del archivo (no el de un JSON reserializado), de
modo que cualquier cambio de espacios, orden o contenido lo invalida.

## Pares para la calificación humana ciega

`scripts/human_rate.py` no lee briefs, sino pares ya renderizados. Cada línea
de `--pairs` contiene los dos candidatos canónicos `x` (O-H) e `y` (L); las
capturas pueden ser un string, una lista o un objeto `desktop`/`mobile`:

```json
{"pair_id": "brief-001", "x": {"desktop": "artifacts/pairs/brief-001/O-H/desktop.png", "mobile": "artifacts/pairs/brief-001/O-H/mobile.png"}, "y": {"desktop": "artifacts/pairs/brief-001/L/desktop.png", "mobile": "artifacts/pairs/brief-001/L/mobile.png"}}
```

El script escribe una página HTML autocontenida por lote, copia las imágenes
con nombres opacos y sirve todo desde `127.0.0.1`. La página solo muestra
`A`/`B`/`tie`; ni `pair_id` ni las rutas de origen (que contienen nombres de
variante) llegan al navegador. Cada respuesta se guarda de inmediato en
`--out`:

```json
{"pair_id": "brief-001", "choice": "A", "winner": "X", "display": {"A": "X", "B": "Y"}, "batch": 0, "token": "p0000", "ts": "2026-11-03T12:00:00+00:00"}
```

`choice` es la etiqueta que el autor vio y pulsó; `winner` la traduce a
`X`/`Y`/`tie`, el mismo espacio que usa el juez. `agreement()` del módulo
`gemmaloop.judge.agreement` consume esas listas de `winner`.
