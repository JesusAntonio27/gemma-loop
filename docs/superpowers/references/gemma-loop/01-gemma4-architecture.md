# 01 — Arquitectura exacta de Gemma 4 E2B y E4B (para Gemma-Loop)

Investigado: 2026-11-03. Fuentes primarias: `config.json` y cabecera del `.safetensors` de los repos oficiales de Hugging Face, código de `transformers` (main en `cc1451df`, 2026-11-02; y tag `v5.22.1`), model card de Google, arXiv 2607.02770. Idioma: español; términos técnicos en inglés.

## Resumen

- **E2B:** 35 capas, `hidden_size=1536`, `intermediate_size=6144` (pero **12288 en las 20 capas que comparten KV** por `use_double_wide_mlp=true`), 8 heads, **1 KV head**, `head_dim=256`, `global_head_dim=512`, `sliding_window=512`, `max_position_embeddings=131072`, PLE de 256 dims.
- **E4B:** 42 capas, `hidden_size=2560`, `intermediate_size=10240` uniforme, 8 heads, **2 KV heads**, mismos `head_dim`/`global_head_dim`/ventana/contexto, `use_double_wide_mlp=false`.
- **Patrón local/global confirmado:** E2B repite `4 sliding + 1 full` (periodo 5, 7 periodos, full en 4, 9, …, 34); E4B repite `5 sliding + 1 full` (periodo 6, 7 periodos, full en 5, 11, …, 41). La hipótesis **35 = 7×5 y 42 = 7×6 es correcta**.
- **KV sharing:** E2B comparte KV en las **20** últimas capas (productores: 0–14) y E4B en las **18** últimas (productores: 0–23). En **5.22.1/main** las capas consumidoras no crean `k_proj`/`v_proj`, leen `shared_kv_states` siempre y el checkpoint (que sí trae esos tensores) los ignora al cargar. En **5.5.0** no es así: los módulos existen y se usan cuando `past_key_values is None` (ver Trampas 3).
- **PLE (Per-Layer Embeddings):** una sola tabla empaquetada `embed_tokens_per_layer` de forma `[262144, L*256]` = **2.35 B** de parámetros en E2B y **2.82 B** en E4B (46 %/35 % del checkpoint). Es la mayor parte de los params "no efectivos".
- **Benchmarks oficiales en thinking mode** (arXiv, Table 5): E2B MMLU Pro 60.0, GPQA-D 43.4, LCB v6 44.0, AIME 2026 37.5; E4B 69.4 / 58.6 / 52.0 / 42.5.
- **Licencia Apache 2.0**, sin Gemma Terms of Use para Gemma 4; repos **no gated** (`gated: false`). Aplica la cláusula de trademarks de Apache 2.0 (§6).
- **Código:** `Gemma4ForConditionalGeneration` → `Gemma4Model` → `Gemma4TextModel` → `Gemma4TextDecoderLayer` / `Gemma4TextAttention`. Mínimo `transformers>=5.5.0`.
- **Consecuencia para el loop:** el corte 1/5/1 del diseño cruza la frontera productor→consumidor (capa 15 en E2B, 24 en E4B). Un bloque **dentro de la región de reuse** (E2B 15–34, 4 periodos; E4B 24–41, 3 periodos) evita caché KV, colisiones de índice y cambios de ancho de MLP.

## Hechos verificados

### (a) Configuración de texto

Fuente de toda la tabla (salvo indicación): `text_config` de los cuatro `config.json` oficiales — https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json , https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json , https://huggingface.co/google/gemma-4-E2B/raw/main/config.json , https://huggingface.co/google/gemma-4-E4B/raw/main/config.json (it y base son idénticos; solo cambia `eos_token_id`).

| Campo | E2B | E4B | Nota |
|---|---|---|---|
| `num_hidden_layers` | **35** | **42** | |
| `hidden_size` | 1536 | 2560 | |
| `intermediate_size` | 6144 | 10240 | E2B: ×2 (12288) en capas reuse |
| `num_attention_heads` | 8 | 8 | |
| `num_key_value_heads` | **1** | **2** | también en capas full |
| `head_dim` | 256 | 256 | capas sliding |
| `global_head_dim` | 512 | 512 | capas full |
| `num_global_key_value_heads` | `null` | `null` | no hay override |
| `sliding_window` | 512 | 512 | |
| `layer_types` | 7×(4 sliding+1 full) | 7×(5 sliding+1 full) | último layer = full |
| `num_kv_shared_layers` | **20** | **18** | 20/35 y 18/42 (TR) |
| `hidden_size_per_layer_input` | 256 | 256 | dim PLE |
| `vocab_size_per_layer_input` | 262144 | 262144 | |
| `vocab_size` | 262144 | 262144 | |
| `max_position_embeddings` | 131072 | 131072 | 128K |
| `rms_norm_eps` | 1e-06 | 1e-06 | |
| `dtype` | `bfloat16` | `bfloat16` | clave `dtype`, **no** `torch_dtype` |
| `tie_word_embeddings` | true | true | `lm_head` atado |
| `attention_k_eq_v` | false | false | K≠V en global (ver Trampas) |
| `use_double_wide_mlp` | **true** | **false** | solo E2B |
| `enable_moe_block` | false | false | |
| `final_logit_softcapping` | 30.0 | 30.0 | |
| `hidden_activation` | `gelu_pytorch_tanh` | idem | |
| `rope_parameters.full_attention` | `proportional`, `partial_rotary_factor=0.25`, `rope_theta=1e6` | idem | p-RoPE |
| `rope_parameters.sliding_attention` | `default`, `rope_theta=1e4` | idem | |

Top-level (ambos): `model_type="gemma4"`, `architectures=["Gemma4ForConditionalGeneration"]`, `vision_soft_tokens_per_image=280`, `image_token_id=258880`, `video_token_id=258884`, `audio_token_id=258881`, `boi=255999`, `eoi=258882`, `boa=256000`, `eoa=258883`, `eos=[1,106]`, `tie_word_embeddings=true`, `transformers_version="5.5.0.dev0"`.

Torres multimodales (idénticas en E2B y E4B, verificado con diff de configs):
- `vision_config`: `hidden_size=768`, `num_hidden_layers=16`, `num_attention_heads=12`, `num_key_value_heads=12`, `intermediate_size=3072`, `patch_size=16`, `pooling_kernel_size=3`, `default_output_length=280`, `model_type="gemma4_vision"`.
- `audio_config`: `hidden_size=1024`, `num_hidden_layers=12`, `num_attention_heads=8`, `output_proj_dims=1536`, `conv_kernel_size=5`, `attention_chunk_size=12`, `model_type="gemma4_audio"`.
- Parámetros reales de las torres (cabecera safetensors): visión **167.4 M**, audio **304.8 M**.

### (b) Índices exactos y tamaños reales

| Dato | E2B | E4B |
|---|---|---|
| Índices de capas `full_attention` (0-based) | **4, 9, 14, 19, 24, 29, 34** | **5, 11, 17, 23, 29, 35, 41** |
| Nº de capas full / sliding | 7 / 28 | 7 / 35 |
| Primera capa consumidora de KV (`L - k`) | **15** | **24** |
| Capas productoras que guardan KV (`store`) | 13 (sliding) y 14 (full) | 22 (sliding) y 23 (full) |
| Capas consumidoras (`reuse`) | 15–34 (20) | 24–41 (18) |
| MLP de las capas reuse | 12288 | 10240 |
| Periodos completos (capas) | 7 × 5 | 7 × 6 |
| Región productora = periodos | 0–14 = 3 periodos | 0–23 = 4 periodos |
| Región reuse = periodos | 15–34 = 4 periodos | 24–41 = 3 periodos |

Esto es un cálculo propio sobre `layer_types`, `num_hidden_layers` y `num_kv_shared_layers` del `config.json`; coincide con la regla `kv_sharing_modes` de `configuration_gemma4.py`. **La hipótesis del diseño queda confirmada**: 35 = 7×5 (E2B) y 42 = 7×6 (E4B) son múltiplos exactos del periodo, hay 7 capas full en cada modelo y la última capa es full (índices 34 y 41).

Tamaños reales (cabecera del safetensors vía range request — https://huggingface.co/google/gemma-4-E2B-it/resolve/main/model.safetensors , https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors — y árbol del repo en la API de HF):

| Dato | E2B | E4B |
|---|---|---|
| `model.safetensors` (bytes) | 10 246 621 918 (~10.25 GB) | 15 992 595 884 (~15.99 GB) |
| Parámetros totales (bf16) | **5.123 B** | **7.996 B** |
| `embed_tokens` | [262144, 1536] = 402.7 M | [262144, 2560] = 671.1 M |
| `embed_tokens_per_layer` (PLE) | [262144, **8960**] = **2.349 B** | [262144, **10752**] = **2.819 B** |
| `per_layer_model_projection` | [8960, 1536] | [10752, 2560] |
| `per_layer_projection_norm` | [256] | [256] |
| Nº de tensores | 2011 | 2130 |
| Forma `k_proj` sliding / full | [256,1536] / [512,1536] | [512,2560] / [1024,2560] |
| Forma `q_proj` sliding / full | [2048,1536] / [4096,1536] | [2048,2560] / [4096,2560] |

El card declara "2.3 B effective (5.1 B con embeddings)" y "4.5 B effective (8 B con embeddings)". La definición exacta de "effective" no se publica: `text_params − PLE` da 2.30 B en E2B (coincide) y 4.71 B en E4B (no coincide con 4.5). Para planificar VRAM usad los bytes, no el número "effective".

### (c) Código de `transformers`

Archivo: `src/transformers/models/gemma4/modeling_gemma4.py` (main, commit `cc1451df`). En `v5.22.1` los números de línea cambian (~+5/+10) pero las clases son las mismas.

| Elemento | Nombre / línea |
|---|---|
| Modelo multimodal | `Gemma4ForConditionalGeneration` (2520); contiene `self.model = Gemma4Model` y `lm_head` |
| Ensamblador | `Gemma4Model` (2153); `self.language_model = AutoModel.from_config(config.text_config)` |
| Modelo de texto | `Gemma4TextModel` (1544) |
| Capa decoder | `Gemma4TextDecoderLayer` (1339), `forward` en 1368 |
| Atención | `Gemma4TextAttention` (1150), `forward` en 1199 |
| MLP | `Gemma4TextMLP` (1059) |
| PLE identidad | `Gemma4TextModel.get_per_layer_inputs` (1695) |
| PLE contextual | `Gemma4TextModel.project_per_layer_inputs` (1741) |
| Bucle de capas | 1673: `for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers])` |

Firma exacta de la capa decoder (main, 1368–1379):

```python
def forward(self, hidden_states, per_layer_input=None, shared_kv_states=None,
            position_embeddings=None, attention_mask=None, position_ids=None,
            past_key_values=None, **kwargs) -> torch.Tensor:
```

**PLE, paso a paso (verificado en código):**
1. `embed_tokens_per_layer` es un `Gemma4TextScaledWordEmbedding` de forma `[vocab, L*256]` que escala por `sqrt(256)`; su salida se reordena a `[batch, seq, L, 256]`.
2. `per_layer_model_projection` proyecta `inputs_embeds` (`hidden→L*256`), escala por `1/sqrt(hidden_size)`, reordena a `[batch, seq, L, 256]` y aplica `per_layer_projection_norm` (RMSNorm de 256).
3. Se combinan: `(contextual + identidad) * 2**-0.5` (línea 1774).
4. En el bucle, cada capa recibe `per_layer_inputs[:, :, i, :]` (línea 1674).
5. Dentro de la capa: `gate = per_layer_input_gate(h)` (Linear 1536→256 / 2560→256), `act(gate) * per_layer_input`, `per_layer_projection` (256→hidden), `post_per_layer_input_norm`, y suma residual (líneas 1415–1423).
6. Si se pasa `inputs_embeds` sin `input_ids`, `get_per_layer_inputs` invierte la embedding principal comparando tensores y **lanza `RuntimeError`** si no coinciden exactamente.

Detalles de atención que afectan a cualquier reimplementación: `scaling=1.0` fijo (no `1/sqrt(head_dim)`; la magnitud la controlan QK-norm y p-RoPE), `q_norm`/`k_norm` son RMSNorm por head, `v_norm` es RMSNorm **sin escala**, y RoPE se aplica con `unsqueeze_dim=2` (por head). Los dos tipos de máscara se crean una vez por forward (`create_causal_mask` y `create_sliding_window_causal_mask`) y se indexan por `layer_type`.

**KV sharing (verificado):**
- `Gemma4TextAttention.__init__` calcula `is_kv_shared_layer` y, en 5.22.1/main, **no crea** `k_proj`, `v_proj`, `k_norm`, `v_norm` en esas capas.
- En `forward` (1221): si es consumidora, `key_states, value_states = shared_kv_states[self.layer_type]`; no toca `past_key_values`.
- Las productoras con `store_full_length_kv` copian K/V completos a `shared_kv_states[layer_type]` (1239), que es un `KVSharingDict` importado de `cache_utils`.
- `Gemma4TextModel` crea `shared_kv_states` al inicio del forward y lo devuelve solo si `return_shared_kv_states=True`.
- `Gemma4TextModel._keys_to_ignore_on_load_unexpected` (1593–1601) descarta `layers.{i}.self_attn.{k_proj,v_proj,k_norm,v_norm}` de las capas consumidoras: **el checkpoint los contiene y se ignoran**.

**Versión mínima:** `transformers>=5.5.0`. El tag `v5.5.0` ya contiene `models/gemma4/modeling_gemma4.py`; `v5.4.0` no (404). Fechas PyPI: 5.5.0 → 2026-04-02 (la doc dice "contributed 2026-04-02"), 5.22.1 → 2026-10-30 (última estable). El refactor `kv_sharing_modes` / `per_layer_config` "formalizado" es del commit `cc1451d` (**2026-11-02, posterior a 5.22.1**): en 5.22.1 el KV sharing se calcula inline con `first_kv_shared_layer_idx`. Para el experimento, **fijad 5.22.1**: 5.5.0 cambia el cálculo de K/V de las capas shared cuando no hay caché (Trampa 3). No copiéis lógica de `main` sin comprobar la versión.

### (d) Licencia y gating

| Dato | Valor | Fuente |
|---|---|---|
| Licencia en el model card | `license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license` | README HF de E4B-it |
| Texto de la licencia | Apache License 2.0 completo, sin cláusulas adicionales; "Last updated 2026-04-01" | https://ai.google.dev/gemma/docs/gemma_4_license |
| Gemma Terms of Use | "For Gemma 4 terms, see the Gemma 4 license" (no aplican a Gemma 4) | https://ai.google.dev/gemma/terms |
| Gating | `gated: false` en E2B-it, E4B-it, E2B y E4B | API de HF |
| Tags | `license:apache-2.0`, `arxiv:2607.02770` | API de HF |
| Repo contiene `LICENSE` | **No** (404/entry not found); la licencia vive en el card y en la URL de Google | árbol del repo |

No hay aceptación de términos ni token para descargar. Apache 2.0 §6 no concede derechos de marca: no usar "Gemma" en el nombre del derivado.

### (e) Benchmarks oficiales (thinking mode)

Fuente: model card (`ai.google.dev/gemma/docs/core/model_card_4`) y arXiv 2607.02770 Table 5; valores idénticos en ambos para las filas comunes (el README de HF de E4B repite la misma tabla).

| Benchmark | E2B | E4B |
|---|---|---|
| MMLU Pro | **60.0** | **69.4** |
| AIME 2026 (no tools) | **37.5** | **42.5** |
| LiveCodeBench v6 | **44.0** | **52.0** |
| Codeforces ELO | **633** | **940** |
| GPQA Diamond | **43.4** | **58.6** |
| Tau2 (average over 3) | 24.5 | 42.2 |
| BigBench Extra Hard | 21.9 | 33.1 |
| MMMLU | 67.4 | 76.6 |
| MMMU Pro (vision) | 44.2 | 52.6 |
| MATH-Vision | 52.4 | 59.5 |
| OmniDocBench 1.5 (edit distance ↓) | 0.290 | 0.181 |
| MedXpertQA MM | 23.5 | 28.7 |
| CoVoST | 33.47 | 35.54 |
| FLEURS (↓) | 0.09 | 0.08 |
| MRCR v2 8-needle 128k | 19.1 | 25.4 |

Solo en el technical report (Table 5): SciCode 21.0/24.0; IFBench 38.0/44.0; IFEval 94.6/96.7; Terminal Bench Hard 3.0/8.0; Tau2 por tarea: airline 31.0/52.0, retail 34.6/67.1, telecom 19.7/18.4.

Ajustes de evaluación publicados: **"All models are in thinking mode unless explicitly stated"** (caption Table 5). El card no publica nº de muestras, presupuesto de tokens de razonamiento, ni few-shot. La única configuración de sampling recomendada es `temperature=1.0`, `top_p=0.95`, `top_k=64` (card, "Best Practices"); `generation_config.json` trae exactamente esos valores, `do_sample=true`, `eos_token_id=[1,106,50]`.

Thinking: se activa con el token `<|think|>` al inicio del **system prompt**; la salida razona dentro de `<|channel>thought\n…<channel|>`. En E2B/E4B, si se desactiva, **no se emite** el bloque vacío (a diferencia del resto de la familia). El chat template tiene `enable_thinking=false` por defecto y `preserve_thinking` para turnos con tool calls.

### (f) Technical report

- URL: https://arxiv.org/abs/2607.02770 — "Gemma 4 Technical Report", Google DeepMind, 2026.
- Sobre PLE: una sola frase, "E2B and E4B use per-layer embeddings as in Gemma 3n, making them 2.3B and 4.5B effective out of 5B and 8B total parameters respectively"; no detalla el pipeline (el detalle está en el código).
- Sobre KV sharing: "we share the KV cache with ratios of 20/35 and 18/42 for the E2B and E4B model"; y "local to global attention ratio patterns … 4-to-1 local attention blocks for E2B and 5-to-1 for the rest". También: "re-using keys as values in the global attention layers (**except in E2B and E4B**)" y reducción del 37.5 % del KV cache global por p-RoPE.
- Sobre "E": "The 'E' stands for effective parameters … PLE gives each decoder layer its own small embedding for every token" (card); TR añade que el escalar por bloque (`layer_scalar`) existe "to bound the activation ranges to fit fp16".
- No hay sección de recetas de entrenamiento LoRA ni de loops/recurrencia.

## Know-how

### 1. Dónde y cómo se reestructura el stack

El único punto que hay que sustituir es el bucle de `Gemma4TextModel.forward` (main 1673–1688; 5.22.1 1678–1693). Todo lo demás (PLE, máscaras, RoPE, KV) ya está indexado por `layer_types[i]` y por el índice absoluto de capa, así que una "agenda" de índices fuente funciona sin tocar la capa:

```python
# schedule: índices de capa a ejecutar, en orden; el bloque se repite N veces
schedule = prelude + block * N + coda          # p.ej. prelude=[0..14], block=[15..34]
for i in schedule:
    per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
    hidden_states = self.layers[i](
        hidden_states, per_layer_input,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings[self.config.layer_types[i]],
        attention_mask=causal_mask_mapping[self.config.layer_types[i]],
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
```

Con eso, N=1 reproduce el original exactamente (misma secuencia de módulos, mismo PLE, mismo `layer_scalar`). No hace falta duplicar pesos: `self.layers[i]` se reutiliza. Ojo: `self.layers[: self.config.num_hidden_layers]` permite truncar, pero **no** crear repeticiones; el schedule es lo que las crea.

Para "PLE por loop" (decisión abierta del diseño), añadir un vector aprendido de 256 dims por loop **antes** de la gate: `per_layer_input = per_layer_inputs[:, :, i, :] + self.loop_ple[n]`. La tabla empaquetada no se puede reindexar por loop sin cambiar su forma `[vocab, L*256]`.

### 2. KV y caché: la parte que rompe

`DynamicCache` **concatena** en cada `update` (no sobrescribe). Si el bloque incluye capas productoras y ejecutas la misma capa dos veces por paso de decodificación, el KV de esa capa se añade dos veces: la longitud del caché deja de coincidir con `cache_position`/máscara y la generación se corrompe o revienta. Opciones, en orden de seguridad:

1. **Bloque en la región `reuse`** (recomendado): las capas consumidoras leen `shared_kv_states` y **no llaman** a `past_key_values.update`. Cero KV extra, cero colisiones, y el productor (13/14 o 22/23) se ejecuta una sola vez por paso.
2. Bloque en la región productora: hay que dar a cada pasada un índice de caché virtual distinto (`layer_idx + pass*L`) y una caché dimensionada para ello, o desactivar `use_cache` y recalcular; ambas cosas encarecen y complican.
3. "Guardar solo el último loop": es una **aproximación**, no equivalente al forward completo. Si se entrena con todas las pasadas viendo su propio KV y se genera guardando solo el último, los logits cambian. Declararlo si se usa.

Regla práctica: si el bloque está en `reuse`, `shared_kv_states` se recalcula en cada forward a partir del productor y los consumidores ven siempre la K/V completa; no hay que tocar nada más.

### 3. Cortes que respetan la arquitectura

| Modelo | Corte | Capas | Periodos | Riesgo |
|---|---|---|---|---|
| E2B | prelude 0–14, loop 15–34, coda 0 | 20 | 4 | **Bajo**: todo reuse, MLP 12288 uniforme |
| E2B | prelude 0–14, loop 15–29, coda 30–34 | 15 | 3+1 | Bajo: el mismo, con coda de 1 periodo |
| E2B | diseño 1/5/1: 0–4 / 5–29 / 30–34 | 25 | 5 | **Alto**: cruza productor→reuse (15) y el MLP cambia de 6144 a 12288 dentro del bloque |
| E4B | prelude 0–23, loop 24–41 | 18 | 3 | **Bajo**: todo reuse, MLP uniforme |
| E4B | diseño 1/5/1: 0–5 / 6–35 / 36–41 | 30 | 5 | Medio: cruza la frontera 24; sin cambio de anchos, pero con caché de productores |
| E2B | prelude 0–4, loop 5–14 | 10 | 2 | Medio: productores, MLP uniforme 6144, requiere caché por pasada |
| E4B | prelude 0–5, loop 6–23 | 18 | 3 | Medio: productores, MLP uniforme, requiere caché por pasada |

El corte 1/5/1 **sí se transfiere en periodos** (ambos tienen 7), pero no es el más limpio: en E2B mete dos anchos de MLP y en ambos cruza la frontera de KV. Si el objetivo es medir el efecto de los loops con el mínimo de variables confusas, usad un bloque íntegramente en `reuse`.

### 4. Verificación obligatoria de N=1

- Cargar el original y el looped con el **mismo** `attn_implementation` (sdpa/eager) y comparar logits con `torch.testing.assert_close(atol=0, rtol=0)`; con el mismo schedule, N=1 debe ser idéntico, no "parecido". Si no lo es, revisad el índice de PLE y el `layer_scalar`.
- Comparar también con `use_cache=True` en un decode de 2–3 tokens: es donde aparecen los bugs de índices.
- `layer_scalar` es un **buffer** `[1]` por capa (init `ones_`), no un parámetro: si se reconstruye el modelo desde cero, hay que copiarlo; si se reutilizan los módulos, no se toca.

### layer_scalar and residual-stream geometry

`layer_scalar` is not 1: every Gemma 4 decoder layer ends with
``hidden_states *= self.layer_scalar`` (`modeling_gemma4.py:1429`), i.e. it
computes ``h <- s * (h + delta)`` on the **whole residual stream**. The stream is
therefore leaky and gets reset at several depths. For E2B the values from the
methodology review (probe on the fp32 snapshot `4adf157`, assistant positions of
three chat samples; review A, Q1) are:

| layer(s) | `layer_scalar` |
| --- | --- |
| L0 | 0.018 |
| L13 | 0.088 |
| L14 | 0.029 |
| L15 | 0.254 |
| L16-24 | 0.43-0.66 |
| L25-33 | 0.70-0.87 |
| L34 | 0.167 |

The product over the legacy 15-29 block is only **4.3e-4**. Consequences
measured in the same review:

- The stream is rewritten at L0 and L13-15: per-layer ``cos(out, in)`` is
  0.02-0.20 there, and only partial at 10-12, 20 and 23. It is cumulative
  (``cos >= 0.75``) only in 4-9, 16-18, 21-22 and 24-33.
- ``cos(h14, h29) = -0.01`` on 3/3 samples and ``RMS(h29) / RMS(h14) = 1.5-1.8``.
- Block input/output cosine: 15-29 = **0.00**, 15-19 = 0.03, 20-24 = 0.07,
  16-18 = 0.66-0.73, 25-29 = 0.75-0.86.

So a loop over 15-29 does **not** re-apply a refinement to a state layer 15
already produced: pass 2 enters layer 15 from a subspace it never saw. High-cosine
blocks (25-29, 16-18) are 4-5x less disruptive in the untrained probe and cost
8.4-14.3% of N=1 matmul FLOPs per extra pass instead of 43%. This is why the
loop block is configurable (`LoopConfig.block`, inclusive) and why the default
remains the legacy 15-29 arm only for backward compatibility.

Source: `methodology-review-A-architecture-loops.md` (Q1/Q2, 2026-11-04),
saved next to this reference in the SDD folder; raw probe output in the
controller scratchpad (`probe2.py`, `probe_out.txt`).

### 5. Carga y entrenamiento en 12 GB (consecuencias de la arquitectura)

- El checkpoint es multimodal: `model.language_model.*` + `model.vision_tower.*` + `model.audio_tower.*`. La ruta simple (`Gemma4ForConditionalGeneration` + `model.model.language_model`) carga también las torres: **472 M de params ≈ 0.94 GB bf16** que no se usan en texto. Si esos 0.94 GB importan en 12 GB, construid `Gemma4TextModel(config.text_config)` y cargad solo el subconjunto `model.language_model.*` renombrando el prefijo (no hay mapping automático: `Gemma4ForCausalLM.base_model_prefix="model"`).
- bf16 completo: 10.25 GB (E2B) y 15.99 GB (E4B). E4B **no cabe** en 12 GB ni para inferencia; QLoRA 4-bit es obligatorio.
- La tabla PLE es un `nn.Embedding` (`embed_tokens_per_layer`), no un `nn.Linear`: PEFT/LoRA no la adapta y bitsandbytes la cuantiza solo si se pasa el modelo entero a 4-bit. Si se deja en bf16, son 4.70 GB (E2B) / 5.64 GB (E4B) solo para PLE.
- Targets LoRA razonables para el texto: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`; añadir `per_layer_input_gate`/`per_layer_projection` es una ablación aparte. En 5.22.1/main las capas consumidoras **no tienen** `k_proj`/`v_proj`: PEFT no encontrará esos módulos ahí (no es un error, es la arquitectura).
- `tie_word_embeddings=true`: al guardar adaptadores, usad `modules_to_save=["lm_head"]` solo si hace falta; el peso atado no debe duplicarse.

## Trampas

1. **El card generaliza mal**: dice que las capas globales de la familia "feature unified Keys and Values", pero `attention_k_eq_v=false` en E2B/E4B y el TR excluye explícitamente a los E. Haced caso al `config.json` y al TR.
2. **`head_dim` no es único**: 256 en sliding y 512 en full (`per_layer_config`). Un bloque period-aligned mezcla ambos; no se pueden compartir pesos entre una capa sliding y una full.
3. **El checkpoint contiene `k_proj`/`v_proj` en las capas consumidoras** (verificado en el header: layer 15/19/20/34 de E2B los tienen). En 5.22.1/main se ignoran al cargar y no se usan. **En 5.5.0 se cargan y se usan si `past_key_values is None`**: con `use_cache=False` (lo normal en training) esas capas calculan su propia K/V y el modelo no es el mismo que con caché. No uséis 5.5.0 para entrenar; quedaos en 5.22.1/main.
4. **`use_double_wide_mlp=true` solo en E2B** y solo para las 20 capas `reuse`: `intermediate_size` real 12288, no 6144. Cualquier cálculo de FLOPs/VRAM con 6144 en esas capas es erróneo.
5. **El corte 1/5/1 cruza la frontera de KV** en ambos modelos (15 y 24). No es un error fatal, pero sí una variable confusa y un problema de caché.
6. **`DynamicCache.update` concatena**: repetir una capa productora en el mismo paso duplica K/V y desalinea la máscara. Ver Know-how §2.
7. **El orden de los índices importa**: `position_embeddings` y `attention_mask` se eligen por `layer_types[i]` y `per_layer_inputs` por el índice absoluto `i`. Si usáis el índice del loop en vez del índice fuente, N=1 deja de ser el original.
8. **PLE con `inputs_embeds`**: si no pasáis `per_layer_inputs` y no hay `input_ids`, el código invierte la embedding por igualdad exacta y lanza `RuntimeError`. Para multimodal, `Gemma4Model` ya los precalcula con el token PAD en las posiciones de imagen/audio.
9. **`dtype` vs `torch_dtype`**: el config trae `"dtype": "bfloat16"`; los ejemplos oficiales de 5.x usan `dtype=`, no busquéis `torch_dtype` en el JSON.
10. **`eos_token_id`**: el config top-level trae `[1,106]` y `generation_config.json` trae `[1,106,50]`. Si evaluáis con el config y no con el generation config, el modelo no parará en el token 50.
11. **Thinking en E2B/E4B**: al desactivarlo no emiten el bloque `thought` vacío; los arneses que esperan `<|channel>thought\n<channel|>` en todas las variantes fallarán con estos dos modelos.
12. **El 4.5 B "effective" de E4B no se reproduce** con `text_params − PLE` (da 4.71 B). No lo uséis para estimar memoria.
13. **El promedio Tau2 del card no es la media de las tres tareas** del TR: E4B card 42.2 vs media 45.8; E2B 24.5 vs 28.4. Reportad la fuente que uséis y no mezcléis.
14. **`transformers` main ≠ 5.22.1**: el refactor de KV sharing (`kv_sharing_modes`, commit `cc1451d`) es posterior a la última release. Los números de línea y parte de la API interna cambian.
15. **Visión/audio no son gratis**: 167 M + 305 M de parámetros (0.94 GB bf16) y el processor espera `pixel_values`/`input_features`. Si usáis la clase multimodal, `from_pretrained` los materializa aunque no paséis imágenes; para texto puro, cargad solo `language_model`.

## Referencias

- https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json — config exacto de E2B-it (text/vision/audio).
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json — config exacto de E4B-it.
- https://huggingface.co/google/gemma-4-E2B/raw/main/config.json — base E2B; `text_config` idéntico, solo cambia `eos`.
- https://huggingface.co/google/gemma-4-E4B/raw/main/config.json — base E4B; idem.
- https://huggingface.co/api/models/google/gemma-4-E4B-it — `gated:false`, tags `license:apache-2.0`, `arxiv:2607.02770`.
- https://huggingface.co/api/models/google/gemma-4-E2B-it — idem para E2B.
- https://huggingface.co/api/models/google/gemma-4-E4B-it/tree/main — tamaños de archivo (model.safetensors 15 992 595 884 B).
- https://huggingface.co/api/models/google/gemma-4-E2B-it/tree/main — idem (10 246 621 918 B).
- https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors — cabecera safetensors (shapes por capa, PLE, k/v en capas shared).
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/model.safetensors — idem E2B.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md — model card: tabla de benchmarks, licencia, PLE, "E" = effective.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/generation_config.json — sampling y eos.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja — `<|think|>`, canales, `enable_thinking`, `preserve_thinking`.
- https://ai.google.dev/gemma/docs/core/model_card_4 — model card oficial: tabla, best practices, thinking, datos.
- https://ai.google.dev/gemma/docs/gemma_4_license — Apache 2.0 íntegro, sin cláusulas extra.
- https://ai.google.dev/gemma/terms — Terms of Use de Gemma 1–3; remite a la licencia de Gemma 4.
- https://arxiv.org/abs/2607.02770 — abstract del technical report.
- https://arxiv.org/html/2607.02770 — texto: §2 arquitectura, ratios 4:1/5:1, KV 20/35 y 18/42, Table 5 (benchmarks, thinking).
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py — clases, PLE, KV sharing (main `cc1451df`).
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/configuration_gemma4.py — `kv_sharing_modes`, `per_layer_config`, defaults.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/modeling_gemma4.py — versión estable; KV sharing inline, sin `kv_sharing_modes`.
- https://raw.githubusercontent.com/huggingface/transformers/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py — primera versión con Gemma 4 (HTTP 200).
- https://api.github.com/repos/huggingface/transformers/commits?path=src/transformers/models/gemma4/modeling_gemma4.py — commit `cc1451df3b5b218295bc16803569256c53c9d6e2`, 2026-11-02, "[Improvement] Formalize KV sharing (#47290)".
- https://huggingface.co/docs/transformers/main/en/model_doc/gemma4 — doc oficial: sección PLE, "contributed 2026-04-02", stable v5.22.1.
- https://pypi.org/pypi/transformers/json — 5.5.0 (2026-04-02), 5.22.1 (2026-10-30), última estable.
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/cache_utils.py — `DynamicLayer.update` concatena (rompe loops con caché).

## Sin verificar

- **Política de uso adicional**: la página https://ai.google.dev/gemma/prohibited_use_policy existe en la navegación, pero su cuerpo no se pudo extraer (solo nav/JS) y no hay copia en el repo de HF. La licencia de Gemma 4 que sí se abrió es Apache 2.0 puro; **no confirmado** si la Prohibited Use Policy o el "Intended use statement" se consideran términos vinculantes para los pesos de Gemma 4. [NO VERIFICADO]
- **Protocolo de evaluación**: no se publica nº de muestras, few-shot, presupuesto de tokens de thinking ni versión de harness por benchmark (el card solo dice thinking mode y recomienda temp 1.0/top_p 0.95/top_k 64). Reproducir "a <3 puntos" con eso es intrínsecamente aproximado. [NO VERIFICADO]
- **Definición de "effective parameters"**: el card da 2.3 B/4.5 B pero no la fórmula; `text_params − PLE` da 2.30 B/4.71 B. [NO VERIFICADO]
- **Tau2 "average over 3"**: no se sabe si son 3 tareas, 3 semillas o 3 ejecuciones; el promedio de las tres tareas del TR no coincide con el card. [NO VERIFICADO]
- **Reproducción de los benchmarks con el checkpoint público**: no ejecutada aquí; no se ha comprobado que el checkpoint `-it` con `<|think|>` reproduzca las cifras del card en esta máquina. [NO VERIFICADO]
- **Que `Gemma4ForCausalLM.from_pretrained` sobre el repo multimodal falle sin renombrar claves**: deducido de `base_model_prefix` y del header (`model.language_model.*`), no ejecutado. [NO VERIFICADO]
