# 08 — Visión de Gemma 4, evaluación visual y razonamiento espacial
Investigado: 2026-11-03. `transformers==5.22.1`, `peft==0.21.2`, `bitsandbytes==0.50.2`, `torch==2.11.0+cu128` (`.venv`). Fuentes primarias: código instalado y raw de `transformers` v5.22.1, `peft` v0.21.2, model cards de HF, arXiv y cards de datasets. Lo marcado **[NV]** es estimación o no verificado.

## Resumen
- **Torre de visión real:** `Gemma4VisionModel` = **167.364.608 params** (0.335 GB bf16) en E2B y E4B; el proyector `embed_vision` añade 1.18 M (E2B) / 1.97 M (E4B). El card dice "~150M" porque no cuenta la tabla de posiciones 2D (15.7 M). [cálculo propio + config]
- **Soft tokens:** el presupuesto por defecto es **280** (`max_soft_tokens`), pero el número real es `num_soft_tokens_per_image` y puede ser menor (70/140/280/560/1120 son los valores soportados). No asumas 280 exactos.
- **Mezcla imagen-texto:** el processor convierte cada `<|image|>` en `boi + <|image|>×N + eoi`; `Gemma4Model` sustituye esos slots por `pad_token_id`, calcula **PLE antes** de insertar las features de visión y luego hace `masked_scatter`. Por tanto, el PLE de las posiciones de imagen es el del token PAD, no el de la feature visual. Un parche del bucle debe reutilizar `per_layer_inputs[:, :, i, :]` por índice de capa fuente, nunca recalcularlo tras el merge.
- **NF4 + visión intacta:** `llm_int8_skip_modules=[r".*vision_tower.*", r".*embed_vision.*", "lm_head"]`. Ojo: pasar esta lista **desactiva** los skips automáticos (`lm_head`/pesos atados); hay que incluir `lm_head` a mano. Mejor aún: `config.audio_config=None` para no materializar la torre de audio.
- **No llames `prepare_model_for_kbit_training` a ciegas:** PEFT convierte a fp32 todo parámetro bf16 que no sea `Params4bit`; eso incluye PLE (2.35 B/2.82 B → 9.4/11.3 GB) y visión. En 12 GB es OOM o deja el modelo sin sitio. Congela a mano.
- **LoRA solo lenguaje:** `target_modules` como string se evalúa con `re.fullmatch`; usa `r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"`. Las capas consumidoras de KV no tienen `k_proj`/`v_proj`; el regex no falla, simplemente encuentra menos módulos.
- **Cifras oficiales:** E4B/E2B → MMMU-Pro **52.6 / 44.2**, MATH-Vision **59.5 / 52.4**. Son con **thinking** y **1120 vision tokens** (Table 6 del TR), no con los 280 por defecto. A 280: MMMU-Pro 51.4 / 43.2; MATH-Vision 59.2 / **53.0** (E2B sube).
- **MMMU-Pro no dice qué variante es:** Google solo publica "MMMU Pro". El paper define Overall = media de Standard(10 opciones) y Vision. Una pregunta directa en HF sobre si es 10-choice quedó sin confirmación. Mide Standard(10) y Vision por separado y reporta ambos.
- **Espacial barato y accionable:** primario **SpatialEval VQA** (4.635 ítems; licencia no declarada, solo eval) + control **SpatialEval TQA**; confirmatorio **CV-Bench 2D** (1.438, Apache 2.0) o **VSR test** (2.195, CC BY 4.0). MDE aproximado (k=1, α=.05, potencia .80): 2.9 pp / 5.2 pp / 4.2 pp sin parear; con pareado r≈.5: 1.5 / 2.6 / 2.1 pp.
- **Olvido:** el fine-tuning solo texto degrada la visión (VFA: −11.92 pp de media en Qwen2.5-VL-7B; MaRVL 53.5→0.0). LoRA no es inmunidad: MDGD muestra colapsos con LoRA en el backbone del LLM (POPE 85.9→2.4). Replay de imagen-texto: **1% ya ayuda**, 5–10% es el rango seguro; usa COCO captions o TextVQA (CC BY 4.0).

## Hechos verificados (con URL)
### (a) Visión en transformers 5.22.1
Config y arquitectura:

| Dato | Valor | Fuente |
|---|---|---|
| `vision_config.hidden_size` / layers / heads / MLP | 768 / 16 / 12 / 3072 | `config.json` de E2B-it |
| `patch_size` / `pooling_kernel_size` / `position_embedding_size` | 16 / 3 / **10240** | idem |
| `default_output_length` / `max_soft_tokens` | 280 / 280 | idem + `processor_config.json` |
| Soft tokens soportados | **70, 140, 280, 560, 1120** | `image_processing_gemma4.py` |
| `Gemma4VisionModel` params | **167.364.608** | cálculo propio con `Gemma4VisionModel(Gemma4VisionConfig(...))` |
| `embed_vision` params | 1.179.648 (E2B) / 1.966.080 (E4B) | `768 × hidden_size` |
| Card / TR Table 10 | **~150M** (768, 3072, 12, 16) | README E2B/E4B · arXiv 2607.02770 |
- El card excluye la tabla `position_embedding_table` `(2, 10240, 768)` = **15.728.640**; 167.36 − 15.73 = 151.6 M. La torre es un ViT **no causal** (`is_causal=False`) con RoPE axial 2D (mitad x, mitad y). https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/docs/source/en/model_doc/gemma4.md
- El checkpoint trae `model.vision_tower.*`, `model.embed_vision.*`, `model.audio_tower.*`, `model.embed_audio.*`; `Gemma4Model` crea cada torre solo si su config no es `None`.

Flujo exacto de una imagen (código v5.22.1):

1. El chat template escribe un `<|image|>` literal por cada parte `{"type":"image"}` o `{"type":"image_url"}`. https://huggingface.co/google/gemma-4-E2B-it/raw/main/chat_template.jinja
2. `Gemma4Processor.replace_image_token()` devuelve `boi + image_token × num_soft_tokens + eoi`; `num_soft_tokens` sale de `num_soft_tokens_per_image`. IDs: `boi=255999`, `image=258880`, `eoi=258882`, `audio=258881`, `pad=0`. https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/processing_gemma4.py · https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json
3. `Gemma4Model.forward` (`v5.22.1` líneas 2280–2345) construye `image_mask = input_ids == 258880`, clona `llm_input_ids` y pone `pad_token_id` en esos slots; `inputs_embeds = embed_tokens(llm_input_ids)`. https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/modeling_gemma4.py
4. **Antes** del merge visual calcula `per_layer_inputs` con `llm_input_ids` y `llm_inputs_embeds` (ambos con PAD en las posiciones de imagen): `per_layer_inputs = language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)`. Es decir, el PLE de un token visual es el PLE del token PAD.
5. La torre produce `pooler_output` por imagen (N soft tokens), `embed_vision` proyecta 768→hidden y `inputs_embeds.masked_scatter(...)` sustituye los slots; comprueba `numel()`. `Gemma4TextModel.forward` recibe `inputs_embeds` + `per_layer_inputs` y exige exactamente uno de `input_ids`/`inputs_embeds`.

Consecuencias para el parche del bucle:

- `per_layer_inputs` `[B,S,L,256]`: cada capa recibe `[:, :, i, :]` con el índice **fuente**. Si repites la capa 15, la pasada 2 usa el índice 15, no el del bucle. `position_embeddings`/`attention_mask` se eligen por `layer_types[i]`.
- No hay atención bidireccional (`use_bidirectional_attention=None`): mask causal puro; la torre ya codifica la imagen. No recalcules PLE tras el merge o N=1 deja de ser idéntico.

### (b) Benchmarks oficiales de visión
Cifras del card (thinking; IT):

| Benchmark | E4B | E2B | Fuente |
|---|---:|---:|---|
| MMMU Pro | **52.6** | **44.2** | https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md |
| MATH-Vision | **59.5** | **52.4** | idem |
- El TR dice: "We use the maximal supported resolution (**1120 vision tokens**) and report results with 280 vision tokens in Table 12". Table 6 = 52.6/44.2 y 59.5/52.4; Table 12 (280) = MMMU 51.4/43.2, MATH-Vision 59.2/**53.0**. https://arxiv.org/html/2607.02770
- El card no publica prompt, k, max tokens ni si MMMU-Pro es Standard(10), Standard(4), Vision o el Overall (media de Standard(10)+Vision). El paper define Overall como media de Standard(10) y Vision. https://arxiv.org/html/2409.02813v2
- Pregunta directa en la discussion de HF: "for MMMU-Pro, is the evaluation based on 10-choice results?". La respuesta de Google describe el benchmark y el setting vision-only, pero **no confirma** la variante. https://huggingface.co/google/gemma-4-31B-it/discussions/44
- La respuesta de Google sí confirma MATH-Vision: overall accuracy sobre **3.040** problemas, 16 disciplinas, 5 niveles. https://huggingface.co/google/gemma-4-31B-it/discussions/44

MMMU-Pro dataset:

| Dato | Valor |
|---|---|
| Configs / filas | `standard (4)`, `standard (10)`, `vision`; **1.730/config**, 3.460 Standard+Vision, solo `test` |
| Licencia / gating | **Apache-2.0**, `gated: false`; Standard(10) ~682 MB, Vision ~1.66 GB |
| Campos / scoring | Standard: `question`, 10 `options`, `image_1..image_7`, `answer` A–J; Vision: una imagen con pregunta+opciones. Accuracy de letra; Direct y CoT, se reporta el mejor; Overall = media Standard(10)+Vision |
Fuentes: https://huggingface.co/datasets/MMMU/MMMU_Pro/raw/main/README.md · https://huggingface.co/api/datasets/MMMU/MMMU_Pro · https://huggingface.co/api/datasets/MMMU/MMMU_Pro/tree/main

MATH-Vision dataset:

| Dato | Valor |
|---|---|
| Filas / formato | **3.040 test** + **304 testmini**; MC + abiertas; 16 disciplinas; niveles 1–5; 56.9 MB parquet + 52.4 MB imágenes |
| Licencia / gating | **MIT**, `gated: false` |
| Scoring | oficial exact-match tras normalizar (MC letra o texto; abiertas `find_math_answer`/`is_equal`); VLMEvalKit usa juez `gpt-4o-mini` para extraer |
Fuentes: https://huggingface.co/datasets/MathLLMs/MathVision/raw/main/README.md · https://raw.githubusercontent.com/mathllm/MATH-V/main/evaluation/evaluate.py · https://arxiv.org/html/2402.14804

Arneses:

- `lmms-eval`: tareas `mmmu_pro_standard`, `mmmu_pro_vision`, `mmmu_pro_standard_cot`, `mmmu_pro_vision_cot`, `mathvision_test`, `mathvision_testmini`, `mathvision_reason_test`. El grupo `mmmu_pro` mezcla Standard+Vision; usa tareas sueltas. `mmmu_pro_standard` trae `max_new_tokens: 256` (súbelo con thinking); el backend OpenAI (`--model openai --model_args model=...,base_url=...`) tiene `timeout=10 s` por defecto. https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/mmmu_pro/mmmu_pro.yaml · https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/mmmu_pro/mmmu_pro_standard.yaml · https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/models/simple/openai.py
- `VLMEvalKit`: `MMMU_Pro_10c`, `MMMU_Pro_10c_COT`, `MMMU_Pro_V`, `MMMU_Pro_V_COT`; `MathVision`/`MathVision_MINI` (juez externo). También `lighteval` tiene `mmmu_pro:standard-4/standard-10/vision`. Para thinking, `SPLIT_THINK=True` busca `<think>`; define un splitter para `<|channel>thought`. https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/vlmeval/dataset/image_mcq.py · https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/README.md · https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/mmmu_pro.py

Coste aproximado (estimación; calibrar con 20 ítems):

| Benchmark | Ítems | Tokens/ítem (280) | Tokens/ítem (1120) | Total k=1 (280→1120) |
|---|---:|---:|---:|---:|
| MMMU-Pro Standard(10) | 1.730 | ~1k texto + ~1.5k imagen + 4k salida | ~1k + ~4k imagen + 4k salida | ~11–16 M |
| MMMU-Pro Vision | 1.730 | ~0.4k + 0.3k imagen + 4k | ~0.4k + 1.1k imagen + 4k | ~8–10 M |
| MATH-Vision | 3.040 | ~0.4k + 0.3k + 4k | ~0.4k + 1.1k + 4k | ~14–17 M |
Total **~30–45 M tokens por pasada k=1**; k=4 ≈ 120–180 M. A 300 tok/s agregados son ~28–42 h por pasada k=1. Con salidas de 8k, casi se duplica. [NV: longitudes reales]

### (c) Razonamiento espacial barato
| Benchmark | n | Licencia | Formato / scoring | Cifras Gemma |
|---|---:|---|---|---|
| **SpatialEval VQA** | **4.635** | **no declarada** (repo sin LICENSE) | 4 tareas (map, maze, grid, real); MCQ A–D; exact-match contra `oracle_option` | no publicadas para Gemma 3/4 |
| **SpatialEval TQA** | **4.635** | idem | mismas preguntas, solo texto; control de atajo lingüístico | idem |
| **CV-Bench 2D** | **1.438** | **Apache-2.0** | spatial relations + counting; 4 opciones; accuracy | tercera parte: Gemma4-31B ~69.8 global (no E2B/E4B) |
| CV-Bench 3D | 1.200 | Apache-2.0 | depth order + relative distance | idem |
| **VSR test** | **2.195** | **CC BY 4.0** | caption verdadero/falso sobre relación; accuracy binaria | no publicadas |
| 3DSRBench | **2.772** únicos | **CC BY 4.0** | 12 tipos; Accuracy + FlipEval/CircularEval; HF parquet 5.157 filas por aumento | no publicadas para Gemma 4 E2B/E4B |
| BLINK | 3.807 filas HF (val+test) | Apache-2.0 | 14 tareas; `Spatial_Relation` = 286 total (~143 test) | Gemma 3 27B PT: BLINK 39.6; Gemma4-31B no-tool 75.7 (terceros) |
| StepGame | 100k test | **MIT** | 9 relaciones; exact-match; texto | no publicadas |
| SpartQA (human) | ~1.1k QA | **MIT** (repo) | FR/FB/CO/YN; MCQ; accuracy | no publicadas |
| bAbI task 17 / SAT | 1.000 / val 4.001, test 150 | CC BY 3.0 / MIT | positional sí-no / MCQ (test exige CircularEval) | no publicadas |
Fuentes: https://huggingface.co/datasets/MilaWang/SpatialEval/raw/main/README.md · https://raw.githubusercontent.com/jiayuww/SpatialEval/main/README.md · https://huggingface.co/datasets/nyu-visionx/CV-Bench · https://huggingface.co/datasets/cambridgeltl/vsr_random/raw/main/README.md · https://huggingface.co/datasets/ccvl/3DSRBench/raw/main/README.md · https://huggingface.co/datasets/BLINK-Benchmark/BLINK/raw/main/README.md · https://huggingface.co/datasets/michaelszx/StepGame/raw/main/README.md · https://huggingface.co/datasets/tasksource/spartqa-mchoice · https://huggingface.co/datasets/facebook/babi_qa · https://huggingface.co/datasets/array/SAT

- CV-Bench: 2.638 total; 2D 1.438 + 3D 1.200. Ignora el "5.276" de webs terceras (cuenta imágenes/duplicados). VSR: 10.972 pares (train 7.680, val 1.097, test 2.195); el subconjunto direccional del test es ~850 y su MDE sube a 6.8 pp.
- 3DSRBench: 2.100 COCO + 672 HSSD = 2.772; el parquet HF tiene 5.157 filas por FlipEval; usa `compute_3dsrbench_results_circular.py`, no promedies filas. BLINK espacial (~143 test) y SAT test (150) son demasiado pequeños para 3–5 pp: no los uses como primario.
- Gemma 3 TR (PT, val, sin P&S): BLINK **39.6**, SpatialSense VQA **59.4**, CountBench **68.0** para 27B. https://arxiv.org/html/2503.19786
- Gemma 4 official no publica ningún benchmark espacial; solo MMMU-Pro/MATH-Vision. Terceros (SpatialClaw, arXiv 2606.13673) dan Gemma4-31B no-tool: BLINK **75.7**, CV-Bench **69.8**; no hay E2B/E4B. https://arxiv.org/html/2606.13673

Recomendación para el experimento:

1. **Primario: SpatialEval VQA (4.635)** + **TQA** con las mismas preguntas (control de atajo lingüístico). Es lo más cercano a HTML/CSS/SVG: mapas, mazes, grids, posiciones, counting y navegación. Si sube TQA pero no VQA, aprendió texto, no representación visual.
2. **Confirmatorio: CV-Bench 2D (1.438, Apache 2.0)**; CV-Bench 3D como control negativo. **Alternativa legalmente limpia: VSR test (2.195, CC BY 4.0)**, binario y barato. SpatialEval no declara licencia: solo evaluación, no redistribuir.

MDE (α=.05 bilateral, potencia .80, p=.5; k=1):

| n | MDE sin parear | MDE pareado r≈.5 | k=4 (÷2) |
|---:|---:|---:|---:|
| 850 (VSR direccional) | 6.8 pp | 3.4 pp | 1.7 pp |
| 1.438 (CV-Bench 2D) | 5.2 pp | 2.6 pp | 1.3 pp |
| 2.195 (VSR test) | 4.2 pp | 2.1 pp | 1.1 pp |
| **4.635 (SpatialEval)** | **2.9 pp** | **1.5 pp** | **0.8 pp** |
| 3.000 (StepGame submuestreo) | 3.6 pp | 1.8 pp | 0.9 pp |
Fórmula: `(1.96+0.8416) * sqrt(2*p*(1-p)*(1-rho)/n)` con `rho=0` (sin parear) o `rho=.5`; con k muestras por ítem, divide por `sqrt(k)`. Para detectar 2–3 pp necesitas n≥4.600 o pareado; SpatialEval VQA es el único barato que llega.

### (d) Olvido y mitigaciones
- **Text-only SFT degrada visión (evidencia directa):** VFA (ACL 2026) entrena el backbone de 5 MLLMs con 100k muestras solo texto. En Qwen2.5-VL-7B la media multimodal cae **46.29 → 34.37 (−11.92)** y MaRVL colapsa **53.50 → 0.00**; Idefics3-8B cae 3.40 pp; LLaVA-OV-4B cae 0.92 pp. No es LoRA específicamente (SFT full/LoRA), pero es exactamente el régimen "solo texto". https://aclanthology.org/2026.acl-long.407.pdf
- **LoRA en el backbone puede colapsar la visión (evidencia con fine-tuning visual):** MDGD (EMNLP 2025 Findings) fine-tunea LLaVA-1.5 con LoRA (29 M params) en Flickr30K. GQA **61.94→17.74**, POPE **85.90→2.40**, MMBench **64.30→9.55**, SQA 66.80→5.38; en OKVQA, MMBench 64.30→37.31. La visión está congelada; el daño viene del stack de lenguaje. https://arxiv.org/html/2502.11740v2
- **LoRA no es inmunidad:** EMT (2023) muestra que el fine-tuning LoRA de LLaVA en un dataset de imágenes degrada otros datasets visuales tras **una epoch**, incluso con el encoder congelado. Y "Fine-tuning MLLMs Without Forgetting" (2026): LoRA conservador mantiene OOD-imagen (−2.97 pp), pero incluso el full fine-tuning regularizado cae 10.4 pp en **OOD-texto con imágenes ID** por sobreajuste de tarea; mezclar datos lo arregla. https://arxiv.org/html/2309.10313v4 · https://arxiv.org/html/2603.14493v1
- **Replay de imagen-texto:**
  - Continual instruction tuning para LMMs y CL-CrossVQA (WACV 2025): guardar **1%** de cada tarea y mezclarlo ya reduce el olvido; el replay supera a la regularización. https://arxiv.org/html/2311.16206v1 · https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_CL-Cross_VQA_A_Continual_Learning_Benchmark_for_Cross-Domain_Visual_Question_WACV_2025_paper.pdf
  - Adapt-∞: replay aleatorio del 10% baja el olvido de 26% a 6.6%; podar a 25k lo baja a 2.1%. https://arxiv.org/html/2410.10636v1
  - "Fine-tuning MLLMs Without Forgetting": mezclar 50% de LLaVA-665K mantiene la tarea objetivo a ~1 pp y recupera OOD-texto; 70% no mejora más. Es el caso agresivo, no el mínimo. https://arxiv.org/html/2603.14493v1
- **Datasets de replay con licencia permisiva:**
  - **COCO captions** (`HuggingFaceM4/COCO`): card **CC BY 4.0** (las imágenes son de Flickr con licencias individuales); 5 captions. https://huggingface.co/datasets/HuggingFaceM4/COCO/raw/main/README.md
  - **TextVQA** (`facebook/textvqa`): **CC BY 4.0**; 45.336 preguntas / 28.408 imágenes. https://textvqa.org/dataset/ · https://huggingface.co/datasets/facebook/textvqa
  - **DocVQA** (`lmms-lab/DocVQA`): card **Apache-2.0**; val 5.349 + test 5.188; útil para layout. **ChartQA** es **GPL-3.0**: no lo mezcles. https://huggingface.co/datasets/lmms-lab/DocVQA/raw/main/README.md · https://huggingface.co/datasets/ahmed-masry/ChartQA/raw/main/README.md

## Know-how
### 1. Cargar E2B/E4B en NF4 con visión bf16 y audio fuera
```python
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, BitsAndBytesConfig
MODEL = "google/gemma-4-E4B-it"
cfg = AutoConfig.from_pretrained(MODEL); cfg.audio_config = None  # sin audio_tower
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=[r".*vision_tower.*", r".*embed_vision.*", "lm_head"],
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, config=cfg, quantization_config=bnb,
    dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
)
```
- `llm_int8_skip_modules` se aplica también a 4-bit (`quantizer_bnb_4bit.py` línea 121) y usa `re.match`/sufijo sobre el **nombre completo** del módulo (`quantizers_utils.py`). Por eso `"vision_tower"` a secas **no** salta los hijos; `.*vision_tower.*` sí.
- Pasar `llm_int8_skip_modules` desactiva `get_keys_to_not_convert`, que normalmente protege `lm_head` y pesos atados (`base.py` 259–278). Incluye `"lm_head"` explícitamente.
- `config.audio_config=None` evita cargar 305 M de audio; el checkpoint tendrá claves `model.audio_tower.*`/`model.embed_audio.*` inesperadas (warning, no error).
- El PLE (`embed_tokens_per_layer`) y `embed_tokens` son `nn.Embedding`: **NF4 no los cuantiza**. Presupuesta PLE 4.70 GB (E2B) / 5.64 GB (E4B) en bf16, más 0.34 GB de visión y 0.81/1.34 GB de embeddings. E2B ≈ 6.9 GB; E4B ≈ 9.5 GB antes de activaciones. [NV: medir]

### 2. Congelar visión sin romper la memoria
No uses `prepare_model_for_kbit_training` sin comprobar el cast: PEFT hace `param.data.to(torch.float32)` para todo bf16 que no sea `Params4bit` (`peft/utils/other.py` 193–209). Eso convierte PLE y visión a fp32 (E2B PLE 9.4 GB; E4B 11.3 GB). Alternativa:

```python
for p in model.parameters():
    p.requires_grad_(False)
model.model.vision_tower.eval()
model.model.embed_vision.eval()
# solo si hiciste el cast a fp32 por error:
# model.model.embed_tokens_per_layer.to(torch.bfloat16)
# model.model.vision_tower.to(torch.bfloat16); model.model.embed_vision.to(torch.bfloat16)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
```
- `prepare_model_for_kbit_training(..., use_gradient_checkpointing=True)` hace el freeze y `enable_input_require_grads`, pero el cast fp32 es letal en 12 GB; si lo usas, mueve PLE a CPU antes o vuelve a bf16 y vigila el pico. No pases `pixel_values` en batches de texto: la torre no corre; en replay sí, en `eval()`, sin grafo pero con activaciones.

### 3. LoRA solo en la pila de lenguaje
```python
from peft import LoraConfig, get_peft_model
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
)
model = get_peft_model(model, lora)
```
- `target_modules` string se evalúa con `re.fullmatch` (`peft/utils/other.py` 1516–1521; `tuners_utils.py` 2370–2371), no con substring. El patrón debe cubrir todo el nombre.
- Para texto puro (`Gemma4ForCausalLM`) el prefijo es `model.layers...`, no `model.language_model.layers...`.
- Añade `|per_layer_(input_gate|projection)` solo como ablación; no es el LoRA estándar.
- Verificación: `assert not any("vision_tower" in n or "embed_vision" in n for n, p in model.named_parameters() if p.requires_grad)`.
- En E2B/E4B las capas consumidoras (15–34 / 24–41) no tienen `k_proj`/`v_proj`; el regex las ignora sin error.

### 4. Processor y mensajes con imagen
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "file:///ruta/imagen.png"},
        {"type": "text", "text": "Describe la relación espacial."},
    ],
}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True,
)
```
- El template acepta `image` y `image_url`; el valor de la URL lo carga el processor. También puedes pasar `images=[pil]` y un texto con `<|image|>`.
- Para 1120 tokens: `processor.image_processor.max_soft_tokens = 1120` y `processor.image_seq_length = 1120`; para 280 deja los defaults. El tamaño real lo confirma `inputs["num_soft_tokens_per_image"]` (no está en `model_input_names`).
- `Gemma4Processor.model_input_names` incluye `pixel_values`, `image_position_ids`, `num_soft_tokens_per_image`; el modelo espera `image_position_ids` con `(-1,-1)` en padding.

### 5. Parche del bucle preservando PLE y KV
```python
for i in schedule:                       # p.ej. [0..14] + [15..34]*N
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
- Usa `i` (índice fuente) para PLE, tipo de atención y máscara. Si añades un vector por loop, súmalo después del slice: `per_layer_input + self.loop_ple[n]`.
- Bloque recomendado: E2B 15–34, E4B 24–41 (región `reuse`): los consumidores leen `shared_kv_states` y no actualizan `past_key_values`. Si repites una capa productora con `use_cache=True`, `DynamicCache.update` concatena y duplica KV.
- En multimodal, `per_layer_inputs` ya viene calculado por `Gemma4Model`; no lo recalcules. Si tu forward recibe `inputs_embeds` sin `per_layer_inputs`, `get_per_layer_inputs` invierte la embedding y lanza `RuntimeError` si no coincide exactamente.

### 6. Labels en entrenamiento con imágenes
```python
labels = input_ids.clone()
for tok in (255999, 258880, 258882):      # boi, image, eoi
    labels[input_ids == tok] = -100
labels[input_ids == processor.tokenizer.pad_token_id] = -100
```
- El processor **no** enmascara los tokens de imagen; si no lo haces, el loss incluye predecir `<|image|>`/boi/eoi y el modelo aprende a generarlos.
- Para SFT de chat, enmascara system/user y deja solo turnos assistant. El template de Gemma 4 no lleva etiquetas `{% generation %}`; `assistant_only_loss=True` de TRL no basta sin comprobarlo.
- En replay imagen-texto, el loss puede ser caption/VQA; no uses los benchmarks MMMU/MATH-Vision como datos de entrenamiento.

### 7. Evaluación visual
```bash
# lmms-eval, servidor OpenAI-compatible propio (los loops no van en vLLM)
python -m lmms_eval --model openai \
  --model_args model=gemma-loop,base_url=http://localhost:8000/v1,api_key=x,timeout=180 \
  --tasks mmmu_pro_standard,mmmu_pro_vision,mathvision_test \
  --batch_size 1 --gen_kwargs max_new_tokens=16384,temperature=1.0,top_p=0.95
```
- Para el modelo local con clase parcheada, usa `--model hf` con un wrapper propio o el servidor; `vLLM` no soporta la arquitectura con loops.
- `mmmu_pro_standard` necesita `max_new_tokens` > 256; `mathvision_test` ya trae 16384. El servidor debe aplicar `enable_thinking=True` y devolver el texto completo; el arnés extrae la última `Answer: X`.
- `VLMEvalKit` equivalente: `MMMU_Pro_10c`, `MMMU_Pro_V`, `MathVision` (juez). Activa `SPLIT_THINK=True` solo si defines un `split_thinking` para `<|channel>thought...<channel|>`; el default busca `<think>` y no sirve para Gemma 4.
- Para comparar con Google, corre **1120** tokens/imagen en un subconjunto (p.ej. 300 ítems) y **280** en el set completo; declara ambas.

### 8. Evaluación espacial
- SpatialEval: clona `jiayuww/SpatialEval`, usa `evals/evaluation.py --mode vqa|tqa --task spatialmap|mazenav|spatialgrid|spatialreal`; exact-match contra `oracle_option`. Corre VQA y TQA con el mismo prompt y compara.
- CV-Bench: `lmms-eval --tasks cv_bench_2d,cv_bench_3d` o `VLMEvalKit --data CV-Bench-2D,CV-Bench-3D`. Scoring: accuracy por tarea; el global es `0.5*(0.5*(2D_ADE+2D_COCO)+3D_Omni)`.
- VSR: descarga `cambridgeltl/vsr_random` + imágenes COCO; prompt binario "Is the caption true?"; accuracy. Filtra relaciones direccionales si quieres una señal más limpia (n≈850).
- 3DSRBench: `lmms-eval --tasks 3dsrbench_circular` o el script oficial; no promedies el parquet de 5.157 filas.
- Fija `enable_thinking` y `max_new_tokens` iguales en original y looped; para MCQ espacial, 1–2k tokens suele bastar. Reporta IC Wilson y MDE.

### 9. Replay para no perder visión
- Presupuesto: **1% de los tokens de entrenamiento como mínimo**; 5–10% si el presupuesto lo permite. Para 100–200 M tokens: 1–2 M tokens de replay (≈2.500–5.000 imágenes a 280 tokens + texto); 5% ≈ 12.500 imágenes.
- Mezcla concreta barata: 2.000 imágenes COCO con caption + 1.000 TextVQA (≈0.84 M tokens de imagen + ~0.3 M de texto) ≈ 1% de un run de 100 M. Añade 500–1.000 DocVQA si el objetivo es layout/HTML.
- Frecuencia: incluye el replay en cada batch (p.ej. 1 muestra visual por cada 8–16 muestras de texto) o cada N steps; no lo dejes para el final.
- Mantén la visión congelada y el LoRA solo en lenguaje; el replay sirve para que el LoRA siga viendo cómo se consumen los soft tokens visuales, no para entrenar la torre.
- Si no puedes permitirte imágenes, al menos evalúa visión antes/después y reporta el daño; no asumas que LoRA lo evita.

## Trampas
1. **"280 por imagen" no es exacto.** Es el presupuesto máximo por defecto; `num_soft_tokens_per_image` puede ser 70/140/…/280. Para 1120 hay que cambiar `max_soft_tokens` y `image_seq_length`; el modelo acepta hasta 1120.
2. **Las cifras del card son a 1120, no a 280.** MMMU-Pro E4B 52.6→51.4 y E2B 44.2→43.2 al bajar a 280; MATH-Vision E2B 52.4→**53.0** (sube). No compares tu run de 280 con 52.6/59.5.
3. **El PLE de las posiciones de imagen es PAD.** `Gemma4Model` calcula PLE antes del `masked_scatter`, con `llm_input_ids` donde los slots de imagen son `pad_token_id`. No recalcules PLE tras el merge ni asumas que la feature visual entra en el PLE.
4. **No uses el índice del loop para PLE/máscara/posición.** `per_layer_inputs[:, :, i, :]`, `layer_types[i]` y `position_embeddings[layer_types[i]]` usan el índice **fuente**. Con el índice del loop, N=1 no reproduce el original.
5. **`llm_int8_skip_modules` con `"vision_tower"` no salta los hijos.** El matching es `re.match` desde el inicio o sufijo exacto; usa `.*vision_tower.*`/`.*embed_vision.*`.
6. **Pasar `llm_int8_skip_modules` desactiva los skips automáticos.** `get_modules_to_not_convert` devuelve `[] + tu lista`; incluye `"lm_head"` o cuantizarás el head atado.
7. **`prepare_model_for_kbit_training` upcasta PLE/visión a fp32.** E2B PLE 2.35 B → 9.4 GB; E4B 2.82 B → 11.3 GB. En 12 GB, OOM. Freeze manual.
8. **NF4 no cuantiza `nn.Embedding`.** PLE y `embed_tokens` siguen bf16: 4.70/5.64 GB + 0.81/1.34 GB. Los 4-bit solo aplican a lineales.
9. **La torre de audio se carga si no la quitas.** `config.audio_config=None` antes de `from_pretrained`; si no, ~305 M params y claves inesperadas.
10. **Las capas consumidoras no tienen `k_proj`/`v_proj`.** Un `target_modules` con k/v no falla, pero solo entrena 15/24 capas productoras en esos módulos.
11. **PEFT usa `re.fullmatch`, no substring.** Un patrón `q_proj` no coincide con `model.layers.0.self_attn.q_proj`; usa el nombre completo o `.*`.
12. **Repetir una capa productora con caché duplica KV.** `DynamicCache.update` concatena; el bloque debe estar en la región `reuse` o entrenar con `use_cache=False`.
13. **Google no dice qué variante de MMMU-Pro es.** El card solo dice "MMMU Pro"; el paper define Overall = media Standard(10)+Vision. Mide Standard(10) y Vision; no etiquetes 52.6 como "vision".
14. **MMMU-Pro Standard(10) baraja opciones con imágenes.** Los `<image_i>` se refieren a `image_i`, no al orden de opciones; usa el script oficial o lmms-eval.
15. **MATH-Vision oficial es exact-match; VLMEvalKit usa juez.** Son métricas distintas; elige una y declara. El juez añade coste y ruido.
16. **`SPLIT_THINK=True` de VLMEvalKit busca `<think>`.** Gemma 4 usa `<|channel>thought ... <channel|>`; define un splitter propio o el thinking contamina la respuesta.
17. **BLINK espacial es pequeño.** `Spatial_Relation` tiene 286 filas totales (~143 test); MDE ~11 pp. No detecta el efecto que buscas.
18. **SAT test son 150 ítems.** Incluso con CircularEval, MDE ~11 pp; no es un benchmark confirmatorio.
19. **SpatialEval no declara licencia.** Úsalo solo para evaluación; no redistribuyas ítems ni lo mezcles en training.
20. **ChartQA es GPL-3.0.** No lo metas en un dataset de replay permisivo; para charts usa COCO/TextVQA o datos propios.
21. **Los labels incluyen tokens de imagen si no los enmascaras.** Enmascara 255999/258880/258882 y padding; si no, el loss aprende a generar marcadores.
22. **`pixel_values` hace correr la torre aunque el texto sea el objetivo.** En batches de texto no los pases; en replay, congela y pon `eval()`.
23. **`dtype` vs `torch_dtype`.** En transformers 5.x usa `dtype=` (el config trae `"dtype": "bfloat16"`); `torch_dtype` está deprecado.
24. **El MDE se dispara con n pequeño.** Con 150–1.000 ítems no detectas 3 pp ni con k=4; usa n≥4.600 o pareado.

## Referencias
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json — config exacto (visión/texto/audio, `image_token_id=258880`, `vision_soft_tokens_per_image=280`).
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json — idem E4B.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/processor_config.json — `max_soft_tokens=280`, `patch_size=16`, `pooling_kernel_size=3`.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/chat_template.jinja — `<|image|>` para `image`/`image_url`.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/README.md — card: benchmarks, "~150M" vision encoder.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md — card: 52.6/59.5.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/docs/source/en/model_doc/gemma4.md — 280 soft tokens, tabla, RoPE 2D, ejemplo de processor.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/modeling_gemma4.py — `Gemma4Model`, `get_image_features`, PLE, `masked_scatter`, loop.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/processing_gemma4.py — `replace_image_token`, `num_soft_tokens_per_image`.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/image_processing_gemma4.py — soft tokens 70/140/280/560/1120, resize.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/configuration_gemma4.py — `position_embedding_size=10240`, `image_token_id`.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/quantizers/quantizer_bnb_4bit.py — skip modules en 4-bit.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/quantizers/quantizers_utils.py — matching de `llm_int8_skip_modules`.
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/quantizers/base.py — `get_modules_to_not_convert`; https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes — `BitsAndBytesConfig`.
- https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/utils/other.py — fullmatch y `prepare_model_for_kbit_training` (upcast fp32); https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/tuners/tuners_utils.py — matching.
- https://huggingface.co/docs/peft/main/en/package_reference/lora — `target_modules` string = regex.
- https://arxiv.org/html/2607.02770 — Gemma 4 TR: Table 6 a 1120, Table 12 a 280.
- https://arxiv.org/html/2503.19786 — Gemma 3 TR: BLINK 39.6, SpatialSense 59.4, CountBench 68.0; https://arxiv.org/html/2606.13673 — SpatialClaw (Gemma4-31B: BLINK 75.7, CV-Bench 69.8).
- https://huggingface.co/datasets/MMMU/MMMU_Pro/raw/main/README.md — 1.730/config, 3.460, Apache-2.0; https://huggingface.co/api/datasets/MMMU/MMMU_Pro — `gated:false`.
- https://arxiv.org/html/2409.02813v2 — MMMU-Pro: Overall = media Standard(10)+Vision; 3.460.
- https://huggingface.co/google/gemma-4-31B-it/discussions/44 — Google confirma MATH-Vision 3.040; no confirma variante MMMU-Pro.
- https://huggingface.co/datasets/MathLLMs/MathVision/raw/main/README.md — 3.040/304, MIT; https://raw.githubusercontent.com/mathllm/MATH-V/main/evaluation/evaluate.py — exact-match.
- https://arxiv.org/html/2402.14804 — MATH-Vision: 16 disciplinas, 5 niveles, testmini 304.
- https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/mmmu_pro/mmmu_pro.yaml — grupo; .../mmmu_pro_standard.yaml — `max_new_tokens: 256`.
- https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/tasks/mathvision/mathvision_test.yaml — exact-match / testmini.
- https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/main/lmms_eval/models/simple/openai.py — backend OpenAI, timeout 10 s.
- https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/vlmeval/dataset/image_mcq.py — `MMMU_Pro_10c/V`, `CV-Bench`, `3DSRBench`; .../image_vqa.py — `MathVision`.
- https://huggingface.co/datasets/MilaWang/SpatialEval/raw/main/README.md — 4.635 × 3 modalidades; sin licencia declarada.
- https://raw.githubusercontent.com/jiayuww/SpatialEval/main/README.md — exact-match, tareas, evaluación.
- https://huggingface.co/datasets/nyu-visionx/CV-Bench — 2.638; Apache-2.0; fórmula.
- https://huggingface.co/datasets/cambridgeltl/vsr_random/raw/main/README.md — CC BY 4.0; splits.
- https://huggingface.co/datasets/ccvl/3DSRBench/raw/main/README.md — 2.100+672, CC BY 4.0, FlipEval/Circular.
- https://huggingface.co/datasets/BLINK-Benchmark/BLINK/raw/main/README.md — 14 tareas, Apache-2.0.
- https://huggingface.co/datasets/michaelszx/StepGame/raw/main/README.md — MIT, 100k test (versión completa).
- https://huggingface.co/datasets/tasksource/spartqa-mchoice — MIT.
- https://huggingface.co/datasets/facebook/babi_qa — CC BY 3.0, task 17.
- https://huggingface.co/datasets/array/SAT — MIT, test 150.
- https://aclanthology.org/2026.acl-long.407.pdf — VFA: text-only SFT −11.92; MaRVL 53.5→0.
- https://arxiv.org/html/2502.11740v2 — MDGD: LoRA colapsa GQA/POPE/MMBench.
- https://arxiv.org/html/2309.10313v4 — EMT: LoRA fine-tuning degrada otros datasets.
- https://arxiv.org/html/2603.14493v1 — LoRA conservador y mezcla 50%.
- https://arxiv.org/html/2311.16206v1 — replay 1% en continual instruction tuning de LMM.
- https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_CL-Cross_VQA_A_Continual_Learning_Benchmark_for_Cross-Domain_Visual_Question_WACV_2025_paper.pdf — 1% replay.
- https://arxiv.org/html/2410.10636v1 — Adapt-∞: replay 10% baja olvido 26→6.6.
- https://huggingface.co/datasets/HuggingFaceM4/COCO/raw/main/README.md — CC BY 4.0; https://textvqa.org/dataset/ — TextVQA CC BY 4.0 (45.336/28.408).
- https://huggingface.co/datasets/lmms-lab/DocVQA/raw/main/README.md — Apache-2.0; val/test.
- https://huggingface.co/datasets/ahmed-masry/ChartQA/raw/main/README.md — GPL-3.0 (evitar).

## Sin verificar
- **Variante exacta de MMMU-Pro que reporta Google** (Standard 10 / Standard 4 / Vision / Overall). El card no lo dice; la discussion 44 no lo confirma; el paper define Overall como media. [NV]
- **Protocolo y reproducción:** no ejecutado que 52.6/44.2 y 59.5/52.4 salgan del checkpoint público; se desconocen k, max tokens, prompt, extracción y si MATH-Vision usa exact-match o juez. Solo se sabe thinking + 1120 tokens. [NV]
- **Cifras espaciales oficiales de E2B/E4B:** no publicadas; las de Gemma4-31B (BLINK 75.7, CV-Bench 69.8) son de terceros. [NV]
- **SpatialEval / "~150M" / memoria:** licencia no declarada (solo eval); el "~150M" coincide con 167.4 M − 15.7 M pero Google no lo explica; 6.9/9.5 GB son estimaciones, no medidas. [NV]
- **Que `prepare_model_for_kbit_training` haga OOM siempre:** depende de si PLE está en CPU y del orden; el cast fp32 está verificado en código, el OOM no. [NV]
- **Que lmms-eval/VLMEvalKit reproduzcan exactamente el scoring de Google:** los arneses están verificados; la equivalencia numérica no. [NV]
- **SAT/StepGame/SpartQA como benchmarks de transferencia desde HTML/CSS/SVG:** hipótesis, no evidencia publicada. [NV]
- **Coste en tokens:** estimaciones con 4k de salida; calibrar con 20 ítems y `usage.completion_tokens`. [NV]
