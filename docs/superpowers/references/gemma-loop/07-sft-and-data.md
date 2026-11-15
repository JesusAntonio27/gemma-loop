# 07 · SFT y datos (adaptación corta + adaptación a OpenCode)

Fecha: 2026-11-03. Ámbito: (1) adaptación corta del Gemma 4 convertido a loops con datos públicos de código/razonamiento y (2) adaptación al arnés OpenCode con trayectorias.
Todo lo marcado con URL se abrió y se leyó en esa fuente. Lo no abierto está en **Sin verificar**.

## Resumen

- Gemma 4 E2B/E4B es Apache 2.0, 35/42 capas, `hidden_size_per_layer_input=256` (PLE), y **comparte KV entre capas** (`num_kv_shared_layers` 20 y 18): esto último rompe QLoRA estándar si `use_cache=False` (bug cerrado en transformers, fix en PR #45312).
- El chat template de Gemma 4 **no tiene marcadores `{% generation %}`** y TRL no trae `gemma4_training.jinja`: `assistant_only_loss=True` lanza `ValueError`. Hay que enmascarar a mano (columna `labels`), usar prompt-completion por turno, o plantar un template propio con marcadores.
- Datasets permisivos listos para usar: `nvidia/OpenCodeReasoning` (CC BY 4.0, 735,255 muestras con trazas R1), `nvidia/OpenCodeInstruct` (CC BY 4.0, 5M, con unit tests ejecutados), `nvidia/OpenMathReasoning` (CC BY 4.0, decontaminado por diseño), `open-thoughts/OpenThoughts3-1.2M` (Apache 2.0), `open-r1/OpenR1-Math-220k` (Apache 2.0), `bigcode/self-oss-instruct-sc2-exec-filter-50k` (ODC-By).
- **Riesgo alto**: `OpenCodeReasoning` incluye problemas de CodeForces/AtCoder/LeetCode sin fecha → contamina LiveCodeBench v6; `OpenR1-Math-220k` y `NuminaMath-1.5` incluyen `amc_aime` (5,872 problemas) → contamina AIME.
- Receta de descontaminación con umbrales publicados: 13-gramas compartidos **o** Indel-LCS ≥ 75 % (OpenThoughts), y coseno ≥ 0.7 + juez LLM (OpenCodeReasoning).
- Para trayectorias de agente basta un dataset **pequeño y verificado**: SWE-Gym usó 491 trayectorias exitosas (~19 turnos, ~19k tokens); SWE-smith ~1,350 a temperatura 0. Mezcla con datos generales: AgentTuning usó η = 0.2 (20 % agente / 80 % general).
- LoRA `r=16, alpha=32` sobre `q/k/v/o/gate/up/down_proj`, `lr=1e-4`, 1–2 épocas, `max_length=4096`, packing `bfd`; QLoRA NF4 con `r=64, alpha=16` es el default del paper, sobredimensionado para 100–200 M tokens.
- Loss 13–15 al inicio en E2B/E4B **es normal** en estos modelos multimodales; si ves 100+, tienes mal la acumulación de gradientes o el enmascarado.
- Presupuesto: 150 M tokens ≈ 36.6k secuencias de 4096 ≈ 2.29k pasos con batch efectivo 16. Ese es el orden de magnitud a planificar, no epochs.

## Hechos verificados

### Modelo, licencia, arquitectura
- Licencia **Apache 2.0**; cinco tamaños E2B, E4B, 12B, 26B-A4B, 31B; contexto 128K en E2B/E4B; thinking configurable. https://ai.google.dev/gemma/docs/core/model_card_4
- E4B: 42 capas, 4.5B efectivos (8B con embeddings), sliding window 512, vocab 262K. E2B: 35 capas, 2.3B efectivos (5.1B con embeddings). https://ai.google.dev/gemma/docs/core/model_card_4
- Benchmarks oficiales E4B: MMLU-Pro 69.4, AIME 2026 sin tools 42.5, LiveCodeBench v6 52.0, Codeforces ELO 940, GPQA Diamond 58.6, Tau2 42.2. E2B: 60.0 / 37.5 / 44.0 / 633 / 43.4 / 24.5. https://ai.google.dev/gemma/docs/core/model_card_4
- Sampling recomendado: `temperature=1.0, top_p=0.95, top_k=64`. https://ai.google.dev/gemma/docs/core/model_card_4
- `config.json` E4B: `hidden_size=2560`, 8 heads, `num_key_value_heads=2`, `head_dim=256`, `global_head_dim=512`, `num_kv_shared_layers=18`, `hidden_size_per_layer_input=256`, `final_logit_softcapping=30.0`, `eos_token_id=[1,106]`. https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json
- `config.json` E2B: `hidden_size=1536`, `num_key_value_heads=1`, `num_kv_shared_layers=20`, resto igual. https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json
- `generation_config.json` E4B: `do_sample=true`, `eos_token_id=[1,106,50]`, `pad_token_id=0`. https://huggingface.co/google/gemma-4-E4B-it/raw/main/generation_config.json
- Instalación oficial del notebook de function calling: `transformers>=5.10.1`; clases `AutoProcessor` y `AutoModelForMultimodalLM`. https://ai.google.dev/gemma/docs/capabilities/function-calling
- **Trampa real y cerrada**: `use_cache=False` corrompía la atención de E2B/E4B por el KV compartido (logits basura, loss 10–15). Cerrado el 2026-04-09, fix en transformers PR #45312. https://github.com/huggingface/transformers/issues/45242
- Unsloth: E2B LoRA en 8–10 GB, E4B LoRA 17 GB, E4B QLoRA 10 GB; loss 13–15 normal en E2B/E4B. https://unsloth.ai/docs/models/gemma-4/train

### Chat template y function calling
- Tokens especiales declarados: `sot=<|turn>`, `eot=<turn|>`, `std=<|tool>`, `etd=<tool|>`, `stc=<|tool_call>`, `etc=<tool_call|>`, `str=<|tool_response>`, `etr=<tool_response|>`, `soc=<|channel>`, `eoc=<channel|>`, `escape=<|"|>`, `think=<|think|>`. https://huggingface.co/google/gemma-4-E4B-it/raw/main/tokenizer_config.json
- Declaración de herramienta en el turno `system`: `<|tool>declaration:NAME{description:<|"|>...<|"|>,parameters:{...}}<tool|>`. https://ai.google.dev/gemma/docs/capabilities/function-calling
- Llamada: `<|tool_call>call:NAME{arg:<|"|>valor<|"|>}<tool_call|>`; respuesta: `<|tool_response>response:NAME{...}<tool_response|>`. https://ai.google.dev/gemma/docs/capabilities/function-calling
- Thinking: se activa con `<|think|>` al inicio del system prompt y se emite como `<|channel>thought\n...<channel|>`. https://ai.google.dev/gemma/docs/core/model_card_4
- Regla multi-turno: el historial solo lleva la respuesta final; el thinking previo se omite **salvo en turnos con tool calls**, donde se conserva. https://ai.google.dev/gemma/docs/core/model_card_4
- El template canónico (publicado 2026-07-09) acepta `role: "tool"` estilo OpenAI, resuelve `tool_call_id → nombre` y renderiza `tool_calls[].function.arguments` **solo como mapping** (si llega string, lanza excepción). https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja
- El template aplica `strip_thinking()` al `content` de los mensajes de rol `model`: cualquier `<|channel>...<channel|>` dentro de `content` se **borra**. El razonamiento debe ir en la clave `reasoning`/`reasoning_content`. https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja

### TRL / SFTTrainer
- `SFTConfig(assistant_only_loss=True)` solo funciona si el chat template tiene `{% generation %}` / `{% endgeneration %}`; si no, TRL intenta cambiar a un "training template". https://huggingface.co/docs/trl/sft_trainer
- TRL reconoce la familia **Gemma4** y guarda copias de `gemma4.jinja` … `gemma4_v10.jinja`, pero **no existe** `gemma4_training.jinja` en `trl/chat_templates/`. https://api.github.com/repos/huggingface/trl/contents/trl/chat_templates
- `get_training_chat_template()` termina en `raise ValueError("...patching is not supported for this template...")` para cualquier template no listado; Gemma 4 no está en el mapeo. https://raw.githubusercontent.com/huggingface/trl/main/trl/chat_template_utils.py
- `assistant_only_loss` exige dataset conversacional (`messages`). https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/sft_trainer.py
- Prompt-completion: la loss va solo en `completion` por defecto (`completion_only_loss=None`); con `completion_only_loss=False` se entrena toda la secuencia. https://huggingface.co/docs/trl/sft_trainer
- `max_length` por defecto 1024; `truncation_mode` solo admite `"keep_start"`; `packing_strategy` por defecto `"bfd"` (trunca el desbordamiento). https://huggingface.co/docs/trl/sft_trainer
- `padding_free=True` solo con FlashAttention 2/3. https://huggingface.co/docs/trl/sft_trainer
- Dataset con tool calls: columna `messages` (con `tool_calls` y mensajes `role:"tool"`) + columna `tools` con JSON schemas; `Dataset.from_list(data, on_mixed_types="use_json")` o `datasets>=4.7`. https://huggingface.co/docs/trl/dataset_formats
- TRL puede usar una columna `labels` precalculada tal cual (`-100` para ignorar), o `assistant_masks`/`completion_mask`. https://huggingface.co/docs/trl/sft_trainer
- Para VLMs, `assistant_only_loss` requiere `transformers>=5.18.0`; para texto no aplica. https://huggingface.co/docs/trl/sft_trainer

### Datasets (tamaño y licencia exactos)
- `nvidia/OpenCodeReasoning`: **CC BY 4.0**, 735,255 muestras (567,850 `split_0` + 167,405 `split_1`) sobre 28,319 problemas, salidas de **DeepSeek-R1** con `<think>`; campos `input, output, solution, source, license, dataset, split, difficulty`. **Sin campo de fecha.** https://huggingface.co/datasets/nvidia/OpenCodeReasoning
- `nvidia/OpenCodeInstruct`: **CC BY 4.0**, 5,000,000 filas, 6.4 GB de descarga, 19.07 GB de dataset; campos `unit_tests`, `tests_execution_status`, `average_test_score` (fracción de tests pasados), `llm_judgement`, `generation_algorithm` ∈ {self-instruct, evol-instruct}. https://huggingface.co/datasets/nvidia/OpenCodeInstruct/raw/main/README.md
- `nvidia/OpenMathReasoning`: **CC BY 4.0**, 306K problemas de AoPS, 3.2M CoT, 1.7M TIR, 566K GenSelect, 193K problemas sin solución; 49.6 GB de descarga; el pipeline aplicó **benchmark decontamination** explícita. https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/main/README.md
- `open-thoughts/OpenThoughts3-1.2M`: **Apache 2.0**, 1.2M filas (850k math, 250k code, 100k science), 28.2 GB, trazas de **QwQ-32B**. https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M/raw/main/README.md
- `open-r1/OpenR1-Math-220k`: **Apache 2.0**, `default` 93,733 filas, `all` 225,129; trazas de DeepSeek-R1 sobre NuminaMath 1.5, generadas con tope de 16k tokens. https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md
- `AI-MO/NuminaMath-1.5`: **Apache 2.0**, 896,215 problemas; desglose por fuente con **`amc_aime`: 5,872**. https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md
- `HuggingFaceM4/WebSight`: **CC BY 4.0**, v0.2 = 1,922,671 pares screenshot↔HTML+Tailwind (144.9 GB), v0.1 = 822,987 HTML+CSS; exige declarar el uso del dataset al publicar el modelo. https://huggingface.co/datasets/HuggingFaceM4/WebSight/raw/main/README.md
- `bigcode/self-oss-instruct-sc2-exec-filter-50k`: **ODC-By**, 50,661 filas; `response` validada por ejecución, derivada de MultiPL-T. https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/raw/main/README.md
- `nvidia/nemotron-post-training-v3` está etiquetado como dataset de post-training de Dory, pero el repo está **gated** (HTTP 401). https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/README.md
- Dory 1.0 4B: bloques 8/20/8 capas, loops 1/2/4, 256K, **OpenMDW-1.1**; SFT propio + GRPO multi-entorno; post-training con destilación de trayectorias de profesores, "aggressively filtered" (repetition de n-gramas, chequeos estructurales de tool calls). https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/README.md

### Contaminación y descontaminación (valores exactos)
- LiveCodeBench es "contamination-free" porque anota cada problema con su **fecha de publicación** y permite evaluar solo sobre problemas posteriores al cutoff. Demostró caída de DeepSeek en problemas de LeetCode posteriores a sept-2023. https://livecodebench.github.io/
- OpenCodeReasoning hizo `exact-match deduplication` (28,904 preguntas únicas) y verificó contaminación con **similitud coseno ≥ 0.7** contra benchmarks + verificación con Llama-3.3-70B (procedimiento de Yang et al. 2023). https://arxiv.org/abs/2504.01943
- OpenThoughts3 descontamina con **dos** criterios y descarta si cualquiera dispara: (a) Indel similarity normalizada por LCS **≥ 75 %**, (b) **13-gramas** compartidos tras tokenizar con el tokenizer de Qwen2-7B. https://arxiv.org/abs/2506.04178
- `lm-sys/llm-decontaminator` cuantifica muestras parafraseadas; tabla real: 15.9 % de contaminación de HumanEval en un subconjunto de StarCoder-Data, 12.8 % en CodeAlpaca. https://raw.githubusercontent.com/lm-sys/llm-decontaminator/main/README.md
- NeMo Curator (v1.4.0) ofrece dedup exacta (MD5), fuzzy (MinHash+LSH, `char_ngrams=24, num_bands=20, minhashes_per_band=13`) y semántica (embeddings + clustering, `eps` como umbral). https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html

### Agentes: números de los papers
- SWE-Gym: 2,438 instancias con entorno ejecutable; **491 trayectorias exitosas** por rejection sampling (~**19 turnos**, ~**19k tokens** cada una); full FT, `lr=1e-4`, máx 5 épocas, batch 8, `max context 32768`; +19 % absoluto en SWE-bench Verified/Lite. https://arxiv.org/abs/2412.21139
- SWE-Gym, self-improvement **falló**: mezclar 868 trayectorias on-policy con 491 off-policy bajó el resolve de 15.3 % a 8.7 %. https://arxiv.org/abs/2412.21139
- SWE-smith: 50k instancias de 128 repos; ~1,003 + 349 trayectorias finales a temperatura 0; full FT `lr=5e-5`, máx 3 épocas, `max context 32768`; al muestrear con el alumno se trunca el historial a los **5 tool outputs más recientes**. https://arxiv.org/abs/2504.21798
- AgentTuning: AgentInstruct = **1,866** interacciones, 6 tareas, 5–35 turnos; mezcla con ShareGPT a **η = 0.2**; loss solo en la salida del modelo; `lr=5e-5` (7B/13B), batch 64, seq 4096, cosine, warmup 2 %. https://arxiv.org/abs/2310.12823
- AgentInstruct (Microsoft): 25M pares sintéticos generados desde documentos/código como semillas, con flujos agénticos; Mistral-7B post-entrenado → Orca-3. https://arxiv.org/abs/2407.03502

### LoRA / QLoRA
- QLoRA: NF4 + double quantization + paged optimizers; config del paper `r=64, alpha=16`, LoRA en **todas** las capas lineales, dropout 0.1 (≤13B) / 0.05 (33B+), `max grad norm 0.3`. https://arxiv.org/abs/2305.14314
- "LoRA Learns Less and Forgets Less": en código y matemáticas LoRA rinde por debajo de full FT con rangos bajos, pero **olvida menos** que weight decay/dropout y mantiene generaciones más diversas; full FT aprende perturbaciones de rango 10–100× mayor. https://arxiv.org/abs/2405.09673
- Unsloth Gemma 4: `r=8, lora_alpha=8` para texto; `r=32, lora_alpha=32, target_modules="all-linear"` para VLM; `finetune_vision_layers=False`, `finetune_language_layers=True`. https://unsloth.ai/docs/models/gemma-4/train
- Nombres reales de módulos en `modeling_gemma4.py`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; PLE: `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`, tabla `embed_tokens_per_layer`. Las capas KV-shared **no tienen** `k_proj`/`v_proj`. https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py

## Know-how

### 1. Elección de datos para la adaptación corta (Fase 1)
Mezcla por **tokens**, no por filas. Objetivo 100–200 M tokens; propuesta defendible:

| Bloque | % tokens | Fuente | Filtro obligatorio |
|---|---|---|---|
| Código con trazas | 40 % | `nvidia/OpenCodeReasoning` | descontaminar LCB v6; `len(output)` acotado |
| Código instruct + tests | 15 % | `nvidia/OpenCodeInstruct` | `average_test_score == 1.0` |
| Matemáticas | 25 % | `nvidia/OpenMathReasoning` (`cot`) | quitar `amc_aime` si evalúas AIME |
| Razonamiento general/código | 10 % | `OpenThoughts3-1.2M` (subset code) | 13-gramas |
| Chat general | 10 % | tu mezcla de chat (estilo ShareGPT) | — |

Carga y filtro (verificado contra los campos del card):

```python
from datasets import load_dataset
ocr = load_dataset("nvidia/OpenCodeReasoning", "split_0")["split_0"]
ocr = ocr.filter(lambda r: 800 <= len(r["output"]) <= 24_000 and r["source"] != "CodeForces")
oci = load_dataset("nvidia/OpenCodeInstruct", split="train")
oci = oci.filter(lambda r: float(r["average_test_score"] or 0) == 1.0)
```

Nota: si excluyes CodeForces del todo pierdes el 53 % de OCR (386,948 de 735,255). Alternativa mejor: **no excluir**, descontaminar contra LCB (paso 2) y dejar que el filtro decida.

Para el control **O-H** usa exactamente el mismo fichero de datos ya tokenizado (mismo orden, misma semilla). Guarda el `.jsonl` final con hash; es lo que hace reproducible la comparación.

### 2. Descontaminación (obligatoria antes de entrenar)
Conjunto de evaluación a proteger: LCB v6 (prompts), AIME 2025/2026, GPQA-D, MMLU-Pro, y **tus** prompts de OpenCode/diseño.

```python
# pip install datasketch rapidfuzz
import re, hashlib
from rapidfuzz.distance import Indel

def norm(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^\w\s]", "", s)

def ngrams(tokens, n=13):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def contaminated(train, evals, tok, n=13, indel_thr=75.0):
    tn = ngrams(tok(train))
    for e in evals:
        if tn & ngrams(tok(e), n):                       # 13-gramas
            return True
        if Indel.normalized_similarity(norm(train), norm(e))*100 >= indel_thr:
            return True
    return False
```

Tres niveles, en este orden (coste creciente):
1. **Exacto/normalizado** por hash del enunciado → descarta.
2. **13-gramas + Indel 75 %** (OpenThoughts) → descarta.
3. **Embedding coseno ≥ 0.7** + juez LLM para los supervivientes dudosos (OpenCodeReasoning) → descarta.

Advertencias específicas:
- `OpenCodeReasoning` **no tiene campo de fecha**: el filtro por fecha no es posible; hay que confiar en 2–3.
- AIME 2026 (30 problemas) es el más frágil: excluye `amc_aime` de NuminaMath/OpenR1-Math, y no uses AIME 2026 como few-shot en ningún sitio.
- GPQA y MMLU-Pro se contaminan sobre todo por **texto de examen** en datos web; el 13-grama los pilla, el coseno no siempre.
- Registra cuántas muestras caen por criterio. Ese número va al paper.
- Si descontaminas con el mismo tokenizer del modelo, documenta el tokenizer y la versión.

### 3. TRL SFTTrainer: configuración que funciona con Gemma 4
Versiones: `trl==1.17.1` (la de la documentación consultada) y `transformers>=5.10.1` (mínimo que pide Google para Gemma 4; incluye el fix de `use_cache`). Fija ambas en el entorno y en el paper.

`assistant_only_loss=True` **no funciona** en Gemma 4 (ver Hechos). Tres caminos válidos, de menos a más trabajo:

**(A) Prompt-completion por turno (recomendado para OpenCode).** Un ejemplo por turno del asistente; el prompt arrastra todo el historial previo, incluidas las respuestas de tools. La loss cae solo en el completion.

```python
{"prompt": messages[:i], "completion": [messages[i]], "tools": tool_schemas}
```

**(B) Pre-tokenizar y pasar `labels`.** TRL respeta la columna `labels` tal cual. Construye el texto con `apply_chat_template(..., return_assistant_tokens_mask=True)` y convierte el mask a `-100`.

**(C) Template propio con marcadores.** Copia `chat_template.jinja`, envuelve el bloque del turno `model` en `{% generation %}`/`{% endgeneration %}` y pásalo con `SFTConfig(chat_template_path="ruta.jinja")`. Con marcadores presentes, TRL no intenta parchear.

Config base (texto, E2B/E4B, 12 GB):

```python
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

args = SFTConfig(
    output_dir="runs/e4b-adapt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_length=4096,
    packing=True,
    packing_strategy="bfd",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=10,
    save_steps=200,
    save_total_limit=4,
    assistant_only_loss=False,   # Gemma 4: no hay generation markers
    seed=3407,
)
peft = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

- `packing=True` con `bfd` **trunca el desbordamiento**: si un ejemplo mide 9k y `max_length=4096`, pierdes la cola. Con trayectorias, trocea tú antes.
- `truncation_mode` solo permite `"keep_start"` → recorta por la izquierda a mano si quieres conservar el final.
- No actives `padding_free` salvo que tengas FA2/FA3 (en Windows/WSL es fricción extra).
- `gradient_checkpointing=True` está bien **solo** con el fix de transformers; verifica con un forward que `use_cache=False` da los mismos logits que `use_cache=True` (test de 5 líneas del issue #45242).

Sanidad antes de gastar: 30 pasos, loss debe bajar de ~13–15 a ~2–6 (rango normal E2B/E4B). Si se queda en 100+, revisa: (a) `use_cache`, (b) acumulación de gradientes, (c) que el mask no esté dejando fuera los tokens de respuesta.

Presupuesto: `pasos = tokens_totales / (batch_efectivo × max_length)`. 150 M / (16 × 4096) ≈ 2,289 pasos.

### 4. Formato de datos de tool calling para Gemma 4
Estructura mínima que el template renderiza bien (OpenAI-style):

```python
messages = [
  {"role": "system", "content": "You are OpenCode..."},
  {"role": "user", "content": "arregla el test que falla"},
  {"role": "assistant", "content": "", "tool_calls": [
      {"id": "c1", "type": "function",
       "function": {"name": "bash", "arguments": {"command": "pytest -q"}}}]},
  {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "1 failed, 4 passed"},
  {"role": "assistant", "content": "El fallo está en ..."},
]
```

Reglas duras del template:
- `arguments` debe ser **dict**; un string lanza excepción (`"...must be a JSON object (mapping), not a string"`).
- Cada `assistant` con `tool_calls` debe ir seguido de sus mensajes `role:"tool"`; si no, el render deja un `<|tool_response>` colgando.
- Para entrenar razonamiento, usa la clave `reasoning` (o `reasoning_content`), **nunca** lo metas en `content`: `strip_thinking()` lo elimina.
- `enable_thinking=True` añade `<|think|>` al system turn. Decide una política y mantenla en train y eval; Unsloth recomienda ≥75 % de ejemplos con razonamiento si quieres preservarla.
- Para tool calls con thinking, el template conserva el bloque de razonamiento (`preserve_thinking` + `tool_calls`) — es la excepción documentada a "no pensar en historial".

### 5. Trayectorias de agente (Fase 2)
Pipeline:
1. **Tareas**: conjunto propio, ejecutable, con tests. Nada de benchmarks públicos.
2. **Rollout** con el maestro (DeepSeek V4.1 Flash) en OpenCode de serie, `temperature` 0.2–0.7, tope de pasos (SWE-agent usó 75) y tope de coste.
3. **Filtrado por éxito**: solo pasan las que resuelven (tests verdes / juez aprueba). Rejection sampling puro.
4. **Filtros de calidad** (Dory): chequeo estructural (tool call sin definición → fuera), repetición patológica de n-gramas → fuera.
5. **Deduplicación** por tarea: máximo 2–4 trayectorias por tarea para no overfittear el enunciado.
6. **Mezcla**: η ≈ 0.2 de tokens de agente sobre tokens totales (AgentTuning), resto datos generales/código.

Volumen realista: **500–2,000 trayectorias**. SWE-Gym demostró que 491 bastan para +19 pts en su setting; con 25 USD no aspires a más de unos miles de turnos.

**Turno vs trayectoria completa**:
- Trayectoria completa: una fila, máscara solo en turnos `model`. Máxima coherencia, pero se pasa de `max_length` con facilidad.
- Por turno (prompt = historial, completion = turno): escala mejor, permite truncar historial sin romper la estructura, y es el único camino limpio con `assistant_only_loss` roto. **Recomendado aquí.**
- Nunca cortes entre un `tool_calls` y su `tool_response`; si trunca, corta por turnos completos desde el principio.
- Truncado de contexto: conserva system + tarea + últimos *k* turnos (SWE-smith usa 5 tool outputs); resume o elimina los outputs más viejos. Entrena con el mismo esquema de truncado que usarás en inferencia, o el modelo aprenderá una distribución que nunca verá.

**No hagas self-training on-policy** en la primera iteración: SWE-Gym empeoró (15.3 → 8.7 %). Destila del maestro y, si acaso, itera después con verifier.

### 6. Generación sintética de tareas
**Código con verificación por ejecución** (patrón OpenCodeInstruct): sembrar con funciones/problemas, generar instrucción + solución + **unit tests**, ejecutar, y conservar solo `average_test_score == 1.0`. Ventaja: el filtro es objetivo y barato. Riesgo: tests generados por el mismo LLM que la solución → tests tautológicos. Mitiga con (a) tests escritos por un modelo distinto, (b) mutación: el test debe **fallar** con una versión mutada de la solución (mutation testing).

**Front-end**: dos fuentes.
- Semillas: `HuggingFaceM4/WebSight` (screenshot↔código, CC BY 4.0; obliga a declarar el uso).
- Verificación: renderizar con navegador headless y comparar screenshot vs referencia con las métricas de Design2Code (similitud visual + block-match + texto), que es el benchmark real de esta tarea (484 páginas). https://arxiv.org/abs/2403.03163
- Para el juez creativo: guarda el HTML generado, renderiza a PNG y pasa **la captura**, no el código (evita que el juez premie código bonito que renderiza mal).

### 7. LoRA vs QLoRA en 12 GB
- **LoRA bf16**: `r=16–32`, `alpha=2r`, dropout 0.05, en las 7 proyecciones. Es lo que cabe con `max_length=4096` en E2B; en E4B, justo.
- **QLoRA NF4 + double quant**: necesario para E4B con contexto largo. Defaults del paper (`r=64, alpha=16, dropout 0.1, max_grad_norm 0.3`) son para 65B y 100k+ ejemplos; para 100–200 M tokens, `r=32` sobra.
- No pongas LoRA en `embed_tokens_per_layer` (tabla PLE: 262k × 256 × capas) ni en las norms: multiplica parámetros entrenables sin ganancia.
- Las capas KV-shared no tienen `k_proj`/`v_proj`; PEFT simplemente no encuentra el módulo en esas capas. Verifica con `model.named_modules()` cuántos adaptadores se crearon.
- **Olvido**: LoRA olvida menos que full FT (Biderman et al.), pero con 1–2 épocas sobre código puedes perder chat general. Mitiga con 10–20 % de datos generales y midiendo regresión en MMLU-Pro/GPQA antes/después.
- Si haces QLoRA, entrena y evalúa con la **misma** cuantización, y dilo en el paper.

## Trampas

1. **`assistant_only_loss=True` revienta en Gemma 4.** No hay `gemma4_training.jinja` y `get_training_chat_template()` lanza `ValueError`. No lo descubras a mitad de un run de nube.
2. **`use_cache=False` + KV sharing.** Estaba roto en E2B/E4B (logits basura, loss 10–15). Cerrado 2026-04-09 (PR #45312), pero si clonas un entorno viejo o un fork, vuelve. Test de humo obligatorio.
3. **Loss 13–15 no es un bug** en E2B/E4B; 100+ sí (acumulación de gradientes o masking).
4. **`strip_thinking()` borra el CoT que pongas en `content`.** Usa `reasoning`. Si no, entrenas sin trazas y no te enteras.
5. **`arguments` como string rompe el template.** Deserializa antes de pasarlo.
6. **`packing="bfd"` trunca el final del ejemplo.** Con trayectorias largas pierdes justo el turno final (el que resuelve).
7. **`truncation_mode` solo `keep_start`.** No hay opción de conservar el final.
8. **OpenCodeReasoning no tiene fechas.** No puedes hacer filtro temporal contra LCB; solo n-gramas/embedding.
9. **`amc_aime` en NuminaMath/OpenR1-Math.** Si evalúas AIME, contamina. Quítalo explícitamente.
10. **`the-stack-v2`, `OpenCodeReasoning-II`, `nemotron-post-training-v3` están gated** (401). No planifiques sobre ellos sin token aceptado.
11. **WebSight obliga a declarar el uso** del dataset al publicar el modelo (CC BY 4.0 + cláusula adicional).
12. **El split temporal de LCB es la única defensa real**; el resto de filtros son heurísticos. Reporta la fecha de corte de tus datos.
13. **No mezcles políticas de thinking** entre train y eval (con/sin `<|think|>`, con/sin `reasoning`). Cambia la distribución y hunde la comparación.
14. **No entrenes con los prompts de tu propio set de evaluación de OpenCode.** Son 20–50 prompts; se memorizan con 1 época.
15. **Windows nativo + bitsandbytes** es frágil; el propio diseño ya prevé WSL2. Si vas por WSL2, fija CUDA y compila `bnb` allí, no en Windows.

## Referencias

- https://ai.google.dev/gemma/docs/core/model_card_4 — model card oficial: licencia, arquitectura, benchmarks, sampling, reglas de thinking/historial.
- https://ai.google.dev/gemma/docs/capabilities/function-calling — tokens y flujo completo de function calling de Gemma 4; `transformers>=5.10.1`.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/tokenizer_config.json — tokens especiales, `response_schema`, `response_template`.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja — template canónico; `strip_thinking`, tool calls, generation prompt.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json — 42 capas, KV shared 18, PLE 256, softcap 30, eos.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json — 35 capas, KV shared 20, hidden 1536.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/generation_config.json — sampling y lista de eos.
- https://huggingface.co/docs/trl/sft_trainer — SFTTrainer/SFTConfig: assistant_only_loss, packing, max_length, truncation, labels.
- https://huggingface.co/docs/trl/chat_templates — familias soportadas y lista de training templates (Gemma4 reconocido, sin template de entrenamiento).
- https://huggingface.co/docs/trl/dataset_formats — formato conversacional con tool calls y columna `tools`.
- https://raw.githubusercontent.com/huggingface/trl/main/trl/chat_template_utils.py — `get_training_chat_template()`: mapeo y `ValueError` final.
- https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/sft_trainer.py — chequeo de `{% generation %}` y swap de template.
- https://api.github.com/repos/huggingface/trl/contents/trl/chat_templates — inventario real de templates (no hay `gemma4_training.jinja`).
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py — nombres de módulos, KV sharing, PLE.
- https://github.com/huggingface/transformers/issues/45242 — bug `use_cache=False`, reproducción, fix PR #45312.
- https://unsloth.ai/docs/models/gemma-4/train — VRAM por variante, `r`/`alpha`, `train_on_responses_only`, loss esperada.
- https://huggingface.co/datasets/nvidia/OpenCodeReasoning — CC BY 4.0, 735,255 muestras, campos, sin fecha.
- https://huggingface.co/datasets/nvidia/OpenCodeInstruct/raw/main/README.md — CC BY 4.0, 5M, unit tests y `average_test_score`.
- https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/main/README.md — CC BY 4.0, 3.2M CoT / 1.7M TIR, decontaminación explícita.
- https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M/raw/main/README.md — Apache 2.0, 1.2M, composición y pipeline.
- https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md — Apache 2.0, 93,733/225,129, trazas R1.
- https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md — Apache 2.0, 896,215, desglose con `amc_aime`.
- https://huggingface.co/datasets/HuggingFaceM4/WebSight/raw/main/README.md — CC BY 4.0, 1.92M pares screenshot↔código.
- https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/raw/main/README.md — ODC-By, 50,661 respuestas validadas por ejecución.
- https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/README.md — Dory: arquitectura, license OpenMDW-1.1, receta SFT+GRPO, filtros de post-training.
- https://arxiv.org/abs/2504.01943 — OpenCodeReasoning: dedup exacta, coseno 0.7, 3 épocas, lr 5e-5, seq 32768, packing.
- https://arxiv.org/abs/2506.04178 — OpenThoughts3: Indel 75 % + 13-gramas, hiperparámetros, packing.
- https://arxiv.org/abs/2412.21139 — SWE-Gym: 491 trayectorias, 19k tokens, hiperparámetros, fracaso del self-improvement.
- https://arxiv.org/abs/2504.21798 — SWE-smith: 50k instancias, RFT, `lr=5e-5`, truncado a 5 tool outputs.
- https://arxiv.org/abs/2310.12823 — AgentTuning: 1,866 trayectorias, η=0.2, hiperparámetros.
- https://arxiv.org/abs/2407.03502 — AgentInstruct: 25M pares sintéticos desde semillas.
- https://arxiv.org/abs/2403.07974 — LiveCodeBench: evaluación por fecha de publicación.
- https://livecodebench.github.io/ — ventanas de contaminación y evidencia empírica.
- https://arxiv.org/abs/2406.01574 — MMLU-Pro: 10 opciones, sensibilidad a prompt.
- https://arxiv.org/abs/2305.14314 — QLoRA: NF4, double quant, r=64/alpha=16.
- https://arxiv.org/abs/2405.09673 — LoRA aprende menos y olvida menos.
- https://raw.githubusercontent.com/lm-sys/llm-decontaminator/main/README.md — herramienta y tasas de contaminación reales.
- https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html — dedup exacta/fuzzy/semántica con parámetros.
- https://arxiv.org/abs/2403.03163 — Design2Code: 484 páginas, métricas automáticas de render.

## Sin verificar

- Contenido y licencia de `nvidia/nemotron-post-training-v3`: repo **gated**, raw devuelve 401. Solo sé que Dory lo lista como dataset de post-training.
- `nvidia/OpenCodeReasoning-II` y `bigcode/the-stack-v2`: 401 al abrir. No cito cifras.
- Términos de DeepSeek para entrenar con salidas de V4.1 Flash: no los abrí; sigue siendo puerta bloqueante del diseño (§5.2).
- Que el id 106 de `eos_token_id` sea exactamente `<turn|>`: es inferencia a partir de `eot_token` en `tokenizer_config.json`; no lo confirmé con el tokenizer cargado.
- `bitsandbytes` 4-bit en Windows nativo (sin WSL2): no verificado; el issue #45242 se reprodujo en T4/Linux.
- Cobertura de idiomas/parseo de `unit_tests` en OpenCodeInstruct fuera de Python: el card no lo especifica.
- Si TRL publicará `gemma4_training.jinja` (hoy no existe en `main`).
