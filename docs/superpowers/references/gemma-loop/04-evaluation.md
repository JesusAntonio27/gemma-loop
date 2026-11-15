# 04 · Evaluación reproducible contra las cifras oficiales de Gemma 4

Fecha: 2026-11-03. Cómo correr LiveCodeBench v6, MMLU-Pro, GPQA-Diamond y AIME 2026 sobre Gemma 4
E4B/E2B (original y variantes con loops), con ajustes declarados, estadística honesta y coste/tiempo.
Lo marcado **[NV]** es estimación o no verificado.

---

## Resumen
- Las cifras de E4B quedan confirmadas: MMLU-Pro **69.4**, GPQA-Diamond **58.6**, LiveCodeBench v6
  **52.0**, AIME 2026 sin herramientas **42.5** (E2B: 60.0 / 43.4 / 44.0 / 37.5).
- Google solo publica “modo thinking” y `AIME 2026 no tools`. **No** publica temperatura, top-p,
  max tokens, k ni plantilla de prompt para esos benchmarks.
- LCB `release_v6` = **2023-05 → 2025-04**, **1055** problemas. El subconjunto posterior al corte de
  preentreno de Gemma 4 (enero 2025) tiene **182** problemas (175 en `test6` + 7 en `test5`).
- El arnés de LCB solo trae runner **vLLM** para modelos abiertos; no hay runner HF. Un modelo con
  arquitectura propia (loops) debe generar aparte y evaluarse con `custom_evaluator.py`.
- MMLU-Pro: **12 032** test / **70** validation, 10 opciones, MIT. GPQA: **448** totales, **198** en
  Diamond, *gated*, CC BY 4.0.
- AIME 2026: **30** problemas, respuesta 0–999. `MathArena/aime_2026` es **CC BY-NC-SA** (solo evaluar);
  `math-ai/aime26` es **Apache 2.0**. lm-eval hace **una muestra por ítem**: para k>1, seeds o sampler propio.
- Con n=198 (GPQA) el IC95 de Wilson es **±6.8 pp**; con n=30 (AIME), **±16.7 pp**. La tolerancia de
  3 pp del plan solo es alcanzable en LCB/MMLU-Pro, no en GPQA ni AIME.
- Una pasada completa con k=8 ronda **44 M tokens**; a 300 tok/s son ~41 h de GPU. Submuestrear antes
  de gastar.

---

## Hechos verificados (con URL)
### Cifras oficiales (modo thinking)

| Benchmark | E4B | E2B | Fuente |
|---|---:|---:|---|
| MMLU-Pro | 69.4 % | 60.0 % | https://huggingface.co/google/gemma-4-E4B-it |
| AIME 2026 (no tools) | 42.5 % | 37.5 % | ídem |
| LiveCodeBench v6 | 52.0 % | 44.0 % | ídem |
| GPQA Diamond | 58.6 % | 43.4 % | ídem |
| Codeforces ELO | 940 | 633 | ídem (no reproducible sin envíos) |

- Pie de la Tabla 5: **“All models are in thinking mode unless explicitly stated”**; la fila es
  `AIME 2026 no tools`. https://arxiv.org/html/2607.02770v2
- 58.6 % de 198 = **116/198** exacto (58.586 %), consistente con una muestra por ítem. 42.5 % de 30
  **no** es múltiplo de 1/30: la cifra de AIME es un promedio de k≥2 muestras (p. ej. 51/120 con k=4).
  Es inferencia aritmética, no una afirmación de Google.
- El reporte no tiene sección de protocolo de evaluación, solo la tabla. https://arxiv.org/abs/2607.02770

### Ajustes publicados
- Muestreo estándar: `temperature=1.0`, `top_p=0.95`, `top_k=64`.
  https://ai.google.dev/gemma/docs/core/model_card_4 (§Best Practices)
- `generation_config.json` repite esos valores con `do_sample: true`.
  https://huggingface.co/google/gemma-4-E4B-it/raw/main/generation_config.json
- Thinking se activa con `enable_thinking=True`; `<|think|>` va al inicio del system prompt y el bloque
  se delimita con `<|channel>thought\n … <channel|>`. `processor.parse_response()` separa
  `thinking`/`content`/`tool_calls` y exige `skip_special_tokens=False`.
  https://ai.google.dev/gemma/docs/capabilities/thinking
- **No publicado**: k, max tokens, temperatura efectiva, top-p, plantilla, extracción de respuesta ni
  ventana de LCB.

### LiveCodeBench
- `release_v6`: **2023-05 → 2025-04**, **1055** problemas (v1…v5: 400/511/612/713/880).
  https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/README.md
- Defaults del runner: `n=10`, `temperature=0.2`, `top_p=0.95`, `max_tokens=2000`, `timeout=6` s.
  https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/parser.py
- `pass@1` con k=1 es `c/n` por problema (estimador insesgado), promediado sobre problemas.
  https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/evaluation/pass_k_utils.py
- `test6.jsonl` descargado: **175** filas, **2025-01-04 → 2025-04-06**, AtCoder 112 / LeetCode 63.
  `test5.jsonl` añade **7** problemas del 2025-01-04. Unión con fecha ≥2025-01-01: **182**; sin solape.
  `release_v6` = 880 + 175 = 1055.
  https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl
  https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl
- El repo no tiene `hf_runner.py`; solo `vllm_runner.py` y runners de API.
  https://api.github.com/repos/LiveCodeBench/LiveCodeBench/contents/lcb_runner/runner
- `custom_evaluator.py` acepta `[{"question_id": ..., "code_list": [...]}]`, exige exactamente
  `len(benchmark)` entradas y ordena por `question_id` internamente.
  https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/custom_evaluator.py

### MMLU-Pro y GPQA
- MMLU-Pro: `test` **12 032**, `validation` **70**, 14 categorías, hasta 10 opciones, MIT.
  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md
- En lm-eval: grupo `mmlu_pro`, 14 tareas, `num_fewshot: 5`, `max_gen_toks: 2048`,
  `do_sample: false`, regex `answer is \(?([ABCDEFGHIJ])\)?`.
  https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/mmlu_pro/_default_template_yaml
- GPQA: **448** preguntas; **Diamond = 198**; CC BY 4.0; *gated* (aceptar términos y `hf auth login`).
  https://huggingface.co/datasets/Idavidrein/gpqa · https://ar5iv.labs.arxiv.org/html/2311.12022 (§2)
- Tareas lm-eval: `gpqa_{main,diamond,extended}_{zeroshot,n_shot,generative_n_shot,cot_zeroshot,cot_n_shot}`.
  https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/README.md
- `cot_zeroshot` es `generate_until`, greedy, sin max tokens fijo, con filtros `strict-match`
  (“The answer is …”) y `flexible-extract` (regex de `(A)`–`(D)`).
  https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml
- lm-eval soporta `enable_thinking` y `think_end_token` como `model_args`, y backend
  `local-chat-completions` para servidores OpenAI-compatibles.
  https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/interface.md

### AIME 2026
- `MathArena/aime_2026`: **30** filas, campos `problem_idx` (int), `answer` (int), `problem` (LaTeX),
  split `train`, **CC BY-NC-SA 4.0**, sin juez (final-answer).
  https://huggingface.co/datasets/MathArena/aime_2026/raw/main/README.md
- `eval.yaml`: framework `math-arena`, `MathArena/aime_2026` split `train`.
  https://huggingface.co/datasets/MathArena/aime_2026/raw/main/eval.yaml
- MathArena usa `--n 4` por defecto y AIME no requiere judging; extrae el último `\boxed{}`/`\fbox{}`
  y compara la respuesta final.
  https://raw.githubusercontent.com/eth-sri/matharena/main/README.md
  https://raw.githubusercontent.com/eth-sri/matharena/main/src/matharena/grader.py
- Alternativa permisiva: `math-ai/aime26`, 30 problemas, `answer` string, Apache 2.0.
  https://huggingface.co/datasets/math-ai/aime26/raw/main/README.md

---

## Know-how
### 1. LiveCodeBench v6
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
uv venv --python 3.11 && . .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
```

**Ruta A — vLLM (modelo estándar, p. ej. baseline O).** Registra el modelo en
`lcb_runner/lm_styles.py` (entrada en `LanguageModelList` con `LMStyle`) y su plantilla en
`lcb_runner/prompts/code_generation.py::format_prompt_generation`. Sin ambas, `LanguageModelStore[...]`
da `KeyError` o `NotImplementedError`.

```bash
python -m lcb_runner.runner.main \
  --model google/gemma-4-E4B-it --local_model_path /ruta/checkpoint \
  --scenario codegeneration --release_version release_v6 \
  --n 10 --temperature 0.2 --top_p 0.95 --max_tokens 32768 \
  --evaluate --num_process_evaluate 4 --timeout 12 --use_cache
```

- `release_v6` fija la ventana; **no** uses `release_latest` (cambiará con v7).
- Subir `--max_tokens` es obligatorio: 2000 trunca el thinking y hunde el pass@1.
- `--timeout 12` reduce falsos negativos por tests lentos.

**Ruta B — generación propia + evaluación oficial.** Genera con transformers/servidor y escribe
`[{"question_id": "...", "code_list": ["...", "..."]}]`; después:

```bash
python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file outputs.json --release_version release_v6 \
  --scenario codegeneration --evaluate --num_process_evaluate 4
```

Debe haber una entrada por problema (1055 en v6) y `code_list` con las k muestras.

**Filtro por fecha.** Gemma 4 preentrena hasta enero 2025; 182 ítems de v6 son posteriores (175 de `test6` + 7 de `test5`):

```bash
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file output/.../model_eval_all.json --start_date 2025-01-01
```

- v6 completa: 1055 ítems (SE ≈ 1.5 pp). Post-corte: 182 (SE ≈ 3.7 pp; MDE k=1 ≈ 13.4 pp).
- La ventana de Google no se publica; “v6” implica el set completo 2023-05→2025-04.

### 2. MMLU-Pro y GPQA con lm-eval
```bash
pip install "lm_eval[api]"
```

Sirve el modelo (incluido el de loops) con una capa OpenAI-compatible y lanza:

```bash
lm_eval --model local-chat-completions \
  --model_args model=gemma-loop,base_url=http://localhost:8000/v1/chat/completions,think_end_token="<channel|>" \
  --tasks mmlu_pro --apply_chat_template --num_fewshot 0 \
  --gen_kwargs max_gen_toks=16384,temperature=1.0,top_p=0.95 \
  --batch_size 8 --seed 0 --output_path results/mmlu_pro

lm_eval --model local-chat-completions \
  --model_args model=gemma-loop,base_url=http://localhost:8000/v1/chat/completions,think_end_token="<channel|>" \
  --tasks gpqa_diamond_cot_zeroshot --apply_chat_template --num_fewshot 0 \
  --gen_kwargs max_gen_toks=16384,temperature=1.0,top_p=0.95 \
  --batch_size 8 --seed 0 --output_path results/gpqa_d
```

- El servidor debe aplicar la plantilla con `enable_thinking=True`; `think_end_token="<channel|>"`
  recorta el trace antes de la métrica. Sin él, el regex puede capturar números del razonamiento.
- `mmlu_pro` usa 5-shot por defecto: `--num_fewshot 0` para el modo 0-shot (Google no lo declara).
- GPQA: `gpqa_diamond_cot_zeroshot` es generativa; `gpqa_diamond_zeroshot` es loglikelihood y no sirve
  para un modelo de razonamiento.
- lm-eval hace una muestra por ítem. Para k>1, repite con `--seed` distintos y agrega por ítem.
- Fija el commit HF: MMLU-Pro `b189ec765aa7ed75c8acfea42df31fdae71f97be`;
  GPQA `83022cefff930aea54f654c0b282e74b9eeda5c6`.

### 3. AIME 2026
```bash
git clone https://github.com/eth-sri/matharena.git && cd matharena
uv run python scripts/run.py --comp aime/aime_2026 --models <config> --n 4
```

`configs/models/<config>.yaml`: `model`, `api: vllm`, `human_readable_id`, `temperature: 1.0`,
`top_p: 0.95`, `max_tokens: 32768`. Para el modelo con loops, usa tu servidor.

**Arnés propio:** prompt que exija `Put the final integer (0-999) alone inside \boxed{}`. Extrae el
**último** `\boxed{...}`, normaliza (`int`, sin `$`/comas/espacios) y compara con `answer`. Reporta
`avg@k` y `pass@k`; con 30 problemas, añade Wilson y MDE. `MathArena/aime_2026` es no comercial.

### 4. Qué copiar de Google
- Thinking activado en todas las filas; AIME “no tools”; `temperature=1.0`, `top_p=0.95`, `top_k=64`.
- No publicados: max tokens, k, plantilla y ventana de LCB. Elige y declara: ≥32k en AIME/LCB;
  AIME 4–8 muestras, GPQA/MMLU-Pro 1–4, LCB n=10; v6 completa + subconjunto post-2025.
- Si tu “original” no cae en el IC de Google, documenta la desviación de protocolo.

### 5. Estadística
```python
import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

lo, hi = proportion_confint(k_correct, n_items, method="wilson")

a = np.array(correct_A, float); b = np.array(correct_B, float)   # mismo orden de ítems
res = bootstrap((a, b), lambda x, y, axis=-1: (x - y).mean(axis=axis),
                paired=True, n_resamples=10000, method="BCA", rng=np.random.default_rng(0))
lo, hi = res.confidence_interval

rej, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="holm")
```

- Miller (2024) recomienda **diferencia pareada + SE por CLT** como primario; el bootstrap pareado
  es robustez, pero debe remuestrear **pares de ítems**, no cada modelo por separado.
  https://arxiv.org/html/2411.00640v1 (§4.2)
- Wilson: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- Bootstrap pareado BCa: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Holm: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

**MDE (Miller, ec. 10)** con α=0.05 bilateral, potencia 0.80, ω²=1/12, σ²=1/6:

| Set | n | k=1 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| AIME 2026 | 30 | 33.0 pp | 20.9 pp | 18.1 pp |
| GPQA-Diamond | 198 | 12.9 pp | 8.1 pp | 7.0 pp |
| LCB v6 post-corte | 182 | 13.4 pp | 8.5 pp | 7.3 pp |
| LCB v6 completa | 1055 | 5.6 pp | 3.5 pp | 3.0 pp |
| MMLU-Pro | 12032 | 1.6 pp | 1.0 pp | 0.9 pp |

Con 30 problemas de AIME **no se detecta** una mejora de 3 pp ni con k=8: declara “sin evidencia”.
IC95 de Wilson de las cifras oficiales: 42.5 % (n=30) → **26.7–60.0**; 58.6 % (n=198) → **51.6–65.2**;
52.0 % (n=1055) → **49.0–55.0**. La tolerancia de 3 pp es más estrecha que el IC de GPQA.

### 6. Coste y tiempo
**Tokens por problema [NV; calibrar con 20 ítems]:** AIME 8k (p90 20k), GPQA 1.5k (4k), LCB 3k (8k),
MMLU-Pro 1k (3k).

**Throughput en RTX 5070 12 GB [NV, medir].** E4B bf16 pesa **9.0 GB** y el KV a 32k **0.14 GB**
(Tabla 3); con 2.5 GB de escritorio no cabe: WSL2 headless, Q4_0 (**2.3 GB**) o QLoRA.
- `transformers.generate` single-stream N=1: ~15–35 tok/s bf16; ~30–60 tok/s 4-bit.
- N=2 casi parte el throughput a la mitad.
- vLLM batch 16–32: ~200–500 tok/s agregados N=1; ~100–250 N=2.

**Pasada completa (LCB k=10, resto k=8):** LCB 31.7 M + AIME 1.9 M + GPQA 2.4 M + MMLU-Pro 2000×4 8.0 M
= **~44 M tokens**. A 300 tok/s, ~41 h por variante; a 150 tok/s, ~81 h. En nube a 0.40–0.60 USD/h
(4090) son **16–49 USD por variante**: no cabe con 25 USD para O, O-H y L.

**Plan recomendado:** LCB fijo de 300 problemas estratificados × 5 (4.5 M), MMLU-Pro 2000 × 1,
GPQA 198 × 4, AIME 30 × 8 → **~10 M por variante**, 12–25 h locales o 5–12 USD en nube.
Antes de gastar: 20 ítems por benchmark, guarda `usage.completion_tokens` y `finish_reason`; si >5 %
termina en `length`, sube `max_new_tokens`.

### 7. BFCL (métrica secundaria de herramientas)
```bash
pip install bfcl-eval
export BFCL_PROJECT_ROOT=/ruta/bfcl
bfcl generate --model gemma-loop --test-category simple_python,parallel,multiple,multi_turn_base \
  --skip-server-setup --num-threads 4
bfcl evaluate --model gemma-loop --test-category simple_python,parallel,multiple,multi_turn_base
```

```bash
export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1
export REMOTE_OPENAI_API_KEY=dummy
export REMOTE_OPENAI_TOKENIZER_PATH=/ruta/tokenizer
```

- `--skip-server-setup` + `REMOTE_OPENAI_BASE_URL` apuntan a tu capa OpenAI-compatible.
- Hay que registrar el modelo en `bfcl_eval/constants/model_config.py`; si no, `MODEL_NAME` no se reconoce.
- Categorías baratas: `simple_python`, `parallel`, `multiple`, `irrelevance`, `multi_turn_base`.
  Evita `web_search` (SerpAPI) y `memory_*`.
- Formato nativo: `<|tool_call>call:nombre{...}<tool_call|>` y
  `<|tool_response>response:...<tool_response|>`; el template lo genera con `tools=[...]`.
  https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4

---

## Trampas
1. **`max_tokens=2000` de LCB** trunca el thinking: sube a 32k y verifica `finish_reason`.
2. **`release_latest` no es estable**: fija `release_v6`.
3. **LCB no tiene runner HF**: usa `custom_evaluator` y exactamente `len(benchmark)` entradas.
4. **El prompt `GenericBase` de LCB no usa chat template**: añade un `LMStyle` propio con
   `apply_chat_template(..., enable_thinking=True)`.
5. **Contaminación**: Gemma 4 vio hasta enero 2025; reporta aparte los 182 problemas post-corte.
6. **GPQA es gated**: sin aceptar términos y `hf auth login`, `datasets` falla con 401.
7. **GPQA `cot_zeroshot` es greedy y con regex estricto** (“The answer is …”), que el thinking de
   Gemma 4 no produce: usa `flexible-extract` o instruye el formato.
8. **GPQA baraja opciones** con `random.shuffle`: fija semilla y guarda el mapeo letra→contenido.
9. **MMLU-Pro cambió en 2026** (espacios iniciales, respuestas médicas): fija `b189ec7…`.
10. **lm-eval no promedia k**: una pasada = una muestra; repite con seeds o sampler propio.
11. **`enable_thinking` no basta**: sin `think_end_token="<channel|>"` el razonamiento contamina el
    exact-match.
12. **MMLU-Pro en lm-eval es 5-shot por defecto**, no 0-shot. Google no lo publica: elige y declara.
13. **AIME 42.5 % no es un conteo directo**: 30 problemas no dan ese porcentaje con una muestra.
    No compares tu `pass@1` de una muestra contra ese número.
14. **MathArena usa n=4** y `MathArena/aime_2026` es CC BY-NC-SA: tu `avg@1` no es comparable y el set no entra en entrenamiento.
15. **E4B no es un 4B**: 8B totales (4.5B efectivos), 9 GB bf16; en 12 GB con escritorio no cabe.
16. **Los loops no corren en vLLM/Ollama**: servidor propio con transformers; no mezcles backends.
17. **Timeouts de LCB** (`--timeout 6`) dan falsos negativos con CPU cargada.
18. **La tolerancia de 3 pp no es alcanzable en GPQA/AIME** con esos n; usa IC o bootstrap pareado.

---

## Referencias
- https://huggingface.co/google/gemma-4-E4B-it — model card: tabla de benchmarks, muestreo, Apache 2.0.
- https://arxiv.org/html/2607.02770v2 — Gemma 4 Technical Report v2; Tabla 5 y modo thinking.
- https://ai.google.dev/gemma/docs/capabilities/thinking — activación de thinking, tokens y `parse_response`.
- https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4 — formato nativo de tool calls.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja — plantilla con `enable_thinking`.
- https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/README.md — releases v1–v6, n=10, filtros de fecha.
- https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/parser.py — defaults exactos.
- https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/evaluation/pass_k_utils.py — estimador pass@k.
- https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/custom_evaluator.py — salidas custom.
- https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl — 175 filas, 2025-01→2025-04; con `test5`, 182 ítems ≥2025-01-01.
- https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md — 12 032 test / 70 validation.
- https://huggingface.co/datasets/Idavidrein/gpqa — GPQA gated, 448 preguntas, CC BY 4.0.
- https://ar5iv.labs.arxiv.org/html/2311.12022 — paper GPQA; Diamond = 198.
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/README.md — variantes GPQA.
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml — CoT y filtros.
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/mmlu_pro/_default_template_yaml — 5-shot y regex.
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/interface.md — `enable_thinking`, `think_end_token`.
- https://huggingface.co/datasets/MathArena/aime_2026/raw/main/README.md — 30 problemas, CC BY-NC-SA.
- https://huggingface.co/datasets/math-ai/aime26/raw/main/README.md — alternativa Apache 2.0.
- https://raw.githubusercontent.com/eth-sri/matharena/main/README.md — `--n 4` y competiciones.
- https://raw.githubusercontent.com/eth-sri/matharena/main/src/matharena/grader.py — extracción de `\boxed{}`.
- https://arxiv.org/abs/2411.00640 — Miller 2024, “Adding Error Bars to Evals”.
- https://arxiv.org/html/2411.00640v1 — SE pareado, potencia y MDE.
- https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html — Wilson.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html — bootstrap pareado/BCa.
- https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html — Holm.
- https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md — BFCL CLI y endpoint.
- https://docs.vllm.ai/en/latest/models/supported_models.html — `Gemma4ForCausalLM` y `Gemma4ForConditionalGeneration`.
- https://docs.vllm.ai/en/latest/features/lora.html — `--enable-lora`, `--max-lora-rank`.

---

## Sin verificar
- Temperatura, top-p, max tokens, k y plantilla exacta de Google por benchmark. **No publicados.**
- Ventana de LCB usada por Google; “v6” sugiere el set completo, pero no se afirma.
- Que la cifra AIME 42.5 use exactamente k=4 (la aritmética exige k múltiplo de 4; el valor no se publica).
- Que GPQA 58.6 sea 116/198 con una muestra greedy (encaja exacto, pero no se declara).
- Tokens por problema y throughput de E4B en RTX 5070: rangos de planificación, no medidos aquí.
- Precio/hora de 4090/5090 y versión mínima de vLLM con Gemma 4: variables; verificar el día de la corrida.
