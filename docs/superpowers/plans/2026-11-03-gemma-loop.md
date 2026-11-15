# Gemma-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir Gemma 4 E2B/E4B en un modelo con capas en loop (pesos compartidos, N = 1..4), medirlo con honestidad contra las cifras oficiales de Google y contra controles emparejados, y después adaptarlo al arnés de OpenCode, dejando un repo reproducible y un paper.

**Architecture:** Un paquete Python (`gemmaloop`) que (1) reprograma el bucle de capas de `Gemma4TextModel` con un "schedule" de índices que repite el bloque de capas consumidoras de KV, (2) entrena adaptadores LoRA idénticos para el control sin loops y para el modelo con loops, (3) genera y puntúa con los arneses oficiales (LiveCodeBench, MathArena, lm-eval) y con un juez de visión para diseño, y (4) captura sesiones de OpenCode para la Fase 2. Todo corre en Windows nativo con una RTX 5070 de 12 GB; la confirmación en E4B corre en una GPU alquilada.

**Tech Stack:** Python 3.12 (uv), torch 2.11.0+cu128, transformers 5.22.1, peft 0.21.2, trl 1.17.1, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.1.0, FastAPI + uvicorn, Playwright (Chromium), mitmproxy, pytest. LiveCodeBench en su propio venv (Python 3.11). OpenCode 1.19.3. API oficial de DeepSeek (`deepseek-flash`) para maestro y juez.

**Spec:** `docs/superpowers/specs/2026-11-03-gemma-loop-design.md` (aprobado el 2026-11-03). Know-how y fuentes verificadas: `docs/superpowers/references/gemma-loop/01..07-*.md`. Cada tarea indica qué referencia consultar; si algo del plan no se entiende, lee esa referencia y sus URLs antes de adivinar.

## Global Constraints

- Repo en `C:\Users\<user>\ML\GemmaLoop`. Artefactos pesados (modelos, checkpoints, datos, resultados crudos) en `C:\Users\<user>\ML\gemmaloop-artifacts` (variable `GEMMALOOP_ARTIFACTS`). **Nada pesado dentro de OneDrive.**
- Versiones fijas en `pyproject.toml`: `torch==2.11.0` (índice `https://download.pytorch.org/whl/cu128`), `transformers==5.22.1`, `peft==0.21.2`, `trl==1.17.1`, `bitsandbytes==0.50.2`, `accelerate==1.15.0`, `datasets==5.1.0`. No instalar vLLM ni Unsloth en este venv (degradan transformers).
- `attn_implementation="sdpa"` en entrenamiento y evaluación, siempre la misma. Cómputo en bf16, nunca fp16.
- Nunca poner `model.config.use_cache = False` a mano (issue transformers #45242).
- La tabla PLE `embed_tokens_per_layer` va en CPU y congelada en ambos modelos. **La torre de visión se carga y queda congelada (sin LoRA, sin cuantizar); la de audio no se carga** (Autor, 2026-11-03). LoRA solo en la pila de lenguaje.
- Modo razonamiento activado en entrenamiento y evaluación (`enable_thinking=True`); las trazas van en la clave `reasoning` del mensaje, nunca en `content` (el template las borra).
- Muestreo de evaluación: `temperature=1.0`, `top_p=0.95`, `top_k=64`, `eos_token_id=[1,106,50]` (de `generation_config.json`).
- Corte del loop: E2B prelude 0–14, loop 15–29, coda 30–34; E4B prelude 0–23, loop 24–35, coda 36–41. N ∈ {1,2,3,4}; **N primario = 2**. Probabilidades de N en entrenamiento: `(0.25, 0.40, 0.20, 0.15)`.
- Puente: loop 1 = identidad; loop t ≥ 2 recibe `w·h_prelude + (1−w)·h_anterior`, `w = sigmoid(a)`, `a` inicial = `log(0.99/0.01)`.
- LoRA idéntica para O-H y L: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`, `lr=1e-4` cosine, `warmup_ratio=0.03`, batch 1 × `gradient_accumulation_steps=16`, `optim="adamw_8bit"`, `max_length` según la regla de abajo, checkpoint cada 200 pasos con `save_only_model=False` y `save_total_limit=4`.
- `max_length`: 2048 en la Fase 1 (igual para O-H y L). En la Fase 2, el mayor valor de {4096, 6144, 8192} que entrene sin OOM en E2B según la calibración (Task 6), y nunca menor que el largo en tokens del prompt de sistema + herramientas de OpenCode + 1024.
- Misma mezcla de datos, mismo orden, misma semilla y mismos tokens de datos para O-H y L.
- Cifras oficiales de Google (modo razonamiento; model card y arXiv 2607.02770, Tabla 5): **E4B** MMLU-Pro 69.4, GPQA Diamond 58.6, LiveCodeBench v6 52.0, AIME 2026 42.5, Codeforces ELO 940. **E2B** 60.0 / 43.4 / 44.0 / 37.5 / 633. Codeforces ELO no se reproduce (requiere envíos en vivo). Visión: MMMU Pro **E4B 52.6 / E2B 44.2**, MATH-Vision **E4B 59.5 / E2B 52.4**.
- Hipótesis exploratoria del autor: ganancia en razonamiento espacial; benchmark según `references/gemma-loop/08-vision-and-spatial.md`, reportado como exploratorio.
- **Ningún dato de entrenamiento ni juicio sale de OpenCode Go ni de proveedores `opencode`/`opencode-go`** (Términos de OpenCode del 15-ago-2026). Maestro y juez: API oficial de DeepSeek, modelo `deepseek-flash`, solo con aprobación de gasto del autor.
- El derivado nunca se llama "Gemma"; se llama "Gemma-Loop" con la nota "based on Gemma 4 (Apache 2.0)".
- Descontaminación obligatoria antes de cualquier entrenamiento. AIME de evaluación: `math-ai/aime26` (Apache 2.0); `MathArena/aime_2026` (CC BY-NC-SA) nunca entra a datos.
- `prereg/prereg.yaml` se commitea y se etiqueta `prereg-v1` antes de cualquier corrida final; los criterios no se cambian después de ver resultados.
- Shell por defecto: PowerShell 5.1 en Windows; `uv run` para todo comando Python.

## Review Focus

1. **El PC se cuelga a mitad de una corrida** → el entrenamiento reanuda del último checkpoint y el evaluador reanuda sin duplicar ni perder ítems. Tests en Task 9 (`test_resume_skips_done_items`) y Task 13 (`test_resume_matches_uninterrupted`).
2. **Salida truncada por longitud en modo razonamiento** (`finish_reason="length"`) → cuenta como fallo, se marca y se reporta su tasa; si supera 5 % en la calibración, se sube `max_new_tokens`. Test en Task 9 (`test_length_finish_flagged`).
3. **El modelo pequeño emite una llamada a herramienta mal formada** (sin cierre `<tool_call|>` o argumentos rotos) → el servidor devuelve el texto como `content`, registra `invalid_tool_call=True` y no responde 500. Test en Task 5 (`test_malformed_tool_call_degrades_to_content`).
4. **El juez devuelve JSON vacío o inválido, o un render falla** → el par cuenta como `invalid`/`render_error`, nunca como victoria del otro. Test en Task 18 (`test_invalid_or_render_error_not_counted_as_win`).
5. **El escritorio ocupa VRAM** (~2.5 GB) → el cargador mide memoria libre antes de cargar y falla rápido con un mensaje claro. Test en Task 4 (`test_fails_fast_when_vram_low`).

---

## Notas de ejecución (leer antes de empezar)

- **Orden de ejecución (obligatorio):** 1–4 → 4b → 5–9 → 17 → 18 → 19 → 10 (`prereg-v1`) → 11 → 12 → 13 → 14 → 15 (etiqueta `phase1-code`) → 16 → 20 → 21 → 22 (congela las tareas de evaluación de agente y crea `prereg-v2`) → 23 → 24 → 25 → 26. La rúbrica del juez (Task 18) y los criterios creativos se congelan en `prereg-v1` (Task 10). **Regla del 2026-11-03 (autor aprobó acelerar):** los briefs creativos y el segundo juez (Task 19) se fijan en un anexo `prereg-v1.1` antes de cualquier generación creativa, porque no influyen en los benchmarks; así las baselines (Task 11) no esperan. Los briefs existen antes de la descontaminación (Task 12); si aún no existen, la Task 12 se re-ejecuta sobre el índice ampliado antes de entrenar. La Parte F necesita la aprobación del gasto de API de DeepSeek.
- **Quién hace qué:** las tareas de código las implementan agentes; las corridas largas (Tasks 11, 14, 16, 20, 23, 25) las lanza el ejecutor en segundo plano. **El autor** hace a mano: aceptar los términos de GPQA en Hugging Face y `hf auth login` (Task 8), crear la clave de la API de DeepSeek (Task 18), alquilar y pagar la GPU en la nube (Task 16), calificar pares a ciegas (Task 19) y publicar en GitHub/Hugging Face (Task 26).
- **Tests de GPU:** marcados `@pytest.mark.gpu` (se saltan sin CUDA) y `@pytest.mark.slow`. `uv run pytest -m "not gpu and not slow"` debe pasar en cualquier máquina.
- **Cuota de OpenCode Go:** sirve para que los agentes implementadores escriban este código y para revisiones. No se usa para generar datos, tareas de entrenamiento ni juicios.

---

## Parte 0 — Repo y entorno

### Task 1: Repo, entorno y rutas de artefactos

**Know-how:** `references/gemma-loop/03-training-stack.md` §(a)(b) y Know-how §1; trampa 10 (OneDrive).

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/gemmaloop/__init__.py`, `src/gemmaloop/paths.py`, `scripts/check_env.py`, `tests/test_paths.py`, `tests/conftest.py`
- Copy: `docs/superpowers/specs/2026-11-03-gemma-loop-design.md`, `docs/superpowers/plans/2026-11-03-gemma-loop.md`, `docs/superpowers/references/gemma-loop/*.md` desde `C:\Users\<user>\OneDrive\Desktop\claudeForEveryPurpose\docs\superpowers\` a `docs/` del repo.

**Interfaces:**
- Produces: `artifacts_dir() -> Path`, `run_dir(name: str) -> Path` (crea el directorio), `data_dir() -> Path`, `results_dir() -> Path`.

- [ ] **Step 1: Crear el repo y el venv**

```powershell
New-Item -ItemType Directory -Force C:\Users\<user>\ML\GemmaLoop, C:\Users\<user>\ML\gemmaloop-artifacts | Out-Null
Set-Location C:\Users\<user>\ML\GemmaLoop; git init
# escribir pyproject.toml (ver abajo) y después:
uv sync --python 3.12      # crea .venv, instala gemmaloop en modo editable y genera uv.lock
```

`pyproject.toml` declara el paquete `gemmaloop` (src layout), las versiones de Global Constraints, `torchvision==0.26.0`, `torchaudio==2.11.0`, `fastapi`, `uvicorn`, `pydantic>=2`, `playwright`, `rapidfuzz`, `datasketch`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `openai`, `pyyaml`, `mitmproxy`, `huggingface_hub[cli]`, `pytest`; markers `gpu` y `slow`; `[tool.pytest.ini_options] pythonpath = ["src", "."]`. torch, torchvision y torchaudio salen del índice CUDA con `[[tool.uv.index]]` (`name = "pytorch-cu128"`, `url = "https://download.pytorch.org/whl/cu128"`, `explicit = true`) y `[tool.uv.sources]` apuntando a ese índice: **sin esto uv instala el torch de CPU de PyPI**. `uv.lock` se commitea (entorno bloqueado). `.gitignore` excluye `.venv*`, `__pycache__`, `*.safetensors`, `artifacts/`, `.oc/`.

- [ ] **Step 2: Escribir los tests que fallan**

```python
# tests/test_paths.py
import pytest
from gemmaloop import paths

def test_artifacts_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", str(tmp_path / "art"))
    assert paths.artifacts_dir() == tmp_path / "art"
    assert paths.run_dir("x").is_dir()

def test_refuses_onedrive(monkeypatch):
    monkeypatch.setenv("GEMMALOOP_ARTIFACTS", r"C:\Users\<user>\OneDrive\Desktop\art")
    with pytest.raises(ValueError, match="OneDrive"):
        paths.artifacts_dir()

def test_default_outside_repo(monkeypatch):
    monkeypatch.delenv("GEMMALOOP_ARTIFACTS", raising=False)
    assert str(paths.artifacts_dir()).endswith("gemmaloop-artifacts")
```

- [ ] **Step 3: Correr y ver que fallan** — `uv run pytest tests/test_paths.py -v` → FAIL (`ModuleNotFoundError: gemmaloop`).

- [ ] **Step 4: Implementar `src/gemmaloop/paths.py`** — default `C:\Users\<user>\ML\gemmaloop-artifacts`; `ValueError` si la ruta contiene `OneDrive` (sin distinguir mayúsculas).

- [ ] **Step 5: Implementar `scripts/check_env.py`** — imprime `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_device_capability()`, `torch.cuda.mem_get_info()`, y hace un matmul `bnb.nn.Linear4bit` 64×64 en GPU. Sale con código 1 si la capacidad no es `(12, 0)` o falla bitsandbytes.

- [ ] **Step 6: Verificar** — `uv run pytest tests/test_paths.py -v` → 3 passed; `uv run python scripts/check_env.py` → imprime `2.11.0+cu128`, `(12, 0)`, `bnb 4bit OK`.

- [ ] **Step 7: Commit** — `git add -A; git commit -m "chore: repo scaffold, pinned env, artifact paths"`

---

## Parte A — Cirugía del modelo

### Task 2: Plan de capas desde el config

**Know-how:** `references/gemma-loop/01-gemma4-architecture.md` §(a)(b) y Know-how §3.

**Files:**
- Create: `src/gemmaloop/model/__init__.py`, `src/gemmaloop/model/layer_plan.py`, `tests/model/test_layer_plan.py`, `tests/fixtures/e2b_text_config.json`, `tests/fixtures/e4b_text_config.json` (el `text_config` de `https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json` y del de E4B, copiados tal cual)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class LayerPlan: num_layers: int; period: int; full_layers: tuple[int, ...]; first_shared: int; prelude: range; loop: range; coda: range`
  - `make_layer_plan(text_config: dict, coda_periods: int = 1) -> LayerPlan`
  - `schedule(plan: LayerPlan, n_loops: int) -> list[int]`

- [ ] **Step 1: Escribir los tests que fallan**

```python
import json, pytest
from gemmaloop.model.layer_plan import make_layer_plan, schedule

def cfg(name): return json.load(open(f"tests/fixtures/{name}_text_config.json"))

def test_e2b_plan():
    p = make_layer_plan(cfg("e2b"))
    assert (p.num_layers, p.period, p.first_shared) == (35, 5, 15)
    assert p.full_layers == (4, 9, 14, 19, 24, 29, 34)
    assert (p.prelude, p.loop, p.coda) == (range(0, 15), range(15, 30), range(30, 35))

def test_e4b_plan():
    p = make_layer_plan(cfg("e4b"))
    assert (p.num_layers, p.period, p.first_shared) == (42, 6, 24)
    assert p.full_layers == (5, 11, 17, 23, 29, 35, 41)
    assert (p.prelude, p.loop, p.coda) == (range(0, 24), range(24, 36), range(36, 42))

def test_schedule_lengths():
    assert len(schedule(make_layer_plan(cfg("e2b")), 2)) == 50
    assert len(schedule(make_layer_plan(cfg("e4b")), 2)) == 54
    p = make_layer_plan(cfg("e2b"))
    assert schedule(p, 1) == list(range(35))

def test_loop_must_be_whole_periods():
    c = cfg("e2b"); c["num_kv_shared_layers"] = 18   # first_shared=17, no alineado
    with pytest.raises(ValueError, match="period"):
        make_layer_plan(c)
```

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/model/test_layer_plan.py -v` → FAIL (import).

- [ ] **Step 3: Implementar `make_layer_plan` y `schedule`** — `period` = distancia entre índices `full_attention` consecutivos de `layer_types`; `first_shared = num_hidden_layers - num_kv_shared_layers`; prelude `[0, first_shared)`, coda = últimos `coda_periods*period`, loop = lo que queda; `ValueError("...period...")` si `first_shared` o el tamaño del loop no son múltiplos de `period`. `schedule = prelude + loop*n + coda`.

- [ ] **Step 4: Verificar** — mismo comando → 4 passed.

- [ ] **Step 5: Commit** — `git commit -am "feat(model): layer plan from Gemma 4 config"` (añade los archivos nuevos con `git add` antes).

### Task 3: Forward con loops

**Know-how:** `references/gemma-loop/01` Know-how §1, §2, §4 y Trampas 6–7; `references/gemma-loop/02` Know-how §A, §E y Trampas 4, 5, 9, 11. Código de referencia del bucle: `src/transformers/models/gemma4/modeling_gemma4.py` en el tag `v5.22.1` (bucle de `Gemma4TextModel.forward`).

**Files:**
- Create: `src/gemmaloop/model/looped.py`, `tests/model/test_looped.py`, `tests/model/tiny.py`

**Interfaces:**
- Consumes: `LayerPlan`, `schedule` (Task 2).
- Produces:
  - `@dataclass class LoopConfig: n_loops: int = 1; bridge: Literal["scalar","none","damped"] = "scalar"; bridge_init: float = 0.99; ple_mode: Literal["shared","loop_vector"] = "shared"`
  - `apply_loops(text_model: "Gemma4TextModel", plan: LayerPlan, cfg: LoopConfig) -> "Gemma4TextModel"` (parchea en sitio el bucle; registra `text_model.loop_bridge_logit: nn.Parameter` escalar y, si `ple_mode="loop_vector"`, `text_model.loop_ple: nn.Parameter` de forma `[4, hidden_size_per_layer_input]` inicializado en ceros)
  - `set_n_loops(text_model, n: int) -> None`
  - `bridge_input(h_prelude: Tensor, h_prev: Tensor, t: int, mode: str, w: Tensor) -> Tensor` (t empieza en 1)
  - `damped_update(x: Tensor, block_out: Tensor, n_loops: int) -> Tensor`
  - `save_loop_state(text_model, out_dir: Path) -> None` (escribe `loop_state.safetensors` con `loop_bridge_logit` y `loop_ple` si existe, y `loop_config.json` con el `LoopConfig` y los rangos del plan) y `load_loop_state(text_model, in_dir: Path) -> LoopConfig`. **El puente no es un parámetro LoRA: sin esto, el `w` entrenado se pierde al guardar el adaptador.**
  - `tests/model/tiny.py: tiny_text_model(seed: int = 0) -> Gemma4TextModel` (config: `num_hidden_layers=15`, `layer_types=(["sliding_attention"]*4+["full_attention"])*3`, `num_kv_shared_layers=10`, `hidden_size=64`, `intermediate_size=128`, `num_attention_heads=2`, `num_key_value_heads=1`, `head_dim=16`, `global_head_dim=32`, `hidden_size_per_layer_input=8`, `vocab_size=256`, `vocab_size_per_layer_input=256`, `sliding_window=16`, `use_double_wide_mlp=False`; ajustar campos obligatorios que pida `Gemma4TextConfig`)

- [ ] **Step 1: Escribir los tests que fallan**

```python
import copy, torch
from gemmaloop.model.layer_plan import make_layer_plan
from gemmaloop.model.looped import (LoopConfig, apply_loops, set_n_loops, bridge_input,
                                    damped_update, save_loop_state, load_loop_state)
from tests.model.tiny import tiny_text_model

IDS = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))

def _pair():
    base = tiny_text_model(); looped = copy.deepcopy(base)
    plan = make_layer_plan(base.config.to_dict())
    apply_loops(looped, plan, LoopConfig(n_loops=1)); return base, looped

def test_n1_identical_hidden_states():
    base, looped = _pair()
    with torch.no_grad():
        a = base(input_ids=IDS).last_hidden_state; b = looped(input_ids=IDS).last_hidden_state
    torch.testing.assert_close(a, b, atol=0, rtol=0)

def test_n2_changes_output():
    base, looped = _pair(); set_n_loops(looped, 2)
    with torch.no_grad():
        assert not torch.equal(base(input_ids=IDS).last_hidden_state, looped(input_ids=IDS).last_hidden_state)

def test_param_count_constant_across_n():
    _, looped = _pair(); n0 = sum(p.numel() for p in looped.parameters())
    for n in (2, 3, 4):
        set_n_loops(looped, n); assert sum(p.numel() for p in looped.parameters()) == n0

def test_bridge_gets_gradient():
    _, looped = _pair(); set_n_loops(looped, 2)
    looped(input_ids=IDS).last_hidden_state.sum().backward()
    assert looped.loop_bridge_logit.grad is not None

def test_bridge_formulas():
    hp, hprev, w = torch.ones(3), torch.zeros(3), torch.tensor(0.99)
    assert torch.equal(bridge_input(hp, hprev, 1, "scalar", w), hp)
    torch.testing.assert_close(bridge_input(hp, hprev, 2, "scalar", w), torch.full((3,), 0.99))
    assert torch.equal(bridge_input(hp, hprev, 2, "none", w), hprev)
    torch.testing.assert_close(damped_update(torch.zeros(3), torch.ones(3), 4), torch.full((3,), 0.25))
    torch.testing.assert_close(damped_update(torch.zeros(3), torch.ones(3), 1), torch.ones(3))

def test_loop_state_roundtrip(tmp_path):
    _, looped = _pair(); set_n_loops(looped, 2)
    with torch.no_grad(): looped.loop_bridge_logit.fill_(0.3)
    save_loop_state(looped, tmp_path)
    _, fresh = _pair(); load_loop_state(fresh, tmp_path); set_n_loops(fresh, 2)
    with torch.no_grad():
        torch.testing.assert_close(looped(input_ids=IDS).last_hidden_state,
                                   fresh(input_ids=IDS).last_hidden_state)

def test_cached_generation_matches_uncached():
    # greedy, 8 tokens, N=2: con caché y sin caché deben dar los mismos ids
    ...
```

`test_cached_generation_matches_uncached` envuelve el tiny model en `Gemma4ForCausalLM` (o equivalente) y compara `generate(do_sample=False, max_new_tokens=8, use_cache=True)` con un bucle manual sin caché (forward completo por paso, argmax).

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/model/test_looped.py -v` → FAIL (import).

- [ ] **Step 3: Implementar `looped.py`**

Reemplazar el bucle de capas de `Gemma4TextModel.forward` (versión 5.22.1) por uno que recorra `schedule(plan, n)` usando siempre el **índice fuente** `i` para `per_layer_inputs[:, :, i, :]`, `position_embeddings[layer_types[i]]` y `causal_mask_mapping[layer_types[i]]` (ver `references/01` Know-how §1). Entre pasadas del bloque aplicar el puente:

```python
h_p = hidden_states                       # salida del prelude
h = h_p
w = torch.sigmoid(self.loop_bridge_logit)
for t in range(1, n + 1):                 # pasadas del bloque
    x = h if mode == "damped" else bridge_input(h_p, h, t, mode, w)   # t == 1 devuelve h_p
    for i in plan.loop:
        x = self.layers[i](x, per_layer_input_for(i, t), ...)          # índice fuente i
    h = damped_update(h, x, n) if mode == "damped" else x
hidden_states = h                          # entra a la coda
```

Con `ple_mode="loop_vector"`, sumar `loop_ple[t-1]` a `per_layer_input` solo para `t >= 2`. Con `n=1` el código debe recorrer exactamente los módulos originales en orden (el test de identidad lo exige). Implementarlo como un método nuevo enlazado a la instancia (`types.MethodType`), sin copiar pesos.

- [ ] **Step 4: Verificar** — `uv run pytest tests/model/test_looped.py -v` → 7 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat(model): looped forward with identity at N=1 and scalar bridge"`

### Task 4: Carga del modelo en 12 GB

**Know-how:** `references/gemma-loop/01` Know-how §5 y Trampas 3, 8, 15; `references/gemma-loop/03` §(g), Know-how §3 y Trampas 1–2, 9.

**Files:**
- Create: `src/gemmaloop/model/load.py`, `tests/model/test_load.py`

**Interfaces:**
- Consumes: `make_layer_plan`, `apply_loops`, `LoopConfig`.
- Produces:
  - `MODEL_IDS = {"e2b": "google/gemma-4-E2B-it", "e4b": "google/gemma-4-E4B-it"}`
  - `@dataclass class ModelSpec: size: Literal["e2b","e4b"]; quant: Literal["bf16","nf4"]; adapters: tuple[str, ...] = (); loop: LoopConfig | None = None` (los adaptadores se aplican en orden; en bf16 todos menos el último se fusionan en los pesos; si un directorio de adaptador trae `loop_state.safetensors` y `loop_config.json`, se cargan con `load_loop_state`)
  - `load_model(spec: ModelSpec, min_free_gib: float = 3.0) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, LayerPlan]` (devuelve un modelo causal solo de texto, con PLE en CPU, `sdpa`, adaptadores cargados si `adapters`, loops aplicados si `loop`)
  - `free_vram_gib() -> float`

- [ ] **Step 1: Escribir los tests que fallan**

```python
import pytest, torch
from unittest import mock
from gemmaloop.model.load import load_model, ModelSpec

def test_fails_fast_when_vram_low():
    with mock.patch("gemmaloop.model.load.free_vram_gib", return_value=1.0):
        with pytest.raises(RuntimeError, match="VRAM"):
            load_model(ModelSpec("e2b", "bf16"), min_free_gib=3.0)

@pytest.mark.gpu
@pytest.mark.slow
def test_e2b_loads_text_only_with_ple_on_cpu():
    m, tok, plan = load_model(ModelSpec("e2b", "bf16"))
    ple = [p for n, p in m.named_parameters() if n.endswith("embed_tokens_per_layer.weight")][0]
    assert ple.device.type == "cpu" and not ple.requires_grad
    assert not any("vision_tower" in n or "audio_tower" in n for n, _ in m.named_parameters())
    assert torch.cuda.memory_allocated() / 2**30 < 9.0

@pytest.mark.gpu
@pytest.mark.slow
def test_use_cache_false_matches_true():          # regresión de transformers #45242
    m, tok, _ = load_model(ModelSpec("e2b", "bf16"))
    ids = tok("def add(a, b):", return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        a = m(ids, use_cache=True).logits; b = m(ids, use_cache=False).logits
    torch.testing.assert_close(a, b)

@pytest.mark.gpu
@pytest.mark.slow
def test_real_e2b_n1_identical_to_original():
    from gemmaloop.model.looped import LoopConfig
    base, tok, _ = load_model(ModelSpec("e2b", "bf16"))
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(base.device)
    with torch.no_grad(): ref = base(ids).logits
    del base; torch.cuda.empty_cache()
    looped, _, _ = load_model(ModelSpec("e2b", "bf16", loop=LoopConfig(n_loops=1)))
    with torch.no_grad(): out = looped(ids).logits
    torch.testing.assert_close(ref, out, atol=0, rtol=0)
```

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/model/test_load.py -v` → FAIL.

- [ ] **Step 3: Implementar `load_model`** — comprobar `free_vram_gib()` antes de cargar (`RuntimeError` con "VRAM" y la sugerencia de cerrar el navegador). Cargar solo la pila de lenguaje (renombrando el prefijo `model.language_model.*`, o cargando el multimodal y borrando `vision_tower`/`audio_tower` antes de mover a GPU; elige y documenta). `nf4`: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`. PLE a CPU y `requires_grad_(False)`; el forward debe mover el resultado del lookup a GPU. Tokenizer y `generation_config` del mismo repo.

- [ ] **Step 4: Verificar** — `uv run pytest tests/model/test_load.py -v` → 4 passed (en la RTX 5070, con el navegador cerrado).

- [ ] **Step 5: Commit** — `git commit -m "feat(model): text-only loader with PLE on CPU and VRAM guard"`

### Task 4b: Conservar la torre de visión (enmienda del 2026-11-03)

**Por qué:** el autor decidió que la visión es importante. La Task 4 ya está hecha y descarta las torres; esta tarea la enmienda para conservar la de visión (congelada) y seguir descartando la de audio.

**Know-how:** `references/gemma-loop/08-vision-and-spatial.md` §(a) (processor, mezcla de soft tokens de imagen, PLE de las posiciones de imagen, NF4 sin cuantizar la visión, LoRA solo en lenguaje); `references/gemma-loop/01` Know-how §5 y Trampas 8 y 15.

**Files:**
- Modify: `src/gemmaloop/model/load.py`, `tests/model/test_load.py`
- Create: `tests/fixtures/images/two_shapes.png` (PNG pequeño generado por el test o guardado: cuadrado rojo a la izquierda, círculo azul a la derecha)

**Interfaces:**
- Consumes: todo lo de la Task 4 y `apply_loops`.
- Produces (cambia el contrato de la Task 4; ninguna tarea posterior está implementada aún):
  - `load_model(spec: ModelSpec, min_free_gib: float = 3.0) -> tuple[model, processor, LayerPlan]`: `model` es el modelo multimodal **sin** la torre de audio (`Gemma4ForConditionalGeneration` o equivalente), con la torre de visión en GPU, en bf16, congelada y sin cuantizar incluso en `nf4`; `processor` es el `AutoProcessor` del mismo repo (`processor.tokenizer` es el tokenizer). La pila de lenguaje sigue con PLE en CPU y loops aplicados al `Gemma4TextModel` interno.
  - `language_lora_target_modules() -> str`: el regex de `target_modules` (PEFT usa `re.fullmatch`) que solo coincide con la pila de lenguaje, `r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"` (ajustar el prefijo si el envoltorio cambia los nombres; el test lo verifica). Lo consumen las Tasks 13 y 25.
  - `IMAGE_TOKEN_IDS = (255999, 258880, 258882)` (boi, image, eoi): las tareas de entrenamiento ponen sus labels en −100.

- [ ] **Step 1: Escribir los tests que fallan** (los 9 tests existentes deben seguir pasando, adaptados solo en el desempaquetado `model, processor, plan`):
  - `test_vision_kept_audio_dropped` (gpu, slow): hay parámetros con `vision` en el nombre, ninguno con `audio`; todos los de visión tienen `requires_grad=False`.
  - `test_image_forward_runs` (gpu, slow): un mensaje con la imagen de prueba y la pregunta "What colors are the shapes?" produce logits finitos y una generación de 16 tokens sin error.
  - `test_n1_identical_with_image` (gpu, slow): con la imagen, N=1 da logits idénticos (`atol=0, rtol=0`) al modelo sin loops.
  - `test_nf4_vision_not_quantized` (gpu, slow): en `nf4` las capas lineales de visión no son `Linear4bit` y la PLE sigue en CPU; `max_memory_allocated` se reporta.
  - `test_lora_targets_language_only` (CPU, tiny o mock): aplicar `LoraConfig(target_modules=language_lora_target_modules())` no crea adaptadores en módulos de visión.
- [ ] **Step 2: Correr y ver que fallan.**
- [ ] **Step 3: Implementar** según `references/08` §(a) y Know-how §1–§2: `AutoModelForImageTextToText` con `config.audio_config = None` antes de `from_pretrained` (la torre de audio no se materializa); en `nf4`, `llm_int8_skip_modules=[r".*vision_tower.*", r".*embed_vision.*", "lm_head"]` (incluir `lm_head` a mano, porque pasar la lista desactiva los saltos automáticos); visión en `eval()` con `requires_grad=False`. El `per_layer_inputs` que `Gemma4Model` calcula antes de insertar la imagen debe llegar intacto al bucle parcheado (nunca recalcularlo). Mantener todas las garantías de la Task 4: PLE fija en CPU incluso ante `model.to("cuda")`, nf4 sin pasar la PLE por la GPU, cadena de adaptadores solo en bf16, verificación de VRAM antes de cargar.
- [ ] **Step 4: Verificar** — `uv run pytest tests/model -v` (incluye GPU) y `uv run pytest -m "not gpu and not slow" -q`; reportar la VRAM medida en bf16 y nf4 con visión.
- [ ] **Step 5: Commit** — `git commit -m "feat(model): keep the frozen vision tower, drop audio"`

---

## Parte B — Generación, servidor y calibración

### Task 5: Generación por lotes y servidor compatible con OpenAI

**Know-how:** `references/gemma-loop/07` Hechos "Chat template y function calling" y Know-how §4; `references/gemma-loop/03` §(f); `references/gemma-loop/05` Know-how §1. Formato nativo: `https://ai.google.dev/gemma/docs/capabilities/function-calling`.

**Files:**
- Create: `src/gemmaloop/serve/__init__.py`, `src/gemmaloop/serve/generate.py`, `src/gemmaloop/serve/tool_parse.py`, `src/gemmaloop/serve/app.py`, `tests/serve/test_tool_parse.py`, `tests/serve/test_app.py`, `tests/fixtures/gemma_outputs/*.txt`

**Interfaces:**
- Consumes: `load_model`, `ModelSpec`, `set_n_loops`.
- Produces:
  - `@dataclass class Sampling: temperature: float = 1.0; top_p: float = 0.95; top_k: int = 64; max_new_tokens: int = 16384; enable_thinking: bool = True`
  - `@dataclass class GenOut: text: str; reasoning: str; tool_calls: list[dict]; completion_tokens: int; finish_reason: Literal["stop","length"]; invalid_tool_call: bool`
  - `generate_batch(model, processor, conversations: list[list[dict]], sampling: Sampling, tools: list[dict] | None = None, seed: int = 0, image_soft_tokens: int = 280) -> list[GenOut]` (los mensajes pueden traer partes `{"type": "image", "url": ...}`; el servidor acepta además partes OpenAI `image_url` con data URL y las convierte. `image_soft_tokens` ∈ {280, 1120}: con 1120 se fijan `processor.image_processor.max_soft_tokens` y `processor.image_seq_length`, ver `references/08` Know-how §4. Test adicional: `test_image_message_accepted` con un generador falso y una parte `image_url` en data URL)
  - `parse_gemma_output(raw: str) -> tuple[str, str, list[dict], bool]` → `(content, reasoning, tool_calls_openai, invalid)`
  - `openai_history_to_gemma(messages: list[dict]) -> list[dict]` (convierte `arguments` string→dict y `reasoning_content`→`reasoning`)
  - `create_app(spec: ModelSpec, max_batch: int = 8, wait_ms: int = 50, gen_fn: Callable | None = None, loader: Callable = load_model) -> FastAPI` con `POST /v1/chat/completions` (stream y no stream) y `GET /v1/models`; los tests inyectan un `gen_fn` y un `loader` falsos
  - CLI: `uv run python -m gemmaloop.serve.app --size e2b --quant bf16 --loops 2 --bridge scalar --adapter <dir> [--adapter <dir2>] --port 8000`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/serve/test_tool_parse.py
from gemmaloop.serve.tool_parse import parse_gemma_output, openai_history_to_gemma

def test_single_tool_call():
    raw = '<|tool_call>call:bash{command:<|"|>pytest -q<|"|>}<tool_call|>'
    content, reasoning, calls, invalid = parse_gemma_output(raw)
    assert calls[0]["function"]["name"] == "bash" and not invalid
    assert calls[0]["function"]["arguments"] == '{"command": "pytest -q"}'

def test_reasoning_split():
    raw = "<|channel>thought\nplan<channel|>Hola"
    assert parse_gemma_output(raw)[:2] == ("Hola", "plan")

def test_malformed_tool_call_degrades_to_content():
    raw = '<|tool_call>call:bash{command:<|"|>ls'
    content, _, calls, invalid = parse_gemma_output(raw)
    assert calls == [] and invalid and "bash" in content

def test_history_arguments_become_dict():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "read", "arguments": '{"filePath": "a.py"}'}}]}]
    assert openai_history_to_gemma(msgs)[0]["tool_calls"][0]["function"]["arguments"] == {"filePath": "a.py"}
```

```python
# tests/serve/test_app.py — con un generador falso inyectado
def test_non_stream_returns_tool_calls(fake_app_client): ...   # choices[0].message.tool_calls y usage.completion_tokens
def test_stream_ends_with_done(fake_app_client): ...           # último evento SSE == "data: [DONE]"
def test_reasoning_in_reasoning_content(fake_app_client): ...  # message.reasoning_content, no en content
```

Primero intentar `processor.parse_response()` de transformers 5.22.1 (exige `skip_special_tokens=False`); si cubre los casos, `parse_gemma_output` lo envuelve y solo añade la degradación de la llamada mal formada.

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/serve -v` → FAIL.

- [ ] **Step 3: Implementar** — `generate_batch`: `apply_chat_template(..., tools=tools, enable_thinking=sampling.enable_thinking, add_generation_prompt=True)`, padding a la izquierda, `model.generate(do_sample=True, ...)`, decodificar con `skip_special_tokens=False`, `finish_reason="length"` si se alcanzó `max_new_tokens`. `app.py`: cola + micro-batcher (hasta `max_batch` peticiones o `wait_ms`), mismo `generate_batch` (DRY con la Task 9).

- [ ] **Step 4: Verificar** — `uv run pytest tests/serve -v` → todos pasan; prueba manual en GPU: servidor E2B y `curl` a `/v1/chat/completions` devuelve texto.

- [ ] **Step 5: Commit** — `git commit -m "feat(serve): batched generation, Gemma tool-call parsing, OpenAI-compatible server"`

### Task 6: Calibración de rendimiento y presupuesto

**Know-how:** `references/gemma-loop/04` Know-how §6; `references/gemma-loop/07` Know-how §3 (fórmula de pasos).

**Files:**
- Create: `src/gemmaloop/budget.py`, `scripts/calibrate.py`, `tests/test_budget.py`

**Interfaces:**
- Consumes: `load_model`, `generate_batch`, adaptadores de benchmarks (Task 8; si aún no existen, usa 20 prompts de ejemplo por benchmark guardados en `tests/fixtures/calib_prompts.jsonl`).
- Produces:
  - `eval_hours(items: int, samples: int, tokens_per_item: float, tok_per_s: float) -> float`
  - `fit_eval_sizes(tok_per_s: float, tokens_per_item: dict[str, float], hours_per_variant: float, minimums: dict[str, int], caps: dict[str, int], priority: list[str]) -> dict[str, int]`: `tokens_per_item` ya incluye todas las muestras del ítem; asigna a cada benchmark su mínimo y reparte las horas sobrantes en el orden de `priority` hasta su `cap` (LCB 1055, MMLU-Pro 12032, GPQA 198, AIME 30); `ValueError` si los mínimos solos no caben en el presupuesto
  - `max_trainable_length(try_fn: Callable[[int], bool], candidates=(4096, 6144, 8192)) -> int` (el mayor candidato con el que 5 pasos de entrenamiento E2B no dan OOM; `ValueError` si ninguno)
  - `train_hours(tokens: int, train_tok_per_s: float) -> float`
  - Salida: `artifacts/calibration/calibration.json` con `{gen_tok_per_s: {e2b_n1_b8, e2b_n2_b8, e2b_n4_b8, e4b_n1_b8, e4b_n2_b8}, train_tok_per_s: {e2b_oh, e2b_l}, tokens_per_item: {lcb, aime, gpqa, mmlupro}: {p50, p90}, length_finish_rate: {...}, phase2_max_length: int}`

- [ ] **Step 1: Escribir los tests que fallan**

```python
import pytest
from gemmaloop.budget import eval_hours, fit_eval_sizes, train_hours, max_trainable_length

def test_eval_hours():
    assert abs(eval_hours(300, 5, 3000, 300.0) - 300*5*3000/300/3600) < 1e-9

def test_fit_fills_by_priority_up_to_caps():
    sizes = fit_eval_sizes(300.0, {"lcb": 3000, "gpqa": 1500}, 10.0,
                           {"lcb": 182, "gpqa": 198}, {"lcb": 1055, "gpqa": 198}, ["lcb", "gpqa"])
    assert sizes == {"lcb": 1055, "gpqa": 198}          # 3.46 M tokens ≤ 10.8 M

def test_fit_raises_when_minimums_exceed_budget():
    with pytest.raises(ValueError):
        fit_eval_sizes(300.0, {"lcb": 3000, "gpqa": 1500}, 0.5,
                       {"lcb": 182, "gpqa": 198}, {"lcb": 1055, "gpqa": 198}, ["lcb", "gpqa"])

def test_max_trainable_length():
    assert max_trainable_length(lambda n: n <= 6144) == 6144

def test_train_hours():
    assert abs(train_hours(40_000_000, 1500.0) - 40e6/1500/3600) < 1e-9
```

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/test_budget.py -v` → FAIL.

- [ ] **Step 3: Implementar `budget.py` y `scripts/calibrate.py`** — `calibrate.py` mide generación por lotes (batch 8, 256 tokens) y 30 pasos de entrenamiento LoRA de O-H y L sobre datos de prueba, y genera 20 respuestas por benchmark con `max_new_tokens=32768`, guardando tokens y `finish_reason`.

- [ ] **Step 4: Verificar** — tests en verde; `uv run python scripts/calibrate.py --sizes e2b` escribe `calibration.json` con todos los campos (sin `null`).

- [ ] **Step 5: Commit** — `git commit -m "feat: throughput and budget calibration"`

---

## Parte C — Evaluación y preregistro

### Task 7: Esquema de resultados y estadística

**Know-how:** `references/gemma-loop/04` Know-how §5 (Miller 2024, Wilson, Holm, MDE).

**Files:**
- Create: `src/gemmaloop/eval/__init__.py`, `src/gemmaloop/eval/results.py`, `src/gemmaloop/eval/stats.py`, `tests/eval/test_stats.py`, `tests/eval/test_results.py`

**Interfaces:**
- Produces:
  - `class ItemResult(BaseModel): item_id: str; sample_idx: int; correct: bool; completion_tokens: int; finish_reason: str`
  - `class RunResult(BaseModel): variant: str; size: str; n_loops: int; bench: str; prereg_tag: str; items: list[ItemResult]`; `RunResult.accuracy() -> float`; `RunResult.per_item_mean() -> dict[str, float]`; `save(path)`, `load(path)`
  - `wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]`
  - `paired_diff(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float, float]` → `(mean, se, lo, hi)` sobre los ítems comunes, SE por CLT
  - `paired_bootstrap(a: dict[str, float], b: dict[str, float], iters: int = 10000, seed: int = 0) -> tuple[float, float]`
  - `holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]`
  - `mde(n: int, k: int, alpha: float = 0.05, power: float = 0.8, omega2: float = 1/12, sigma2: float = 1/6) -> float` = `(z_{1−α/2} + z_{power}) * sqrt((omega2 + 2*sigma2/k) / n)`

- [ ] **Step 1: Escribir los tests que fallan**

```python
from gemmaloop.eval.stats import wilson, paired_diff, holm, mde

def test_wilson_matches_reference():                 # 58.6 % de 198 → 51.6–65.2 (ref. 04)
    lo, hi = wilson(116, 198); assert round(lo, 3) == 0.516 and round(hi, 3) == 0.652

def test_paired_diff_identical_is_zero():
    a = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    assert paired_diff(a, a)[:2] == (0.0, 0.0)

def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [True, False, False]

def test_mde_reference_table():                       # tabla de ref. 04
    assert abs(mde(30, 1) - 0.330) < 0.005
    assert abs(mde(198, 1) - 0.129) < 0.005
    assert abs(mde(1055, 1) - 0.056) < 0.005
    assert abs(mde(30, 4) - 0.209) < 0.005
```

`tests/eval/test_results.py`: ida y vuelta JSON, `accuracy()` y `per_item_mean()` con varias muestras por ítem.

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/eval -v` → FAIL.
- [ ] **Step 3: Implementar** con `scipy.stats.norm` para los cuantiles.
- [ ] **Step 4: Verificar** — `uv run pytest tests/eval -v` → todos pasan.
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): results schema and statistics"`

### Task 8: Adaptadores de benchmarks

**Know-how:** `references/gemma-loop/04` Hechos y Know-how §1–§4, Trampas 1–14.

**Files:**
- Create: `src/gemmaloop/eval/benchmarks/{__init__,lcb,aime,gpqa,mmlupro,mmmupro,mathvision,spatialeval,cvbench}.py`, `scripts/setup_lcb.ps1`, `tests/eval/test_benchmarks.py`
- Visión y espacial (enmienda del 2026-11-03; reglas en `references/gemma-loop/08` §(b), §(c) y Know-how §7–§8): `mmmupro` mide **Standard(10)** y **Vision** por separado (`MMMU/MMMU_Pro`, Apache 2.0), accuracy de letra A–J respetando que `<image_i>` se refiere a `image_i`; `mathvision` usa el test de 3.040 (`MathLLMs/MathVision`, MIT) con el exact-match oficial de `evaluation/evaluate.py` del repo MATH-V (sin juez); `spatialeval` hace VQA y TQA con las mismas preguntas (`MilaWang/SpatialEval`; sin licencia declarada: solo evaluación, nunca entrenamiento ni redistribución), exact-match contra `oracle_option`; `cvbench` usa la parte 2D (1.438, `nyu-visionx/CV-Bench`, Apache 2.0). Los ítems de visión llevan la imagen como parte `{"type": "image"}`. Tests: puntuación de MMMU-Pro con opciones barajadas, normalización de MATH-Vision en un caso de opción múltiple y uno abierto, y que SpatialEval VQA y TQA de un mismo ítem comparten `item_id` base.

**Interfaces:**
- Produces, para cada módulo `X`:
  - `@dataclass class Item: item_id: str; prompt: str; answer: str | None; meta: dict`
  - `load_items(limit: int | None = None, seed: int = 0) -> list[Item]` (submuestra estable por semilla)
  - `build_messages(item: Item) -> list[dict]`
  - `score(item: Item, completion: str) -> bool` (excepto LCB)
- `lcb.run_official_evaluator(outputs: list[dict], release: str = "release_v6") -> dict[str, list[bool]]` (subproceso con `.venv-lcb\Scripts\python -m lcb_runner.runner.custom_evaluator`; el archivo debe tener exactamente una entrada por problema del release, con `code_list` vacío para los no muestreados y esos se excluyen al puntuar)
- `lcb.extract_code(completion: str) -> str` (último bloque ```python)
- Fuentes fijadas: LCB `release_v6` (1055 problemas; subconjunto post-corte = fecha ≥ 2025-01-01, 182 problemas); AIME `math-ai/aime26`; GPQA `Idavidrein/gpqa` Diamond, revisión `83022cefff930aea54f654c0b282e74b9eeda5c6`; MMLU-Pro `TIGER-Lab/MMLU-Pro`, revisión `b189ec765aa7ed75c8acfea42df31fdae71f97be`, 0-shot.

- [ ] **Step 1: Paso manual del autor** — aceptar los términos de `Idavidrein/gpqa` en Hugging Face y ejecutar `uv run hf auth login`. Sin esto, GPQA falla con 401.

- [ ] **Step 2: Escribir los tests que fallan**

```python
from gemmaloop.eval.benchmarks import aime, gpqa, mmlupro, lcb

def test_aime_last_boxed():
    it = aime.Item("1", "p", "204", {})
    assert aime.score(it, r"... \boxed{12} ... final \boxed{204}")
    assert not aime.score(it, r"\boxed{2 04}x \boxed{205}")

def test_aime_strips_reasoning_first():
    it = aime.Item("1", "p", "7", {})
    assert not aime.score(it, "<|channel>thought\n\\boxed{7}<channel|>The answer is 8")

def test_mmlupro_regex():
    it = mmlupro.Item("1", "p", "C", {})
    assert mmlupro.score(it, "blah. The answer is (C)")

def test_gpqa_shuffle_stable_across_processes():      # hash() cambia entre procesos; esto no debe
    import os, subprocess, sys
    code = "from gemmaloop.eval.benchmarks import gpqa; print(gpqa.shuffle_map('q1', 0), gpqa.shuffle_map('q2', 0))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2")}
    assert len(outs) == 1 and outs.pop().strip()

def test_lcb_extract_code():
    assert lcb.extract_code("x\n```python\nprint(1)\n```\n```python\nprint(2)\n```") == "print(2)"
```

`gpqa.load_items` se marca `@pytest.mark.slow` (descarga).

- [ ] **Step 3: Correr y ver que fallan** — `uv run pytest tests/eval/test_benchmarks.py -v` → FAIL.
- [ ] **Step 4: Implementar** — todo `score` quita primero el bloque `<|channel>thought…<channel|>`. AIME: último `\boxed{}`/`\fbox{}`, normaliza a entero (regla del grader de MathArena). GPQA: `shuffle_map(item_id: str, seed: int) -> list[int]` (permutación de las 4 opciones con `random.Random(int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest(), 16))`; **nunca** `hash()`, que cambia entre procesos y rompería la comparación pareada entre variantes), guarda `letter_map`, extracción flexible `(A)`–`(D)` y "The answer is". MMLU-Pro: regex `answer is \(?([ABCDEFGHIJ])\)?`. `scripts/setup_lcb.ps1`: clona `https://github.com/LiveCodeBench/LiveCodeBench` en `C:\Users\<user>\ML\LiveCodeBench`, crea `.venv-lcb` con Python 3.11 e instala `-e .`.
- [ ] **Step 5: Verificar** — `uv run pytest tests/eval/test_benchmarks.py -v` → pasan; `scripts/setup_lcb.ps1` termina y `custom_evaluator --help` responde.
- [ ] **Step 6: Commit** — `git commit -m "feat(eval): LCB, AIME, GPQA, MMLU-Pro adapters with official scoring rules"`

### Task 9: Evaluador reanudable

**Know-how:** `references/gemma-loop/04` Know-how §6 y Trampa 1.

**Files:**
- Create: `scripts/run_eval.py`, `src/gemmaloop/eval/runner.py`, `tests/eval/test_runner.py`

**Interfaces:**
- Consumes: `generate_batch`, `Sampling`, `RunResult`, adaptadores de Task 8, `load_model`.
- Produces:
  - `run_bench(spec: ModelSpec, variant: str, bench: str, items: list[Item], samples: int, sampling: Sampling, out_path: Path, gen_fn=generate_batch, batch: int = 8) -> RunResult` (escribe cada lote al terminarlo en `out_path.with_suffix(".partial.jsonl")`; al reanudar salta los pares `(item_id, sample_idx)` ya hechos)
  - CLI: `uv run python scripts/run_eval.py --variant L --size e2b --quant bf16|nf4 --loops 2 --bridge scalar|none|damped --adapter <dir> [--adapter <dir2>] --bench lcb,aime,gpqa,mmlupro --prereg prereg/prereg.yaml` (R = `--bridge none` sin adaptador; R-damped = `--bridge damped` sin adaptador; O = sin `--loops`)

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_resume_skips_done_items(tmp_path): ...   # generador falso que cuenta llamadas; 1.ª corrida se interrumpe a mitad (excepción); 2.ª corrida solo genera lo que faltaba; total de ítems == items*samples, sin duplicados
def test_length_finish_flagged(tmp_path): ...     # GenOut(finish_reason="length") → ItemResult.correct=False y finish_reason="length"
def test_uses_prereg_sampling(tmp_path): ...      # el CLI lee temperature/top_p/top_k/max_new_tokens del prereg
```

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/eval/test_runner.py -v` → FAIL.
- [ ] **Step 3: Implementar** — LCB: acumula `code_list` por problema y llama a `run_official_evaluator` al final.
- [ ] **Step 4: Verificar** — tests en verde.
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): resumable benchmark runner"`

### Task 10: Preregistro y verificador de criterios

**Know-how:** spec §5.6 (S1–S5); `references/gemma-loop/04` Know-how §4–§5 (qué copiar de Google, MDE); `references/gemma-loop/02` Trampa 2 (no asumir monotonía).

**Files:**
- Create: `prereg/prereg.yaml`, `prereg/PREREGISTRATION.md`, `src/gemmaloop/prereg.py`, `scripts/check_criteria.py`, `tests/test_prereg.py`

**Interfaces:**
- Consumes: `calibration.json` (Task 6), `RunResult`, `paired_diff`, `wilson`, `holm`, `mde`.
- Produces:
  - `class Prereg(BaseModel)` con: `primary_n: int = 2`, `loop_probs: list[float]`, `sampling: Sampling`, `benches: dict[str, BenchSpec]` (`limit`, `samples`, `seed`), `google: dict[str, dict[str, float]]` (E2B y E4B, valores de Global Constraints), `harness_tolerance_pts: float = 3.0`, `train_tokens: dict[str, int]`, `seeds: dict[str, list[int]]`, `creative: CreativeSpec` (`n_briefs`, `briefs_sha256`, `judge_model="deepseek-flash"`, `judge_version="DeepSeek-V4.1-Flash-1017"`, `second_judge_model`, `second_judge_revision`, `human_pilot=40`, `human_main=120`), `criteria: Criteria`, `phase2: Phase2Spec | None` (se llena solo en `prereg-v2`: `eval_tasks_sha256`, `max_length`, `usd_cap`, umbral de S4)
  - `load_prereg(path: Path) -> Prereg` (valida que `loop_probs` sume 1 y que `primary_n` esté en 1..4)
  - `class CreativeSummary(BaseModel)`: `win_rate`, `lo`, `hi`, `flip_rate`, `n_valid`, `n_invalid`, `n_render_error`, `kappa: float | None`, `kappa_ci95: tuple[float, float] | None` (lo produce la Task 20)
  - `class AgentSummary(BaseModel)`: `valid_tool_call_rate`, `success_rate`, `success_ci95`, `mean_steps`, `tokens_per_solved_task`, `n_tasks` (lo produce la Task 25)
  - `check_criteria(prereg: Prereg, results: list[RunResult], creative: dict[str, CreativeSummary] | None = None, agent: dict[str, AgentSummary] | None = None) -> list[Verdict]` con `Verdict(name: str, passed: bool | None, detail: str)` para S1–S5 y la validez del arnés por benchmark. Si faltan datos creativos o de agente, S3/S4 salen `passed=None` con detalle "sin datos". Validez del arnés: `valid` si `|mine − google| ≤ tolerance` **o** Google cae dentro del Wilson 95 % de "mine"; si el semiancho de ese Wilson supera la tolerancia, el detalle dice "chequeo de baja potencia" (pasa en GPQA y AIME) y el paper lo reporta así.

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_rejects_bad_probs(tmp_path): ...            # loop_probs que no suman 1 → ValidationError
def test_s2_passes_when_ci_excludes_zero(): ...      # resultados sintéticos L(N=2) vs O-H con diferencia clara → S2 passed
def test_s2_fails_when_ci_includes_zero(): ...
def test_harness_validity_rule(): ...                # mine=50.5 vs google=52.0 → valid; mine=40 con n=1055 → invalid
def test_exploratory_n_uses_holm(): ...              # N=3,4 marcados exploratory y corregidos por Holm
def test_s3_s4_none_without_data(): ...              # sin creative/agent → S3 y S4 con passed=None y "sin datos"
def test_low_power_flag(): ...                       # AIME n=30 → detalle "chequeo de baja potencia"
```

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/test_prereg.py -v` → FAIL.
- [ ] **Step 3: Implementar** y rellenar `prereg.yaml` con: tamaños de evaluación de `fit_eval_sizes` (mínimos: LCB post-corte 182 + submuestra de release_v6, GPQA 198, AIME 30, MMLU-Pro 2000; visión: MMMU-Pro Standard(10) y Vision 300 cada uno a **1120** tokens de imagen para compararse con Google y MATH-Vision 300 a 1120, más el resto a 280 si la calibración lo permite; espacial exploratorio: SpatialEval VQA y TQA completos a 280 y CV-Bench 2D completo), muestras (LCB 4, AIME 8, GPQA 4, MMLU-Pro 1, visión y espacial 1 salvo que la calibración permita más; ver el costo en `references/08` §(b)), tokens de entrenamiento por brazo (defecto 40 M en E2B, ajustado con `train_hours`), semillas (E2B O-H y L: `[0,1,2]`; E4B: `[0]`). `PREREGISTRATION.md` explica cada valor en prosa y lista el efecto mínimo detectable de cada benchmark.
- [ ] **Step 4: Verificar** — tests en verde; `uv run python -c "from gemmaloop.prereg import load_prereg; load_prereg('prereg/prereg.yaml')"` sin error.
- [ ] **Step 5: Commit y etiqueta** — `git commit -m "docs: preregistration v1"; git tag prereg-v1`. **Desde aquí los campos de `prereg-v1` no se cambian; cualquier corrección exige una versión nueva explicada en el paper.** La Fase 2 añade `prereg-v2` (Task 22) llenando solo `phase2`, antes de recolectar datos.

### Task 11: Baselines de la Fase 0 (corrida)

**Files:**
- Create: `src/gemmaloop/report/three_column.py`, `tests/report/test_three_column.py`

**Interfaces:**
- Produces: `three_column_table(google: dict[str, float], mine_original: dict[str, RunResult], mine_modified: dict[str, RunResult] | None, tolerance: float) -> str` (markdown con columnas Google / Mi original / Mi modificado / Validez y Wilson 95 %).

- [ ] **Step 1: Test que falla** — `test_three_column_flags_invalid`: con una fila fuera de tolerancia y fuera del IC, la columna Validez dice `no validado`.
- [ ] **Step 2: Implementar y verificar** — `uv run pytest tests/report -v` → pasa.
- [ ] **Step 3: Corrida E2B original (O)** — `uv run python scripts/run_eval.py --variant O --size e2b --bench lcb,aime,gpqa,mmlupro --prereg prereg/prereg.yaml` (segundo plano; reanudable). Verificar que existen los 4 `RunResult` y que la tasa de `finish_reason="length"` es ≤ 5 %.
- [ ] **Step 4: Corrida E4B original (O)** — `--size e4b --quant nf4` (local si cabe; si no, se mueve a la Task 16). Declarar en el paper que E4B se evaluó en NF4.
- [ ] **Step 5: Tabla** — `uv run python -m gemmaloop.report.three_column --size e2b` y `--size e4b` escriben `artifacts/report/three_column_*.md`.
- [ ] **Step 6: Commit** — `git commit -m "feat(report): three-column table; phase 0 baselines"` (solo código y tablas pequeñas; los JSON crudos quedan en artefactos).

---

## Parte D — Fase 1: entrenamiento con loops

### Task 12: Datos de adaptación descontaminados

**Know-how:** `references/gemma-loop/07` Know-how §1–§2 y Trampas 4, 8–12.

**Files:**
- Create: `src/gemmaloop/data/{__init__,sources,decontam,mix}.py`, `scripts/prepare_heal_data.py`, `tests/data/test_decontam.py`, `tests/data/test_mix.py`

**Interfaces:**
- Produces:
  - `normalize(s: str) -> str`; `ngrams(tokens: list[str], n: int = 13) -> set[tuple[str, ...]]`
  - `class EvalIndex`: `EvalIndex.build(texts: list[str], n: int = 13) -> EvalIndex`; `EvalIndex.check(text: str) -> Literal["exact","ngram13","indel75"] | None`
  - `SOURCES: dict[str, SourceSpec]` con `hf_id`, `config`, `split`, `license`, `weight`: `ocr` = `nvidia/OpenCodeReasoning` split_0 (CC BY 4.0, 0.40), `oci` = `nvidia/OpenCodeInstruct` con `average_test_score == 1.0` (CC BY 4.0, 0.15), `omr` = `nvidia/OpenMathReasoning` cot sin fuentes `amc_aime` (CC BY 4.0, 0.25), `ot3` = `open-thoughts/OpenThoughts3-1.2M` subconjunto de código (Apache 2.0, 0.10), `chat` = un dataset de chat general con licencia Apache 2.0/MIT/CC BY elegido por el implementador y registrado aquí (0.10)
  - `to_messages(row: dict, source: str) -> list[dict]` (separa `<think>…</think>` a la clave `reasoning` del asistente)
  - `build_mix(target_tokens: int, tokenizer, eval_index: EvalIndex, seed: int = 0) -> tuple[list[dict], Manifest]`
  - `Manifest(BaseModel)`: `sha256`, `tokens_per_source`, `rows_per_source`, `dropped: dict[str, int]` (por criterio), `licenses`
  - Salida: `artifacts/data/heal/train.jsonl` + `manifest.json`

- [ ] **Step 1: Escribir los tests que fallan**

```python
from gemmaloop.data.decontam import EvalIndex
from gemmaloop.data.sources import to_messages

EVAL = ["Given an array of n integers, find the maximum sum of a contiguous subarray of length exactly k and print it."]

def test_exact_and_ngram_and_indel():
    idx = EvalIndex.build(EVAL)
    assert idx.check(EVAL[0]) == "exact"
    assert idx.check("Intro. " + EVAL[0][:-10] + " please.") in {"ngram13", "indel75"}
    assert idx.check("Write a haiku about autumn leaves falling slowly.") is None

def test_reasoning_goes_to_reasoning_key():
    msgs = to_messages({"input": "q", "output": "<think>plan</think>```python\nx=1\n```"}, "ocr")
    assert msgs[-1]["reasoning"] == "plan" and "<think>" not in msgs[-1]["content"]
```

`tests/data/test_mix.py`: con fuentes falsas, proporciones de tokens dentro de ±1 punto de los pesos; `manifest.sha256` estable para la misma semilla; las filas contaminadas se cuentan en `dropped`.

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/data -v` → FAIL.
- [ ] **Step 3: Implementar** — el índice de evaluación incluye los 1055 problemas de LCB `release_v6`, GPQA Diamond, todo MMLU-Pro test, AIME 2026, el texto de las preguntas de MMMU-Pro, MATH-Vision, SpatialEval (TQA) y CV-Bench, y los briefs creativos de evaluación (ya congelados: la Task 19 corre antes). Usa `rapidfuzz.distance.Indel.normalized_similarity ≥ 0.75` y 13-gramas con el tokenizer de Gemma 4 (documentar tokenizer y versión en el manifest). Es la receta de OpenThoughts3; el tercer nivel por embeddings de `references/07` §2 no se aplica y el paper lo declara. Para O-H-ext, `build_mix` se llama con `flops_matched_tokens(...)` tokens y la misma semilla: los primeros 40 M son idénticos a los de O-H/L y el resto son tokens únicos adicionales de la misma mezcla; el paper declara que O-H-ext ve más datos únicos, no solo más cómputo.
- [ ] **Step 4: Verificar** — tests en verde; `uv run python scripts/prepare_heal_data.py --tokens 40000000` produce `train.jsonl` y `manifest.json` con `dropped` no vacío.
- [ ] **Step 5: Commit** — `git commit -m "feat(data): decontaminated healing mixture with manifest"`

### Task 13: Entrenador LoRA con muestreo de loops

**Know-how:** `references/gemma-loop/07` Know-how §3 y Trampas 1–3, 6–7; `references/gemma-loop/03` Know-how §3 y Trampas 8, 12; `references/gemma-loop/02` Know-how §C–§D y Trampas 8–9.

**Files:**
- Create: `src/gemmaloop/train/{__init__,loop_sampler,budget,labels,trainer}.py`, `scripts/train.py`, `tests/train/test_loop_sampler.py`, `tests/train/test_budget.py`, `tests/train/test_labels.py`, `tests/train/test_trainer_tiny.py`

**Interfaces:**
- Consumes: `load_model`, `ModelSpec`, `LoopConfig`, `set_n_loops`, `make_layer_plan`, salida de Task 12.
- Produces:
  - `sample_n_loops(seed: int, micro_step: int, probs: tuple[float, ...] = (0.25, 0.40, 0.20, 0.15)) -> int` (determinista: semilla `int(hashlib.sha256(f"{seed}:{micro_step}".encode()).hexdigest(), 16)`; nunca `hash()`)
  - `expected_layer_counts(plan: LayerPlan, probs) -> list[float]` (E[N] para capas del loop, 1 para el resto)
  - `flops_matched_tokens(base_tokens: int, per_layer_flops: list[int], plan: LayerPlan, probs) -> int` = `round(base_tokens * Σ E[count_i]·f_i / Σ f_i)`
  - `per_layer_flops(text_config: dict) -> list[int]` (2 × parámetros lineales de cada capa: q/k/v/o, MLP con su ancho real, gate y proyección PLE; las capas consumidoras no tienen k/v; E2B usa MLP 12288 en las consumidoras)
  - `assistant_labels(tokenizer, messages: list[dict], max_length: int) -> dict` (input_ids + labels con −100 fuera de los turnos del asistente; **no** usar `assistant_only_loss`, roto en Gemma 4)
  - `class LoopSFTTrainer(SFTTrainer)`: sobrescribe `compute_loss` para llamar `set_n_loops(model, sample_n_loops(seed, self._micro_step))` antes del forward cuando el brazo tiene loops
  - `class TokenBudgetCallback(TrainerCallback)`: detiene al alcanzar `budget_tokens` (cuenta tokens no-padding)
  - CLI: `uv run python scripts/train.py --arm O|O-H|L|O-H-ext|L-plevec --size e2b|e4b --data <dir> --tokens <int> --seed <int> --out <run_dir> [--init-adapter <dir>] [--max-length <int>] [--resume <checkpoint_dir>]`
    - `--init-adapter`: carga ese adaptador (y su `loop_state`), lo fusiona en los pesos bf16 (solo E2B) y entrena un LoRA nuevo encima; el adaptador nuevo guarda `adapters_chain.json` con la lista ordenada de adaptadores previos para que `ModelSpec.adapters` reproduzca la cadena.
    - El brazo `O` entrena desde el original sin loops (en la Fase 2 es O+OC). Los brazos con loops llaman a `save_loop_state` en cada checkpoint.

- [ ] **Step 1: Escribir los tests que fallan**

```python
from collections import Counter
from gemmaloop.train.loop_sampler import sample_n_loops
from gemmaloop.train.budget import flops_matched_tokens

def test_sampler_deterministic_and_distribution():
    assert [sample_n_loops(0, s) for s in range(50)] == [sample_n_loops(0, s) for s in range(50)]
    c = Counter(sample_n_loops(0, s) for s in range(20000))
    for n, p in zip((1, 2, 3, 4), (0.25, 0.40, 0.20, 0.15)):
        assert abs(c[n] / 20000 - p) < 0.01

def test_flops_matched_uniform_e2b_structure():
    from gemmaloop.model.layer_plan import LayerPlan
    plan = LayerPlan(35, 5, (4,9,14,19,24,29,34), 15, range(0,15), range(15,30), range(30,35))
    assert flops_matched_tokens(1_000_000, [1]*35, plan, (0.25,0.40,0.20,0.15)) == 1_535_714

def test_flops_matched_nonuniform():
    from gemmaloop.model.layer_plan import LayerPlan
    plan = LayerPlan(3, 1, (0,1,2), 1, range(0,1), range(1,2), range(2,3))
    assert flops_matched_tokens(100, [1, 2, 1], plan, (0.0, 1.0, 0.0, 0.0)) == 150
```

`test_labels.py`: con el tokenizer de Gemma 4, solo los tokens del turno del asistente (incluido su `reasoning`) tienen label ≠ −100; los del sistema, usuario y respuestas de herramientas son −100. `test_trainer_tiny.py`: con `tiny_text_model`, 6 pasos en CPU bajan la pérdida; `test_resume_matches_uninterrupted`: 6 pasos seguidos vs 3 + reanudar desde checkpoint + 3 dan la misma pérdida final (tolerancia 1e-4).

- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/train -v` → FAIL.
- [ ] **Step 3: Implementar** con los hiperparámetros de Global Constraints. E2B en bf16 con PLE en CPU; E4B en NF4. **No usar `prepare_model_for_kbit_training`**: convierte a fp32 la PLE y la visión (9–11 GB) y no cabe en 12 GB; congelar a mano, `gradient_checkpointing_enable` y `enable_input_require_grads` (ver `references/08` Know-how §2). LoRA con `target_modules=language_lora_target_modules()` (Task 4b); el test verifica que ningún parámetro de visión queda entrenable. Labels de los tokens de imagen (`IMAGE_TOKEN_IDS`) en −100. `gradient_checkpointing_kwargs={"use_reentrant": False}`. Sin replay de imagen-texto por defecto (el autor acepta perder visión); brazo opcional `L-replay` con 1 % de tokens de COCO captions + TextVQA (CC BY 4.0) si el presupuesto lo permite, nunca ChartQA (GPL-3.0). Para O-H el sampler no se usa (N=1 fijo) y el puente no existe; para O-H-ext el presupuesto es `flops_matched_tokens(...)`. El brazo `L-plevec` es L con `ple_mode="loop_vector"` (ablación).
- [ ] **Step 4: Verificar** — `uv run pytest tests/train -v` → pasan; prueba de humo en GPU: `scripts/train.py --arm L --size e2b --tokens 200000` termina, la pérdida inicial está en 13–15 o menos y baja (13–15 es normal en E2B/E4B; 100+ indica error de máscara o acumulación).
- [ ] **Step 5: Commit** — `git commit -m "feat(train): LoRA trainer with sampled loop counts, token budget and resume"`

### Task 14: Corridas de la Fase 1 en E2B

**Files:**
- Create: `scripts/phase1_e2b.ps1` (lanza todo en orden y es reentrante: cada paso comprueba si su salida ya existe)

- [ ] **Step 1: Entrenar** — para `seed` en `[0,1,2]`: `--arm O-H` y `--arm L`; luego `--arm O-H-ext --seed 0` y `--arm L-plevec --seed 0`. Tokens según `prereg.yaml`. Verificar que cada corrida deja `trainer_state.json` con `global_step` final y que el manifest de datos es el mismo (mismo sha256) en todos los brazos.
- [ ] **Step 2: Evaluar** — matriz del preregistro: suite completa para O-H (3 semillas), L con N=2 (3 semillas) y L con N=1 (semilla 0; S1 lo exige en todas las evaluaciones); solo LCB para L con N=3,4, para R y R-damped con N=2,3,4 (sin entrenar), y para O-H-ext y L-plevec. `run_eval.py` reanuda solo.
- [ ] **Step 3: Verificar criterios** — `uv run python scripts/check_criteria.py --prereg prereg/prereg.yaml --size e2b` imprime S1–S5 y la validez del arnés; guardar la salida en `artifacts/report/criteria_e2b.md`.
- [ ] **Step 4: Commit** — `git commit -m "chore: phase 1 E2B runs script and criteria output"`

### Task 15: Figuras y cómputo

**Know-how:** `references/gemma-loop/02` Know-how §F y Trampa 2.

**Files:**
- Create: `src/gemmaloop/report/figures.py`, `tests/report/test_figures.py`

**Interfaces:**
- Consumes: `RunResult`, `per_layer_flops`, `schedule`.
- Produces: `flops_per_token(per_layer_flops: list[int], plan: LayerPlan, n_loops: int) -> int`; `plot_acc_vs_n(results, out_png)`; `plot_acc_vs_tokens(results, out_png)`; `plot_acc_vs_flops(results, per_layer_flops, plan, out_png)` (FLOPs por problema = tokens generados × `flops_per_token`, aclarando en el pie que excluye el costo de atención dependiente de la longitud).

- [ ] **Step 1: Tests que fallan** — `flops_per_token([1]*35, plan_e2b, 2) == 50`; las tres funciones crean PNG no vacíos con resultados sintéticos y barras de error Wilson.
- [ ] **Step 2: Implementar y verificar** — `uv run pytest tests/report/test_figures.py -v` → pasa.
- [ ] **Step 3: Commit y etiqueta** — `git commit -m "feat(report): accuracy vs loops, tokens and FLOPs figures"; git tag phase1-code` (la nube usa este código exacto)

### Task 16: Confirmación en E4B (nube)

**Know-how:** `references/gemma-loop/03` §(i) y Know-how §6.

**Files:**
- Create: `scripts/cloud/setup_pod.sh`, `scripts/cloud/run_e4b.sh`, `scripts/cloud/pull_results.ps1`, `src/gemmaloop/cloud.py`, `tests/test_cloud.py`

**Interfaces:**
- Produces: `estimate_cost(train_hours: float, eval_hours: float, usd_per_hour: float, overhead: float = 1.25) -> float`; `go_no_go(estimate: float, cap: float = 25.0) -> bool`.

- [ ] **Step 1: Tests que fallan** — `estimate_cost(8, 4, 0.40) == 6.0`; `go_no_go(26.0) is False`.
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Estimar** — con `calibration.json` (E4B) y el precio del día, calcular el costo de O-H y L (seed 0) más su evaluación. Si `go_no_go` es falso, recortar evaluación de E4B a LCB + GPQA y volver a estimar.
- [ ] **Step 4: Paso del autor** — el autor elige proveedor (Vast.ai o RunPod), crea la cuenta y alquila la GPU (RTX 5090 o A100). El ejecutor no paga ni crea cuentas.
- [ ] **Step 5: Correr** — `setup_pod.sh` instala el mismo entorno (`uv sync` con el `uv.lock` del repo) y el repo en la etiqueta `phase1-code`; `run_e4b.sh` entrena O-H y L (seed 0) y evalúa O-H y L con N=2 (y O si no se hizo en la Task 11), guardando checkpoints en volumen persistente; `pull_results.ps1` trae resultados y adaptadores. El autor destruye la instancia al terminar.
- [ ] **Step 6: Verificar** — `check_criteria.py --size e4b` produce `criteria_e4b.md`.
- [ ] **Step 7: Commit** — `git commit -m "feat(cloud): E4B confirmation scripts and cost guard"`

---

## Parte E — Eje creativo (puede ir en paralelo con la Parte D)

### Task 17: Renderizado reproducible

**Know-how:** `references/gemma-loop/06` §(e) y Know-how §1–§2.

**Files:**
- Create: `src/gemmaloop/judge/{__init__,render}.py`, `tests/judge/test_render.py`

**Interfaces:**
- Produces:
  - `extract_html(completion: str) -> str | None` (último bloque ```html, o documento `<html>…</html>`; quita el razonamiento antes)
  - `@dataclass class RenderResult: ok: bool; desktop_png: Path | None; mobile_png: Path | None; error: str | None`
  - `render_html(html: str, out_dir: Path, desktop=(1440, 900), mobile=(390, 844), timeout_ms: int = 15000) -> RenderResult` (Chromium headless, `offline=True`, `service_workers="block"`, aborta toda petición http/https, `reduced_motion="reduce"`, `animations="disabled"`, `document.fonts.ready`; `ok=False` si la captura sale en blanco)
  - `render_svg(svg: str, out_png: Path) -> RenderResult`

- [ ] **Step 1: Tests que fallan** — un HTML con `<div style="background:#ff0000;width:100vw;height:100vh">` da un píxel central `(255,0,0)`; un `<img src="https://example.com/x.png">` no genera ninguna petición de red (contar con `page.on("request")`) y aun así renderiza; HTML vacío → `ok=False`; `extract_html` toma el último bloque.
- [ ] **Step 2: Correr y ver que fallan** — `uv run playwright install chromium; uv run pytest tests/judge/test_render.py -v` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Verificar** — tests en verde.
- [ ] **Step 5: Commit** — `git commit -m "feat(judge): deterministic offline renderer"`

### Task 18: Cliente del juez y agregación

**Know-how:** `references/gemma-loop/06` §(a), Know-how §3–§5 y Trampas.

**Files:**
- Create: `src/gemmaloop/judge/{client,rubric,aggregate}.py`, `scripts/judge_vision_gate.py`, `tests/judge/test_client.py`, `tests/judge/test_aggregate.py`

**Interfaces:**
- Produces:
  - `rubric.SYSTEM_PROMPT: str` y `rubric.user_text(brief: str, required_sections: list[str], required_copy: list[str], forbidden: list[str]) -> str` (texto de la plantilla de `references/06` Know-how §4, fijado aquí antes de ver resultados)
  - `class Verdict(BaseModel)`: `render_ok: dict[str, bool]`, `criteria: dict[str, dict]`, `weighted: dict[str, float]`, `winner: Literal["A","B","tie"]`, `confidence: int`, `decisive_reason: str`, `flags: list[str]`
  - `judge_pair(brief_spec: dict, a: RenderResult, b: RenderResult, *, client, model: str = "deepseek-flash", max_tokens: int = 4096) -> JudgeRecord` (`JudgeRecord` guarda request, respuesta cruda, `usage`, `system_fingerprint`, `verdict: Verdict | None`, `status: Literal["ok","invalid","render_error"]`); si un render falló no llama al juez (`render_error`); si el JSON es vacío o inválido reintenta una vez y luego `invalid`
  - `aggregate(order1: str, order2: str, strict: bool = True) -> tuple[str, str]` (regla conservadora de MT-Bench; `strict=False` = regla secundaria de Arena 2026)
  - `win_rate(outcomes: list[str]) -> tuple[float, float, float]` (`(wins + 0.5·ties)/N` con IC bootstrap) y `flip_rate(pairs) -> float`
  - Versión del juez: la puerta de visión registra `response.model` y `system_fingerprint` (si existe) en `docs/gates.md`; `judge_pair` los compara en cada respuesta y aborta la corrida si cambian (el juez dejó de ser el preregistrado).
  - Cliente: `openai.OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])`, `response_format={"type":"json_object"}`, `reasoning_effort="high"`, `extra_body={"thinking":{"type":"enabled"}}`, imágenes como `image_url` con data URL en el mensaje `user`, rúbrica al principio (caché de prefijo).

- [ ] **Step 1: Paso del autor y puerta legal** — el autor crea una clave en la plataforma de DeepSeek y la guarda como variable de entorno de usuario `DEEPSEEK_API_KEY`. **Nunca** usar la clave ni el endpoint de OpenCode Go. El ejecutor vuelve a abrir `https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html` y registra en `docs/gates.md` la fecha y la cláusula §4.2(3) ("training other models (such as model distillation)"); si cambió, detiene las Partes E y F y avisa al autor.
- [ ] **Step 2: Tests que fallan**

```python
from gemmaloop.judge.aggregate import aggregate

def test_aggregate_truth_table():
    assert aggregate("X", "X") == ("X", "win")
    assert aggregate("X", "Y") == ("tie", "split")
    assert aggregate("X", "tie") == ("tie", "not_both")
    assert aggregate("X", "tie", strict=False) == ("X", "win_plus_tie")

def test_invalid_or_render_error_not_counted_as_win(fake_client): ...
# render de B falla → status "render_error", no se llama al cliente; JSON vacío dos veces → "invalid";
# win_rate excluye ambos y los reporta aparte

def test_empty_content_retried_once(fake_client): ...
```

- [ ] **Step 3: Correr y ver que fallan** — `uv run pytest tests/judge -v` → FAIL.
- [ ] **Step 4: Implementar.**
- [ ] **Step 5: Puerta de visión (gasto mínimo)** — `uv run python scripts/judge_vision_gate.py` envía un PNG con un cuadrado rojo a la izquierda y un círculo azul a la derecha y exige que el juez nombre ambos colores y posiciones. Si falla, detener la Parte E y avisar al autor.
- [ ] **Step 6: Verificar** — tests en verde y puerta superada.
- [ ] **Step 7: Commit** — `git commit -m "feat(judge): DeepSeek vision judge, conservative pair aggregation"`

### Task 19: Briefs creativos y calibración humana

**Know-how:** `references/gemma-loop/06` §(d) y Know-how §6; `references/gemma-loop/07` Know-how §6 (WebSight debe declararse si se usa).

**Files:**
- Create: `data/creative/briefs_eval.jsonl`, `data/creative/BRIEFS.md`, `scripts/make_briefs.py`, `scripts/human_rate.py`, `src/gemmaloop/judge/agreement.py`, `tests/judge/test_agreement.py`

**Interfaces:**
- Produces:
  - Formato de brief: `{"brief_id", "kind": "page"|"component"|"svg", "brief", "required_sections": [...], "required_copy": [...], "forbidden": [...]}`
  - `agreement(judge: list[str], human: list[str], iters: int = 10000, seed: int = 0) -> dict` → `{"p_o", "kappa", "kappa_ci95", "confusion"}`
  - `scripts/human_rate.py --pairs <jsonl> --out <jsonl>`: abre una página HTML local con las capturas de cada par en orden aleatorio y etiquetas "A"/"B" sin nombres de variante; guarda `A`/`B`/`tie`.

- [ ] **Step 1: Tests que fallan** — `agreement` con un ejemplo de kappa conocido (comparar con `sklearn.metrics.cohen_kappa_score`); la página de calificación no contiene los nombres de variante (`"O-H"`, `"L"`, `"loop"`) en el HTML generado.
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Generar y congelar 150 briefs de evaluación** — `make_briefs.py` los genera con la API oficial de DeepSeek (no OpenCode Go) o los escribe el autor; 60 % páginas, 25 % componentes, 15 % SVG. Guardar su sha256 en `prereg.yaml` (`creative.briefs_sha256`) y añadirlos al índice de descontaminación (Task 12).
- [ ] **Step 4: Elegir y fijar el segundo juez** — un modelo abierto de visión que no sea de la familia DeepSeek ni Gemma, que acepte imágenes según su model card, con licencia que permita evaluar, y que corra en ≤ 9 GB de VRAM (4 bits). Fijar `second_judge_model` y `second_judge_revision` en `prereg.yaml` antes de la Task 10, con una prueba de humo sobre 3 pares. Si ninguno funciona, poner `null` y declararlo como limitación en el paper.
- [ ] **Step 5: Commit** — `git commit -m "feat(judge): frozen creative briefs, blind human rating, agreement"`. **Esta tarea termina antes de la Task 10.**

### Task 20: Evaluación creativa de la Fase 1 (corrida)

**Files:**
- Modify: `scripts/run_eval.py` (añade `--bench creative`), `src/gemmaloop/judge/aggregate.py` (añade `summarize_creative`)
- Test: `tests/judge/test_summary.py`

**Interfaces:**
- Produces: `summarize_creative(records: list[JudgeRecord], human: list[str] | None = None) -> CreativeSummary` (tipo definido en la Task 10).

- [ ] **Step 1: Generar** — para cada brief, O-H y L con N=2 (seed 0, E2B; E4B si hubo nube) generan HTML/SVG en un intento con razonamiento activado (`run_eval.py --bench creative`, que reutiliza `generate_batch`).
- [ ] **Step 2: Renderizar** — `render_html`/`render_svg`; los fallos de render se cuentan y se reportan.
- [ ] **Step 3: Juzgar** — cada par en los dos órdenes con `judge_pair`; guardar todos los `JudgeRecord`. Repetir el 10 % para medir ruido.
- [ ] **Step 4: Calibración humana** — el autor califica a ciegas el piloto de 40 pares y luego 120; `agreement` con el juez.
- [ ] **Step 5: Segundo juez** — sobre el 15 % de los pares, un modelo de visión de otra familia (ni DeepSeek ni Gemma; por ejemplo un modelo abierto de visión que quepa en la GPU local). Si no hay uno viable, se declara como limitación.
- [ ] **Step 6: Resumir y verificar S3** — test `test_summarize_creative` (registros de ejemplo con un `invalid` y un `render_error` que no cuentan en el win-rate); luego `uv run python scripts/check_criteria.py --creative artifacts/report/creative_phase1.json` incluye el win-rate de L sobre O-H con su IC, la tasa de flips y el acuerdo humano.
- [ ] **Step 7: Commit** — `git commit -m "feat(judge): creative summary and phase 1 creative run"`

---

## Parte F — Fase 2: adaptación a OpenCode (requiere aprobación del gasto de API)

### Task 21: Proxy de captura y entorno aislado de OpenCode

**Know-how:** `references/gemma-loop/05` (todo; en especial Know-how §1, §2, §4 y Trampas).

**Files:**
- Create: `tools/capture/capture.py`, `src/gemmaloop/agent/{__init__,sse,opencode_env}.py`, `tests/agent/test_sse.py`, `tests/agent/test_opencode_env.py`, `tests/fixtures/sse/*.txt`

**Interfaces:**
- Produces:
  - `capture.py`: addon de mitmproxy (modo `reverse`) que escribe una línea JSON por petición con `ts`, `session` (de `x-opencode-session-id`), `request` (cuerpo JSON), `status`, `response_raw`; **nunca** escribe la cabecera `Authorization`.
  - `reassemble_sse(raw: str) -> dict` → `{"content", "reasoning_content", "tool_calls": [...], "usage"}` (une `delta.tool_calls` por `index`)
  - `@dataclass class ProviderCfg: provider_id: str; base_url: str; model_id: str; api_key_env: str; context: int; output: int`
  - `TEACHER = ProviderCfg("deepseek-capture", "http://127.0.0.1:4000", "deepseek-flash", "DEEPSEEK_API_KEY", 128000, 32000)` (el proxy reenvía a `https://api.deepseek.com`)
  - `local_provider(port: int, context: int = 32768, output: int = 8192) -> ProviderCfg` (`provider_id="gemma-loop"`)
  - `make_worker_env(worker_dir: Path, provider: ProviderCfg) -> dict[str, str]`: escribe `worker_dir/opencode.json` (`@ai-sdk/openai-compatible`, `tool_call: true`, `reasoning: true`, `limit.context`/`limit.output`, `model` y `small_model` = el mismo modelo, `default_agent: "build"`, `autoupdate: false`, permisos `external_directory` y `doom_loop` en `allow`) y devuelve variables con `OPENCODE_DB=<worker_dir>/oc.db`, `OPENCODE_DISABLE_CLAUDE_CODE=1`, `OPENCODE_DISABLE_EXTERNAL_SKILLS=1` y `HOME`/`USERPROFILE`/`XDG_*` dentro de `worker_dir`
  - `assert_tos_safe(provider: ProviderCfg) -> None`: `ValueError` si `provider_id` o `base_url` contienen `opencode` (proveedores hospedados de OpenCode)

- [ ] **Step 1: Tests que fallan** — reensamblado de SSE con `tool_calls` repartidos en varios chunks y `reasoning_content`; `opencode.json` generado contiene `limit.context` y `limit.output` y no menciona `opencode-go`; `assert_tos_safe(ProviderCfg("opencode-go", ...))` lanza `ValueError`; el entorno del worker no apunta a `C:\Users\<user>\.claude` ni a `C:\Users\<user>\.local\share\opencode`.
- [ ] **Step 2: Correr y ver que fallan** — `uv run pytest tests/agent -v` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Prueba de humo (gasto mínimo, solo con aprobación)** — `mitmdump --mode reverse:https://api.deepseek.com@4000 -s tools/capture/capture.py` y una tarea trivial con `opencode run --dir <tmp> --auto --format json --thinking --agent build --model deepseek-capture/deepseek-flash "Crea hello.py que imprima hola"`; verificar que `capture.jsonl` tiene el mensaje `system` completo y el arreglo `tools`, y que no contiene `Authorization`.
- [ ] **Step 5: Commit** — `git commit -m "feat(agent): capture proxy, SSE reassembly, isolated OpenCode workers, ToS guard"`

### Task 22: Conjunto de tareas de agente

**Know-how:** `references/gemma-loop/07` Know-how §5–§6.

**Files:**
- Create: `tasks/README.md`, `src/gemmaloop/agent/tasks.py`, `scripts/make_agent_tasks.py`, `tests/agent/test_tasks.py`, `tests/fixtures/tasks/{good,tautological,stub_passes}/`

**Interfaces:**
- Produces:
  - Estructura de tarea: `tasks/{train,eval}/<task_id>/{TASK.md, repo/, tests/, solution/, meta.json}`; `meta.json` = `{"task_id","kind":"code"|"design","split","source","created","sha256"}`
  - `validate_task(task_dir: Path) -> list[str]` (vacío = válida): la solución de referencia pasa los tests; el `repo/` sin solución los falla; una mutación de la solución (operador o constante cambiados) también los falla; las tareas de diseño traen brief con `required_sections`
  - `make_agent_tasks.py --n-train 600 --n-eval 60 --split-seed 0`: genera tareas con la API oficial de DeepSeek (70 % código Python con pytest, 30 % diseño front-end), las valida y descarta las inválidas; las de evaluación se congelan con su sha256 en `prereg.yaml` y se añaden al índice de descontaminación

- [ ] **Step 1: Tests que fallan** — `validate_task(good) == []`; `tautological` (tests que pasan con la solución mutada) → error "mutation"; `stub_passes` (tests que pasan sin solución) → error "stub".
- [ ] **Step 2: Implementar y verificar** — `uv run pytest tests/agent/test_tasks.py -v` → pasa.
- [ ] **Step 3: Generar en dos tiempos** — primero las 60 de evaluación; congelarlas (sha256) y añadir sus enunciados y tests al índice de descontaminación; después generar las de entrenamiento y descartar las que choquen con ese índice (13-gramas o Indel ≥ 75 %). Verificar ≥ 600 de entrenamiento válidas.
- [ ] **Step 4: `prereg-v2`** — llenar `phase2` en `prereg.yaml` (`eval_tasks_sha256`, `max_length` de la Fase 2 según `calibration.json`, `usd_cap` aprobado por el autor, umbral de S4); commitear y etiquetar `prereg-v2` antes de la Task 23. Los campos de `prereg-v1` no se tocan.
- [ ] **Step 5: Commit** — `git commit -m "feat(agent): validated agent task suite (train/eval)"`

### Task 23: Recolección de trayectorias del maestro (corrida)

**Files:**
- Create: `scripts/collect_trajectories.py`, `src/gemmaloop/agent/collect.py`, `tests/agent/test_collect.py`

**Interfaces:**
- Consumes: `make_worker_env`, `TEACHER`, `assert_tos_safe`, `validate_task`, `reassemble_sse`.
- Produces:
  - `run_task(task_dir: Path, worker_dir: Path, provider: ProviderCfg, timeout_s: int = 1800, opencode_cmd: list[str] | None = None) -> TaskRun` (`TaskRun`: `task_id`, `session_id`, `success`, `exit_code`, `steps`, `usage`, `cost_usd`)
  - `collect(task_dirs: list[Path], out_dir: Path, workers: int = 4, attempts: int = 2, usd_cap: float) -> CollectionReport` (se detiene al alcanzar `usd_cap`, calculado con el `usage` de las respuestas y los precios de `references/06`; reanudable)

- [ ] **Step 1: Tests que fallan** — con un comando `opencode` falso y un log de captura falso: las peticiones se asocian a su tarea por `session`; `success` refleja el resultado de pytest; el corte por `usd_cap` detiene la recolección; reanudar no repite tareas exitosas.
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Puerta** — confirmar con el autor el tope `usd_cap` aprobado y que `assert_tos_safe(TEACHER)` pasa.
- [ ] **Step 4: Correr** — `uv run python scripts/collect_trajectories.py --split train --workers 4 --attempts 2 --usd-cap <aprobado>` con el proxy activo (lanzar en segundo plano; horario off-peak de DeepSeek si es posible). Verificar ≥ 500 trayectorias exitosas o reportar cuántas se lograron con el tope.
- [ ] **Step 5: Commit** — `git commit -m "feat(agent): teacher trajectory collection with cost cap"`

### Task 24: Dataset SFT desde las trayectorias

**Know-how:** `references/gemma-loop/07` Know-how §4–§5 y Trampas 4–7, 13–14; `references/gemma-loop/05` Trampas (prompt con rutas y fecha).

**Files:**
- Create: `src/gemmaloop/data/build_agent_sft.py`, `scripts/build_agent_sft.py`, `tests/data/test_build_agent_sft.py`

**Interfaces:**
- Produces:
  - `normalize_system(system: str) -> str` (sustituye `Working directory`, `Workspace root folder` y `Today's date` del bloque `<env>` por marcadores fijos)
  - `to_turn_examples(request: dict, response: dict, keep_last_tool_outputs: int = 5) -> list[dict]` → filas `{"messages": [...], "tools": [...]}` donde el último mensaje es el turno del asistente a aprender; argumentos como dict; razonamiento en `reasoning`; nunca separa un `tool_calls` de sus mensajes `tool`
  - `quality_filter(example: dict, tool_names: set[str]) -> str | None` (`"undefined_tool"`, `"repetition"` o `None`)
  - `build_agent_mix(examples, heal_rows, eta: float = 0.2, max_per_task: int = 3, seed: int = 0) -> tuple[list[dict], Manifest]`

- [ ] **Step 1: Tests que fallan** — normalización elimina la ruta real y la fecha; truncado conserva sistema + tarea + últimos 5 resultados de herramienta sin romper pares; `quality_filter` detecta herramienta inexistente; la mezcla respeta η=0.2 en tokens (±1 punto) y el máximo de 3 trayectorias por tarea; las etiquetas solo cubren el último turno del asistente (reutiliza `assistant_labels`).
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Construir** — antes, medir el largo en tokens del prompt de sistema + herramientas capturado; si supera `phase2.max_length − 1024`, detener y avisar al autor (la Fase 2 tendría que ir a la nube). Las filas de adaptación reutilizadas (`heal_rows`) se vuelven a revisar contra el índice que ya incluye las tareas de evaluación de agente. Luego `uv run python scripts/build_agent_sft.py` → `artifacts/data/agent_mix/train.jsonl` + manifest.
- [ ] **Step 4: Commit** — `git commit -m "feat(data): per-turn agent SFT dataset with normalization and mixing"`

### Task 25: Entrenamiento y evaluación de la Fase 2 (corrida)

**Files:**
- Create: `scripts/run_agent_eval.py`, `src/gemmaloop/agent/metrics.py`, `tests/agent/test_metrics.py`

**Interfaces:**
- Consumes: `scripts/train.py`, `create_app`, `make_worker_env`, `local_provider`, `run_task`, `reassemble_sse`, `wilson`.
- Produces: `agent_metrics(captures: list[dict], runs: list[TaskRun]) -> dict` → `{"valid_tool_call_rate", "success_rate", "success_ci95", "mean_steps", "tokens_per_solved_task"}`.

- [ ] **Step 1: Tests que fallan** — métricas sobre capturas y corridas de ejemplo (incluye una llamada inválida y una tarea sin resolver).
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Entrenar** — en E2B, todos con `--data agent_mix`, el mismo presupuesto y `--max-length` = `phase2.max_length`: `O-H+OC` = `--arm O-H --init-adapter <O-H seed 0>`; `L+OC` = `--arm L --init-adapter <L seed 0>`; `O+OC` = `--arm O` sin `--init-adapter`. El paper declara que O+OC recibe la misma mezcla (agente + repetición de datos de adaptación) en una sola etapa.
- [ ] **Step 4: Evaluar en OpenCode** — para cada variante (y O-H, L sin adaptar como referencia): levantar el servidor local (Task 5), capturar con el proxy en modo `reverse:http://127.0.0.1:8000/v1@4000` y correr las 60 tareas de evaluación con `run_task(..., local_provider(4000))`. Añadir el maestro como línea de referencia (las mismas 60 tareas con `TEACHER`).
- [ ] **Step 5: Diseño dentro de OpenCode** — para las tareas de diseño de evaluación, renderizar el resultado final (`index.html` o el SVG que pida la tarea) con `render_html`/`render_svg` y juzgar `L+OC` contra `O-H+OC` por pares en los dos órdenes (Task 18); `summarize_creative` da el resumen creativo de la Fase 2, que se reporta junto a S4 como resultado secundario.
- [ ] **Step 6: Regresión** — repetir la suite oficial (Task 9) para `L+OC` y `O-H+OC` con N=2, usando la cadena de adaptadores (`--adapter <fase1> --adapter <fase2>`): es el criterio S5.
- [ ] **Step 7: Verificar S4** — `uv run python scripts/check_criteria.py --agent artifacts/report/agent_summaries.json` compara `L+OC` con `O-H+OC` usando `AgentSummary` (y reporta aparte la mejora de cada `+OC` sobre su versión sin adaptar).
- [ ] **Step 8 (opcional): Terminal-Bench** — si hay tiempo, Harbor con el agente `opencode` fijado en `--ak version=1.19.3` y el servidor local accesible como `host.docker.internal` (ver `references/05` §(e)).
- [ ] **Step 9: Commit** — `git commit -m "feat(agent): phase 2 training and OpenCode evaluation"`

---

## Parte G — Paper y publicación

### Task 26: Generador del reporte

**Know-how:** spec §6; `references/gemma-loop/01` §(d) (Apache 2.0, sin marca "Gemma"); `references/gemma-loop/06` §(b) (DeepSeek §3.1: declarar salidas generadas por IA); `references/gemma-loop/07` Trampa 11 (WebSight).

**Files:**
- Create: `scripts/build_report.py`, `scripts/export_release.py`, `src/gemmaloop/release/{__init__,export,modeling_gemma_loop}.py`, `report/templates/paper.md.j2`, `report/templates/model_card.md.j2`, `tests/report/test_build_report.py`, `tests/release/test_export.py`

**Interfaces:**
- Consumes: todos los `RunResult`, `JudgeRecord`, `CreativeSummary`, `AgentSummary`, manifests, `prereg.yaml`, figuras, `ModelSpec`, `load_model`.
- Produces:
  - `build_report(artifacts: Path, out_dir: Path) -> Path` que escribe `out_dir/paper.md` (en inglés), `out_dir/figures/*.png`, `out_dir/gallery/*.png` y `out_dir/MODEL_CARD.md`.
  - `export_merged(spec: ModelSpec, out_dir: Path) -> Path`: fusiona la cadena de adaptadores en pesos bf16 (E2B; E4B si hubo nube), copia `loop_state` y escribe `config.json` con `loop_prelude`, `loop_block`, `loop_coda`, `n_recurrent_loops` (2 por defecto), `loop_bridge` y `auto_map` hacia `modeling_gemma_loop.py`, para cargar con `trust_remote_code=True`.
  - `export_raw(artifacts: Path, out_dir: Path) -> Path`: parquet con generaciones crudas, juicios crudos y resultados por ítem (para publicarlos como dataset; el diseño exige publicarlos).

- [ ] **Step 1: Tests que fallan** — con resultados de ejemplo, `paper.md` contiene las secciones `Abstract`, `Method`, `Setup`, `Results`, `Creative evaluation`, `OpenCode adaptation`, `Limitations and negative results`, `Reproducibility`, `Licenses and attributions`; contiene la tabla de tres columnas y los veredictos S1–S5 tal cual; el nombre del modelo es "Gemma-Loop" y aparece "based on Gemma 4"; aparece la declaración de que los datos del maestro fueron generados por IA (DeepSeek); si se usó WebSight, aparece su declaración; si el maestro y el juez son de la misma familia, la sección de limitaciones lo dice. `tests/release/test_export.py`: con `tiny_text_model` y un adaptador de prueba, `export_merged` produce un directorio que `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` carga y cuyas salidas con N=2 coinciden con el modelo en loop original; `export_raw` escribe un parquet con una fila por (variante, benchmark, ítem, muestra).
- [ ] **Step 2: Implementar y verificar.**
- [ ] **Step 3: Construir** — `uv run python scripts/build_report.py --out report/build` y `uv run python scripts/export_release.py --out artifacts/release`.
- [ ] **Step 4: Paso del autor** — el autor revisa `report/build`, genera su página de GitHub con el paper y sube a Hugging Face los pesos fusionados, los adaptadores, el model card y el dataset de resultados crudos. El ejecutor no publica nada.
- [ ] **Step 5: Commit** — `git commit -m "feat(report): paper and model card generator"`

---

## Apéndice: índice de referencias

| Archivo | Para qué consultarlo |
|---|---|
| `references/gemma-loop/01-gemma4-architecture.md` | `config.json` reales, índices de capas, KV compartido, PLE, código de transformers 5.22.1, licencia, cifras oficiales |
| `references/gemma-loop/02-looped-transformers.md` | Dory, papers de loops, puente, muestreo de N, por qué R degrada, presupuestos de tokens publicados |
| `references/gemma-loop/03-training-stack.md` | Instalación en Windows/Blackwell, bitsandbytes, memoria en 12 GB, checkpoints, precios de nube |
| `references/gemma-loop/04-evaluation.md` | LiveCodeBench, MMLU-Pro, GPQA, AIME, ajustes de Google, estadística, costo de evaluación |
| `references/gemma-loop/05-opencode.md` | Configuración de OpenCode, captura, herramientas y prompt, paralelo, Términos de OpenCode |
| `references/gemma-loop/06-judge-and-deepseek.md` | API de DeepSeek, términos, metodología del juez, Playwright, rúbrica |
| `references/gemma-loop/07-sft-and-data.md` | Datasets y licencias, descontaminación, TRL con Gemma 4, formato de tool calls, SFT de agentes |
