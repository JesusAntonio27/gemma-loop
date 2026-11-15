# 03 — Stack de entrenamiento e inferencia en esta máquina

Máquina objetivo: Windows 11, RTX 5070 12 GB (Blackwell, sm_120), driver 617.42 (CUDA UMD 13.4 según `nvidia-smi` local), CUDA toolkit 12.8, Python 3.12 vía `uv`, 31.8 GB RAM, WSL2 solo con distro de Docker.
Fecha de la investigación: 2026-11-03. Todo precio y versión está fechado.

## Resumen

- El índice `cu128` de PyTorch llega **solo hasta `torch==2.11.0`**; `cu130` cubre 2.9.0–2.14.1 y `cu132` cubre 2.12.0–2.15.0. Para sm_120 no hace falta CUDA 13: `2.11.0+cu128` es la última versión “cu128” publicada y funciona.
- `transformers` mínimo real para Gemma 4: **5.5.0** (el `config.json` oficial se generó con `5.5.0.dev0`). Hoy hay 5.22.1, pero **vLLM 0.33.0 pincha `transformers<5.21.0`** y `torch==2.14.0`; hay que aislar vLLM en su propio venv.
- `bitsandbytes 0.50.2` documenta soporte **Windows x86-64 + CUDA 12.8 con sm120 y QLoRA 4-bit ✅**. Ya no es cierto que QLoRA “solo funcione en Linux”.
- Gemma 4 E4B tiene **42 capas = 7×(5 sliding + 1 full)** y E2B **35 = 7×(4+1)**; se confirma el corte 1/5/1 del diseño. Pero E4B tiene `num_kv_shared_layers=18` y E2B `20`: **el bloque en loop cruza la frontera store→reuse** (capa 22/23 en E4B, 13/14 en E2B). Es el mayor riesgo de implementación.
- El bug de Gemma 4 con `use_cache=False` (logits basura con KV sharing) se cerró en abril de 2026 (`#45312`); aun así fija `transformers>=5.5.x` y **no pongas `model.config.use_cache=False` a mano**.
- La tabla PLE (`embed_tokens_per_layer`) es **2.82 B de parámetros en E4B (5.25 GiB en bf16)**: no la cuantiza bitsandbytes (solo sustituye `nn.Linear`). En 12 GB hay que dejarla en CPU; si no, OOM antes de la primera iteración.
- Atención: FA2 **no lista Blackwell** en su README, FA3 es solo Hopper y FA4 (CuTeDSL) apunta a Hopper/B200. En sm_120 usa **SDPA** (`attn_implementation="sdpa"`); transformers 5.x soporta además `_supports_flash_attn`, `sdpa` y `flex_attn` en Gemma 4.
- vLLM **sí soporta Gemma 4 nativo** (`Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`) y hasta un modelo de loops (`DoryForCausalLM`), pero **no corre en Windows nativo** (WSL2 o Docker) y en 12 GB el margen es mínimo.
- Presupuesto: 25 USD ≈ **90 h de RTX 5090 en Vast** (~0.27 USD/h observado) o **21 h en RunPod Secure** (1.19 USD/h) o **62 h de A100 40 GB** (~0.40 USD/h).

## Hechos verificados

### (a) PyTorch con CUDA 12.8 y sm_120

| Dato | Valor exacto | Fuente |
|---|---|---|
| Última versión en el índice cu128 | `torch-2.11.0+cu128` (win_amd64 y manylinux_2_28) | https://download.pytorch.org/whl/cu128/torch/ |
| Rango completo cu128 | 2.7.0 → 2.11.0 | ídem |
| Rango cu130 | 2.9.0 → 2.14.1 | https://download.pytorch.org/whl/cu130/torch/ |
| Rango cu132 | 2.12.0 → 2.15.0 (2.15.0 sin cp310) | https://download.pytorch.org/whl/cu132/torch/ |
| Pareja exacta en cu128 | torch 2.11.0 ↔ torchvision 0.26.0 ↔ torchaudio 2.11.0 | https://download.pytorch.org/whl/cu128/torchvision/ · https://download.pytorch.org/whl/cu128/torchaudio/ |
| Soporte Blackwell | “Blackwell support added across native kernels, CUDA math libraries, and torch.compile” y fix “for `sm100`, `sm120`” | https://github.com/pytorch/pytorch/releases/tag/v2.7.0 |
| Última estable | PyTorch 2.15.0 (28-oct-2026), wheel por defecto CUDA 13.2 | https://github.com/pytorch/pytorch/releases |
| Python soportado | cp310–cp314 (2.15.0 solo cp311+) | índices de wheels |

El toolkit CUDA instalado **no** condiciona el wheel: los wheels traen su runtime. Solo el driver importa (617.42 soporta hasta CUDA 13.4).

### (b) Versiones mínimas y bitsandbytes

| Paquete | Versión actual (PyPI) | Fecha | Mínimo para Gemma 4 / nota |
|---|---|---|---|
| transformers | 5.22.1 | 2026-10-30 | **>=5.5.0** (config oficial generado con 5.5.0.dev0) |
| peft | 0.21.2 | 2026-10-01 | Unsloth exige >=0.18.0; LoRA opera sobre `nn.Linear` |
| trl | 1.17.1 | 2026-10-30 | requiere `transformers>=4.56.2`, `peft>=0.17.0`, `bitsandbytes>=0.45.2` (extras) |
| accelerate / datasets | 1.15.0 / 5.1.0 | 2026-09-09 / 10-05 | `torch>=2.0`; Unsloth pincha `datasets<5.0.0` |
| bitsandbytes | 0.50.2 | 2026-08-27 | `torch>=2.4`, `python>=3.10` |
| vllm | 0.33.0 | 2026-11-02 | `torch==2.14.0`, `transformers>=5.16.1,<5.21.0` |
| unsloth | 2026.11.1 | 2026-11-03 | `transformers<=5.17.0`, `torch<2.15`, `datasets<5` |
| flash-attn | 2.8.3.post2 | 2026-10-14 | ver (e) |

Versiones de los JSON de PyPI, `https://pypi.org/pypi/<paquete>/json`.

bitsandbytes, estado 4-bit/QLoRA (tabla oficial de aceleradores):

- Linux x86-64 CUDA 12.8: targets `sm70, sm75, sm80, sm86, sm89, sm90, sm100, sm120` → QLoRA ✅
- **Windows x86-64 CUDA 12.8: targets `sm70 … sm100, sm120` → QLoRA ✅** (MSVC 19.43+ para compilar)
- Windows x86-64 CUDA 13.0–13.2: también `sm120` ✅
- Un solo wheel por OS (p. ej. `bitsandbytes-0.50.2-py3-none-win_amd64.whl`, 39 MB): los kernels van dentro, no hay extras `[cu128]`.
- `Linear4bit`/`Linear8bitLt` sustituyen capas **Linear**; embeddings y RMSNorm quedan en su dtype.

Fuentes: https://huggingface.co/docs/bitsandbytes/main/en/installation · https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md · https://pypi.org/pypi/bitsandbytes/json

### (c) Unsloth

- Soporta Gemma 4 E2B/E4B/12B/26B-A4B/31B, texto, visión y audio: https://unsloth.ai/docs/models/gemma-4 y https://unsloth.ai/docs/models/gemma-4/train
- Corre en **Windows, Linux y WSL**: https://raw.githubusercontent.com/unslothai/unsloth/main/README.md
- Blackwell (RTX 50) soportado; guía propia con `TORCH_CUDA_ARCH_LIST="12.0"` para xformers: https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth
- Memoria declarada (QLoRA/LoRA, texto): **E2B entrena en 8 GB**, **E4B en 10 GB**; “E2B LoRA 8–10 GB, E4B LoRA 17 GB” (16-bit). “E4B QLoRA works on 10GB” → en 12 GB entra solo si liberas los ~2.5 GB del escritorio.
- Rendimiento declarado: ~1.5× más rápido y ~60 % menos VRAM que FA2 (el notebook E2B anuncia “1.5x faster / 50% less”).
- Notebooks: `Gemma4_(E2B)-Vision.ipynb`, `Gemma4_(E4B)-Vision.ipynb`, `Gemma4_(E4B)-Audio.ipynb`, `Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb` (github.com/unslothai/notebooks/tree/main/nb). Recomienda **E4B QLoRA** antes que E2B LoRA.
- Model card: https://huggingface.co/unsloth/gemma-4-E4B-it — E2B **2.3 B efectivos / 5.1 B con embeddings**, E4B **4.5 B efectivos / 8 B con embeddings**.

### (d) Ubuntu en WSL2 con GPU

- Solo se instala el **driver Windows**; NVIDIA lo expone como `libcuda.so` dentro de WSL. Texto literal de la guía: “Do not install any Linux display driver in WSL” y “users must not install any NVIDIA GPU Linux driver within WSL 2”.
- El toolkit Linux se instala con el paquete **WSL-Ubuntu** (no contiene driver). **No** usar los meta-paquetes `cuda`, `cuda-12-x` ni `cuda-drivers`; solo `cuda-toolkit-12-x`.
- `nvidia-smi` en WSL puede estar en `/usr/lib/wsl/lib/nvidia-smi`.
- Limitaciones: no hay Unified Memory completo, la memoria pinned es limitada (afecta a dataloaders grandes).
- Microsoft exige kernel ≥ 5.10.43.3.

### (e) Atención en Blackwell consumer

- `Gemma4PreTrainedModel`: `_supports_flash_attn = True`, `_supports_sdpa = True`, `_supports_flex_attn = True`, `_supports_attention_backend = True`. Los ejemplos oficiales de la model card usan `attn_implementation="sdpa"`.
  https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py · https://huggingface.co/docs/transformers/main/en/model_doc/gemma4
- El submódulo de audio tiene `_supports_flash_attn = False` (no afecta si no cargas audio).
- FA2 (README oficial) soporta “Ampere, Ada, or Hopper”; **Blackwell no aparece**. FA3: “optimized for Hopper GPUs… Requirements: H100 / H800”. FA4: “CuTeDSL… optimized for Hopper and Blackwell GPUs (e.g. H100, B200)”.
  https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md
- SDPA elige el kernel CUDA más rápido automáticamente; se puede forzar con `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`.
  https://huggingface.co/docs/transformers/main/en/attention_interface
- Alternativa sin compilar: `attn_implementation="kernels-community/flash-attn2"` (kernels desde el Hub; transformers hace fallback).
- Detalle de Gemma 4: `Gemma4TextAttention.scaling = 1.0` porque aplica `q_norm`/`k_norm` (QK-RMSNorm); el softcap (`final_logit_softcapping=30`) va al final, no en la atención. No cambies el `scaling` al portar la atención.

### (f) vLLM y servidor OpenAI-compatible

- vLLM **no soporta Windows nativo**; la doc remite a WSL2 (o forks de la comunidad). Requiere CC ≥ 7.5 (sm_120 entra).
  https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- Wheels por defecto compilados con **CUDA 12.9**; hay variantes cu128/cu130. Tras instalar, ejecuta `vllm download-kernels` (en Blackwell, si no, compila kernels en el arranque).
- Modelos soportados nativamente: `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`, y **`DoryForCausalLM`** (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`).
  https://docs.vllm.ai/en/latest/models/supported_models/
- Registro out-of-tree:
  ```python
  # entry point de tu paquete
  def register():
      from vllm import ModelRegistry
      ModelRegistry.register_model("YourModelForCausalLM", "your_code:YourModelForCausalLM")
  ```
  `setup.py`: `entry_points={"vllm.general_plugins": ["register_x = your_pkg:register"]}`.
  https://docs.vllm.ai/en/latest/contributing/model/registration.html · https://docs.vllm.ai/en/latest/design/plugin_system/
- Ruta más simple que el plugin: **Transformers modeling backend**. Si tu modelo es un `PreTrainedModel` con `config.json` que tenga `auto_map.AutoModel`, `vllm serve <dir>` lo carga (y `--trust-remote-code` si viene del Hub). Requisitos: `MyAttention` debe llamar **exactamente una vez** a `ALL_ATTENTION_FUNCTIONS.get_interface(...)`, tener `layer_idx` **único** (vLLM indexa el KV cache por ese índice) y pasar `scaling=`.
- `transformers serve`: servidor OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`), tool calling estilo OpenAI, “Qwen and Gemma 4 work out of the box”. Modelos bajo demanda, descarga tras 300 s de inactividad (`--model-timeout`).
  https://huggingface.co/docs/transformers/main/en/serving

### (g) Memoria: por qué 12 GB no son 12 GB

Datos de `config.json` (primarios):
- E4B: 42 capas, hidden 2560, intermediate 10240, `hidden_size_per_layer_input=256`, `vocab_size_per_layer_input=262144`, `num_kv_shared_layers=18`, `vocab_size=262144`, `use_double_wide_mlp=false`.
- E2B: 35 capas, hidden 1536, intermediate 6144, PLE 256, `num_kv_shared_layers=20`, `use_double_wide_mlp=true`.

Cálculo propio (aritmética sobre esos números, no medida):
- PLE = `vocab × capas × 256`: E4B **2.82 B → 5.25 GiB bf16**; E2B **2.35 B → 4.38 GiB**. Es una sola tabla `[vocab, num_layers*ple_dim]`.
- `embed_tokens` (tied con lm_head): E4B 0.671 B → 1.25 GiB; E2B 0.403 B → 0.75 GiB.
- Resto (lineales): E4B ≈ 4.5 B → **NF4 ≈ 2.25 GiB** + escalas; E2B ≈ 2.3 B → ≈1.15 GiB.
- Presupuesto GPU E4B QLoRA con PLE en CPU: 2.25 (NF4) + 1.25 (embed) + 0.5–1 (contexto/workspace) + activaciones (~0.5 GiB a 2K, ~1.5–2 GiB a 8K con checkpointing por bloque).
- KV cache E4B (solo inferencia): ~62 KiB/token contando capas que almacenan → 0.12 GiB a 2K, 0.48 GiB a 8K, 7.75 GiB a 128K.

### (h) Checkpoints y reanudación

- `Trainer.train(resume_from_checkpoint="ruta" | True)`: `True` carga el último checkpoint de `args.output_dir`. Restaura modelo, optimizer, scheduler y RNG.
- `TrainingArguments`: `save_strategy="steps"`, `save_steps` (por defecto 500), `save_total_limit` (borra los antiguos), `save_only_model=False` (por defecto; si `True` **no** guarda optimizer/scheduler/RNG → no se puede reanudar).
- Semillas: `TrainingArguments(seed=..., data_seed=...)`; `transformers.set_seed(...)`. Un checkpoint contiene `config.json`, `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`.
  https://huggingface.co/docs/transformers/main/en/main_classes/trainer

### (i) GPU en la nube (precios con fecha)

RunPod, “Updated October 28, 2026” (https://www.runpod.io/pricing), on-demand por hora:

| GPU | Community | Secure |
|---|---|---|
| RTX 5090 32 GB | 0.69 USD/h | **1.19 USD/h** |
| RTX 4090 24 GB | 0.34 | 0.89 |
| A100 80 GB (PCIe / SXM) | 1.19 / 1.39 | **1.79** / 1.79 |
| RTX PRO 6000 96 GB | 1.69 | 2.49 |

Storage RunPod: Network Volume **0.07 USD/GB/mes** (<1 TB), 0.05 (>1 TB); Volume Disk 0.10/GB/mes en marcha y **0.20/GB/mes parado**. Network volumes solo en Secure Cloud y deben montarse al crear el Pod (no se pueden añadir después).
https://docs.runpod.io/pods/storage/create-network-volumes

Vast.ai: precios dinámicos. Consulta en vivo vía `https://console.vast.ai/api/v0/bundles/?q=...` (2026-11-03):
- RTX 5090 32 GB, `compute_cap=1200`, `dph_total=0.2689` USD/h, 20 vCPU, 64 GB RAM, driver 580.126.09, `rentable=false`.
- A100 SXM4 **40 GB** 0.4030 USD/h y H100 SXM 80 GB 0.60 USD/h, ambos `rentable=true`.
- Interruptible: “50%+ cheaper”, preemptible; reservado hasta −50 %.

## Know-how

### 1. Windows nativo, venv y PyTorch

```powershell
winget install -e --id Python.Python.3.12   # si no está
uv venv .venv --python 3.12
.\.venv\Scripts\activate
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 `
  --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=5.5.0" peft trl accelerate datasets bitsandbytes
# fija la versión exacta en requirements.txt; 5.17.0 es el techo que impone Unsloth

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_capability(), torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir `(12, 0)` y `True`. Si prefieres la última PyTorch (2.15.0), el índice es `cu132` y **no hay cp310**: usa 3.11–3.15.

### 2. WSL2 (solo si necesitas vLLM o kernels Linux)

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --shutdown
```

Dentro de Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y build-essential
# toolkit SIN driver (paquete WSL-Ubuntu de developer.nvidia.com/cuda-downloads)
# NUNCA: apt install cuda / cuda-drivers
nvidia-smi || /usr/lib/wsl/lib/nvidia-smi
```

En `C:\Users\<tú>\.wslconfig` (WSL asigna ~50 % de la RAM por defecto): `[wsl2]` con `memory=20GB`, `processors=8`, `swap=8GB`, `localhostForwarding=true`.

PyTorch dentro de WSL: mismo comando que Windows pero sin `win_amd64` (mismo índice cu128). Para vLLM, venv aparte:

```bash
uv venv .venv-vllm --python 3.12 && source .venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto   # instala torch 2.14.0 + cu130/cu132
vllm download-kernels
```

### 3. QLoRA mínimo con HF Trainer (sin Unsloth)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # bf16, NO fp16 (sm_120 lo soporta)
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it",
    quantization_config=bnb,
    dtype=torch.bfloat16,
    device_map="auto",                      # manda a CPU lo que no cabe (ideal: la PLE)
    max_memory={0: "9GiB", "cpu": "24GiB"}, # deja aire para activaciones
    attn_implementation="sdpa",
)
# Comprueba el reparto real: print(model.hf_device_map)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
))

args = SFTConfig(
    output_dir="runs/qlora_e4b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_length=2048,
    learning_rate=1e-4,
    bf16=True,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps", save_steps=200, save_total_limit=4,
    save_only_model=False,
    logging_steps=10,
    seed=3407, data_seed=3407,
    report_to="none",
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                     processing_class=tokenizer)
trainer.train(resume_from_checkpoint=None)   # al reanudar: "runs/.../checkpoint-1200"
```

Notas de implementación:
- `model.config.use_cache = False` **no** se pone a mano: el bug está corregido, pero `gradient_checkpointing_enable()` lo fuerza por dentro en muchas versiones.
- Con el modelo en loop, aplica el checkpointing **dentro** de cada iteración del bloque (`for n in range(N): hidden = self.block(hidden)`) para que el coste de activaciones sea el de un bloque, no N.
- `embed_tokens_per_layer` (la tabla PLE) debe quedar en CPU y congelada; si la dejas en GPU suma 5.25 GiB y no cabe. Confirma la ruta con `print(model)` antes de moverla a mano (cuelga del `Gemma4TextModel`, no del wrapper condicional).
- Alternativa manual (más frágil con hooks de accelerate): `model.model.embed_tokens_per_layer.to("cpu")` y comprobar que el forward sigue funcionando; cada forward pagará PCIe.

### 4. Unsloth (la vía corta en Windows)

```powershell
uv venv .venv-unsloth --python 3.12
.\.venv-unsloth\Scripts\activate
uv pip install torchao
uv pip install unsloth --torch-backend=auto
```

El orden importa: `--torch-backend` busca `torchao` solo en el índice de PyTorch y en Windows no hay versión reciente, así que sin el primer paso resuelve un Unsloth viejo (README de Unsloth).

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it", max_seq_length=2048,
    load_in_4bit=True, full_finetuning=False,
)
model = FastModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407)
```

Si hay OOM: `per_device_train_batch_size=1`, baja `max_seq_length`, y `use_gradient_checkpointing="unsloth"`.

### 5. Servir el modelo con loops

Opción A (recomendada para evaluar): `transformers serve`.

```bash
uv pip install "transformers[serving]"
transformers serve --model-timeout -1                     # http://localhost:8000
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"./gemma-loop-e4b","messages":[{"role":"user","content":"hola"}],"max_tokens":64}'
```

Opción B (vLLM, solo si el loop no rompe el KV cache): plugin con `ModelRegistry.register_model`, o `auto_map` en `config.json` y `vllm serve <dir>` con el Transformers backend.

### 6. Nube

RunPod: crea el Network Volume en la consola (Storage → New Network Volume) o por API v2 (`POST https://api.runpod.io/v2/network-volumes` con `{"name","size","dataCenter"}`), y móntalo al desplegar el Pod (Secure Cloud, se monta en `/workspace`; no se puede añadir después). Mueve checkpoints con `runpodctl send/receive` o `rsync`.

Vast.ai:

```bash
curl -fsSL https://vast.ai/install.sh | bash     # o pip install vastai
vastai set api-key $VAST_API_KEY
vastai search offers 'gpu_name=RTX_5090 num_gpus=1 rentable=true direct_port_count>=1' -o 'dlperf_usd-'
vastai create instance <OFFER_ID> --image <imagen pytorch con CUDA>=12.8> \
  --disk 100 --ssh --direct --onstart-cmd "nvidia-smi"
vastai search volumes
vastai create volume <VOLUME_OFFER_ID> -s 100 -n gemma-loop-ckpts
vastai copy local:./runs/ <INSTANCE_ID>:/workspace/runs/
vastai destroy instance <INSTANCE_ID>
```

## Trampas

1. **`use_cache=False` + KV sharing = logits basura.** Afecta a E2B/E4B (`num_kv_shared_layers` 20/18). Issue #45242, arreglado en `#45312` (abr-2026). No lo pongas a mano; si tu build es anterior al fix, actualiza.
2. **bitsandbytes no cuantiza embeddings.** La tabla PLE de E4B son 5.25 GiB en bf16. Sin offload a CPU, el OOM ocurre al cargar, no al entrenar. Verifícalo con `torch.cuda.memory_allocated()` tras `from_pretrained`.
3. **El bloque en loop cruza `store`→`reuse`.** E4B: capas 0–21 normales, 22 y 23 `store`, 24–41 `reuse`. Un bloque 1/5/1 (capas 6–35) rompe la semántica: las capas `reuse` del loop 2 consumirían K/V del loop 1. Opciones: (a) cortar el bloque antes de la capa 22 y dejar la cola fuera; (b) aceptar recomputar K/V por loop y declararlo; (c) mover el corte a 2/4/1. Medir N=1 debe dar logits idénticos al original antes de seguir.
4. **FA2 no está listada para Blackwell.** Instalar `flash-attn` en sm_120 puede compilar y fallar en runtime. Usa `sdpa`; si necesitas FA, prueba `kernels-community/flash-attn2` y valida con una generación corta antes de entrenar.
5. **`cu128` se quedó en 2.11.0.** Un `pip install torch --index-url .../cu128` sin pin hoy instala 2.11.0, no 2.15. Si un tutorial asume 2.15, usa `cu132`.
6. **vLLM arrastra su propio torch (2.14.0) y `transformers<5.21`.** No lo instales en el mismo venv que el entrenamiento: te degrada transformers y puede romper PEFT/TRL. Venv separado y `uv pip install vllm --torch-backend=auto`.
7. **Unsloth degrada transformers a <=5.17 y datasets a <5.** Si compartes venv con TRL 1.17/datasets 5.1, el resolver puede dejar un stack incoherente. Un venv por herramienta.
8. **`save_only_model=True` impide reanudar** (no guarda optimizer/scheduler/RNG). Déjalo en `False` y limita con `save_total_limit`. Y ojo: `resume_from_checkpoint=True` usa `output_dir`, falla si aún no hay checkpoint (primera corrida: `None`) y empieza de cero si cambias de `output_dir`. Pásale la ruta explícita.
9. **El escritorio come VRAM.** `nvidia-smi` local muestra ~2.5 GB usados por Edge/WebView. Con 12 GB reales trabaja con ~9.5 GB; cierra el navegador antes de entrenar y mide `torch.cuda.mem_get_info()`.
10. **OneDrive sincronizando checkpoints.** El repo está dentro de OneDrive: cada `save_steps` sube cientos de MB y puede bloquear el archivo. Guarda `output_dir` fuera de OneDrive (p. ej. `D:\runs`).
11. **bf16, no fp16.** El audio de Gemma 4 usa `-1e9` en `masked_fill` y desborda fp16 (documentado por Unsloth). En sm_120 bf16 es nativo.
12. **Loss 13–15 en E2B/E4B es normal** (peculiaridad multimodal, documentada por Unsloth). No es señal de que el loop esté mal; compara contra O-H con los mismos datos.
13. **No cambies `attn_implementation` entre train y eval.** SDPA y eager no dan exactamente los mismos logits; el paper debe declarar una sola.
14. **vLLM indexa el KV cache por `layer_idx`.** Un bloque reutilizado N veces con el mismo `layer_idx` no es compatible con el Transformers backend de vLLM sin trabajo extra; para la fase de evaluación usa `transformers serve`.

## Referencias

- https://download.pytorch.org/whl/cu128/torch/ · /torchvision/ · /torchaudio/ — wheels cu128 (2.7.0–2.11.0) y parejas exactas torch↔torchvision↔torchaudio.
- https://download.pytorch.org/whl/cu130/torch/ — wheels cu130 (2.9.0–2.14.1).
- https://download.pytorch.org/whl/cu132/torch/ — wheels cu132 (2.12.0–2.15.0).
- https://github.com/pytorch/pytorch/releases/tag/v2.7.0 — notas de 2.7: soporte Blackwell y fix sm100/sm120.
- https://github.com/pytorch/pytorch/releases — 2.15.0 (28-oct-2026), CUDA 13.2 por defecto.
- https://pypi.org/pypi/transformers/json · /peft/json · /trl/json · /accelerate/json · /datasets/json — versiones 5.22.1 / 0.21.2 / 1.17.1 / 1.15.0 / 5.1.0 y sus dependencias.
- https://pypi.org/pypi/bitsandbytes/json · /vllm/json · /unsloth/json — 0.50.2 (un wheel por OS), 0.33.0 (pins `torch==2.14.0`, `transformers<5.21`), 2026.11.1 (pins transformers/datasets/torch).
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json — 42 capas, 5:1, KV sharing 18, PLE 256×42.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json — 35 capas, 4:1, KV sharing 20, double-wide MLP.
- https://huggingface.co/docs/transformers/main/en/model_doc/gemma4 — clases Gemma4*, PLE, uso con `sdpa`, function calling.
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/configuration_gemma4.py — `kv_sharing_modes`, patrón de capas, forma de la tabla PLE.
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py — `_supports_flash_attn/sdpa/flex_attn`, atención con QK-norm.
- https://github.com/huggingface/transformers/issues/45242 — bug `use_cache=False`, causa y fix.
- https://huggingface.co/docs/transformers/main/en/attention_interface — backends (`sdpa`, `flash_attention_2`, `flex_attention`, kernels del Hub).
- https://huggingface.co/docs/transformers/main/en/serving — `transformers serve`, endpoints y tool calling.
- https://huggingface.co/docs/transformers/main/en/main_classes/trainer — `resume_from_checkpoint`, `save_only_model`, `save_total_limit`.
- https://huggingface.co/docs/bitsandbytes/main/en/installation — tabla de targets por OS/CUDA (Windows 12.8 incluye sm120).
- https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md — features y requisitos.
- https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md — soporte FA2/FA3/FA4 por arquitectura.
- https://unsloth.ai/docs/models/gemma-4 — modelos, requisitos de memoria, NVFP4, vLLM.
- https://unsloth.ai/docs/models/gemma-4/train — receta QLoRA, VRAM (E2B 8 GB, E4B 10 GB), bugs de Gemma 4.
- https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth — instalación en Blackwell y WSL.
- https://raw.githubusercontent.com/unslothai/unsloth/main/README.md — soporte Windows/WSL, instalación con `torch-backend`.
- https://huggingface.co/unsloth/gemma-4-E4B-it — parámetros efectivos/totales y benchmarks oficiales.
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html — CUDA en WSL2 y la advertencia de no instalar driver Linux.
- https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl — requisitos de WSL y kernel.
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html — instalación, CUDA 12.9 por defecto, no Windows nativo.
- https://docs.vllm.ai/en/latest/models/supported_models/ — Gemma 4 / Dory nativos y backend Transformers.
- https://docs.vllm.ai/en/latest/contributing/model/registration.html · /design/plugin_system/ — `ModelRegistry.register_model` y entry point `vllm.general_plugins`.
- https://www.runpod.io/pricing — precios on-demand y storage (actualizado 28-oct-2026).
- https://docs.runpod.io/pods/storage/create-network-volumes — network volumes, montaje y API.
- https://docs.vast.ai/cli/get-started — CLI de Vast, `search offers`, `create instance`, `copy`, `destroy`.
- https://docs.vast.ai/cli/reference/create-volume — `vastai create volume`.
- https://console.vast.ai/api/v0/bundles/?q=%7B%22limit%22%3A5%7D — ofertas y precios en vivo de Vast (consultado 2026-11-03).

## Sin verificar

- Que `bitsandbytes 0.50.2` cargue kernels sm120 **en esta máquina concreta** con torch 2.11.0+cu128 y driver 617.42: la tabla oficial lo dice, pero no se ejecutó. Smoke test pendiente: cargar E2B en NF4 y hacer un forward.
- Que existan wheels de `flash-attn 2.8.3.post2` con cubins sm120: el README no lo declara y no se inspeccionó el wheel.
- Que `transformers serve` acepte un directorio local con código propio (`auto_map`) sin `trust_remote_code`: la página de serving no lo documenta explícitamente.
- Que vLLM 0.33.0 sirva Gemma 4 en 12 GB con contexto útil: no se ejecutó; el margen de VRAM es dudoso.
- Las cifras de memoria de (g) son aritmética sobre `config.json`, no medidas con `nvidia-smi`.
- Que Unsloth E4B QLoRA entre en 12 GB con `max_seq_length=8192`: solo hay declaración de 10 GB para la receta corta.
- Precios de Vast.ai: cambian por hora; los valores citados son una foto de las ofertas devueltas por la API el 2026-11-03, y la oferta de 5090 consultada tenía `rentable=false`.
- Tarifas de salida de red de Vast (p. ej. 0.0026 USD/TB en la oferta de 5090) no se contrastaron con la factura real.
