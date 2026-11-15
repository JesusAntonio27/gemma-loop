# 05 — OpenCode como arnés de agente y captura de trayectorias

Verificado el 2026-11-03 contra OpenCode **1.19.3** instalado en este PC, el repo `anomalyco/opencode`
(tag `v1.19.3`) y la documentación de `opencode.ai`. `sst/opencode` redirige hoy a `anomalyco/opencode`
(la API de GitHub devuelve `full_name: "anomalyco/opencode"`).

## Resumen

- OpenCode 1.19.3 es MIT, TypeScript, y en Windows guarda **todo** en un SQLite: `C:\Users\<user>\.local\share\opencode\opencode.db`
  (aquí 12,9 GB para 753 sesiones). El directorio `storage/` ya **no** contiene las sesiones, solo `session_diff` y `migration`.
- El formato interno es `session` → `message.data` (JSON con rol/modelo/tokens/coste) → `part.data` (JSON con
  `text`, `reasoning`, `tool`, `step-start`, `step-finish`, `patch`, `file`).
- `opencode export <sessionID>` produce `{info, messages:[{info, parts:[...]}]}` con mensajes y llamadas a herramientas.
  Es la vía estable para SFT; el SQLite es la vía rápida pero requiere parar el servidor o hacer backup con WAL.
- Para SFT en formato OpenAI hace falta un **proxy de registro**: el request crudo (system + tools + messages) no se
  guarda en ninguna tabla. `mitmdump --mode reverse:...@4000` + un addon de ~15 líneas captura request/response exactos
  sin instalar certificados.
- El system prompt de un modelo no-OpenAI/GPT/Claude/Kimi/Gemini es `packages/opencode/src/session/prompt/default.txt`
  (8.528 bytes), y se envía como **un solo mensaje `system`** junto con `<env>`, `AGENTS.md`, skills y MCP.
- El agente `build` usa las herramientas `bash, read, glob, grep, edit, write, task, webfetch, todowrite, websearch, skill,
  apply_patch`; `opencode run --format json --thinking` emite `text/reasoning/tool_use/step_start/step_finish/error`.
- OpenCode Go cuesta **$10/mes** y da **$100/mes** de límite para `deepseek-v4.1-flash` (20% por 5 h, 50% semanal),
  con precios $0,15/$0,60 por 1M tokens off-peak y $0,30/$1,20 peak.
- **Aviso legal crítico:** los Términos de OpenCode (15-ago-2026) prohíben “automatically or programmatically extract
  data or Output” y “use Output to develop artificial intelligence models that compete with the Services or any Third
  Party Models”. Recolectar trayectorias de `opencode-go` para entrenar un agente de código cae de lleno en esas cláusulas.
- Harbor (Terminal-Bench 4.0) trae `opencode` preintegrado: `harbor run -d terminal-bench/terminal-bench@4.0.0 -a opencode -m <provider/model>`.

## Hechos verificados (con URL)

### Versión, repo y licencia

- Versión instalada: `opencode --version` → `1.19.3`; `opencode debug info` → `opencode version: 1.19.3`, `os: Windows_NT 10.0.26200 x64`.
  Comandos locales, sin URL.
- Repo: <https://api.github.com/repos/sst/opencode> devuelve `full_name: anomalyco/opencode`, `license.spdx_id: MIT`,
  `default_branch: dev`, `language: TypeScript`.
- Las rutas de datos/config se obtienen con `opencode debug paths` (documentado en <https://opencode.ai/docs/cli/#paths>):
  `data C:\Users\<user>\.local\share\opencode`, `config C:\Users\<user>\.config\opencode`,
  `cache C:\Users\<user>\.cache\opencode`, `state C:\Users\<user>\.local\state\opencode`.

### (a) Configuración

- `opencode.json`/`opencode.jsonc`; global en `~/.config/opencode/opencode.json`, proyecto en `opencode.json`; los archivos
  se **fusionan** y el de proyecto gana. Orden completo en <https://opencode.ai/docs/config/#precedence-order>.
- Proveedor OpenAI-compatible: `npm: "@ai-sdk/openai-compatible"`, `options.baseURL`, `options.apiKey`, `options.headers`,
  `models.<id>` y `limit.context`/`limit.output` (ambos obligatorios). Ejemplo oficial en
  <https://opencode.ai/docs/providers/#custom-provider>.
- El esquema `https://opencode.ai/config.json` admite por modelo: `tool_call` (boolean), `reasoning`, `temperature`,
  `attachment`, `interleaved`, `limit`, `options`, `variants` (verificado descargando el JSON).
- `small_model` se usa para títulos/resúmenes; si no se fija, OpenCode elige uno barato del proveedor. En local hay que
  fijarlo al mismo modelo o fallará la generación de título (<https://opencode.ai/docs/config/#models>).
- Agentes: `agent.<name>` con `mode` (`primary`/`subagent`/`all`), `model`, `prompt`, `temperature`, `top_p`, `steps`,
  `permission`, `hidden`. `default_agent` elige el primario por defecto; `build` y `plan` son los integrados
  (<https://opencode.ai/docs/agents/>, <https://opencode.ai/docs/config/#default-agent>).
- Permisos: `permission` acepta `"allow"|"ask"|"deny"` o mapas glob; `--auto` aprueba lo que no esté denegado
  explícitamente. `doom_loop` y `external_directory` son `ask` por defecto; `read` permite todo salvo `.env*`
  (<https://opencode.ai/docs/permissions/>).
- Headless: `opencode run [message..]` con `--format json`, `--model provider/model`, `--agent`, `--dir`, `--auto`,
  `--thinking`, `--session`, `--attach`. Flags exactos en <https://opencode.ai/docs/cli/#run-1> y en `opencode run --help`.
- Servidor: `opencode serve --port 4096 --hostname 127.0.0.1`; OpenAPI 3.1 en `/doc`; endpoints clave
  `POST /session`, `POST /session/:id/message`, `POST /session/:id/prompt_async`, `GET /session/:id/message`,
  `GET /experimental/tool?provider=&model=`, `GET /event` (SSE). Ver <https://opencode.ai/docs/server/>.
- SDK JS: `npm install @opencode-ai/sdk`; `createOpencode()` levanta servidor+cliente, `createOpencodeClient({baseUrl})`
  se conecta a uno existente; `client.session.prompt({path:{id}, body:{model, parts}})` y `client.event.subscribe()`
  (<https://opencode.ai/docs/sdk/>).

### (b) Sesiones: dónde y en qué formato

- Ruta exacta: `opencode db path` → `C:\Users\<user>\.local\share\opencode\opencode.db`.
- Tablas (verificado con `opencode db "SELECT name FROM sqlite_master WHERE type='table'"`):
  `session, message, part, session_message, session_input, session_context_epoch, todo, permission, project, event`.
- Esquema mínimo (verificado con `PRAGMA table_info`):
  - `session(id, project_id, parent_id, slug, directory, title, version, agent, model, cost, tokens_input,
    tokens_output, tokens_reasoning, tokens_cache_read/time_created/time_updated...)`
  - `message(id, session_id, time_created, time_updated, data TEXT)`
  - `part(id, message_id, session_id, time_created, time_updated, data TEXT)`
- `message.data` real: `{"role":"assistant","parentID":"msg_...","agent":"build","modelID":"...","providerID":"...",
  "cost":0,"tokens":{"input":..,"output":..,"reasoning":..,"cache":{"read":..,"write":..}},"time":{...},"error":{...}}`.
- `part.data` real: `{"type":"tool","tool":"webfetch","callID":"call_...","state":{"status":"completed","input":{...},
  "output":"...","title":"...","metadata":{...},"time":{...}}}`. Tipos observados en 111.672 parts:
  `tool` 31.219, `step-start` 26.802, `step-finish` 26.760, `reasoning` 18.162, `text` 5.098, `patch` 3.493, `file` 209.
- `opencode export <sessionID>` (documentado en <https://opencode.ai/docs/cli/#export>) devuelve un JSON con
  `info` (la fila de `session`) y `messages: [{info, parts}]`; verificado con una sesión real. `--sanitize` redacta
  transcripción y datos de archivos. `opencode import <file>` lo restaura.
- `opencode session list --format json -n N` lista `{id,title,updated,created,projectId,directory}`.
- En 1.19.3 `storage/` solo contiene `session_diff/` y `migration/`; no hay JSON por sesión como en versiones antiguas.

### (c) Herramientas y system prompt

- IDs reales devueltos por `GET /experimental/tool/ids` en 1.19.3:
  `invalid, question, bash, read, glob, grep, edit, write, task, webfetch, todowrite, websearch, skill, apply_patch`.
- Esquemas exactos (servidos por `GET /experimental/tool?provider=opencode-go&model=deepseek-v4.1-flash`):
  - `bash {command*, timeout(ms), workdir}`
  - `read {filePath*, offset, limit}`
  - `glob {pattern*, path}`
  - `grep {pattern*, path, include}`
  - `edit {filePath*, oldString*, newString*, replaceAll}`
  - `write {filePath*, content*}`
  - `task {description*, prompt*, subagent_type*, task_id, command}`
  - `webfetch {url*, format: text|markdown|html, timeout(s)}`
  - `todowrite {todos:[{content,status,priority}]*}`
  - `websearch {query*, numResults, livecrawl, type, contextMaxCharacters}`
  - `skill {name*}`; `question {questions*}`; `apply_patch` (solo modelos `gpt-*`).
- Implementación: `packages/opencode/src/tool/`. Ojo: el **id** no coincide con el nombre del archivo:
  `shell.ts`→`bash`, `todo.ts`→`todowrite`, `webfetch.ts`→`webfetch`, `apply_patch.ts`→`apply_patch`. Las descripciones
  que ve el modelo están en los `.txt` hermanos (`bash` no tiene `.txt`; usa el texto de `shell.ts`).
- `apply_patch` sustituye a `edit` y `write` cuando el `modelID` contiene `gpt-` y no contiene `oss` ni `gpt-4`
  (`packages/opencode/src/tool/registry.ts`, función `tools`). Para Gemma se usan `edit`/`write`.
- `websearch` solo se expone si `providerID` es `opencode`/`opencode-go` o si `OPENCODE_ENABLE_EXA`/`OPENCODE_ENABLE_PARALLEL`
  está activo (`registry.ts`, `webSearchEnabled`).
- System prompt por familia: `packages/opencode/src/session/system.ts`, función `provider(model)`. Elige
  `anthropic.txt` para `claude`, `gemini.txt` para `gemini-`, `gpt.txt`/`codex.txt`/`gpt-astra.txt` para GPT,
  `kimi.txt`, `trinity.txt`, `meta.txt` para “muse”, y **`default.txt` para todo lo demás** (Gemma cae aquí).
- `default.txt`: 8.528 bytes, 95 líneas; empieza `You are opencode, an interactive CLI tool that helps users with
  software engineering tasks.` Incluye tono, estilo, política de herramientas y referencias `file:línea`.
- Ensamblado exacto en `packages/opencode/src/session/llm/request.ts`, `prepare()`:
  `[agent.prompt ?? SystemPrompt.provider(model), ...system, user.system].join("\n")`, donde `system` =
  `[env, instructions, mcp, skills]`. Se envía como un único `{role:"system"}` al principio de `messages`.
- `SystemPrompt.environment()` añade un bloque `<env>` con `Working directory`, `Workspace root folder`,
  `Is directory a git repo`, `Platform` y `Today's date`; también lista `<available_references>`.
- `instructions` = `AGENTS.md` del proyecto + `~/.claude/CLAUDE.md` y skills de `.claude` (se pueden desactivar con
  `OPENCODE_DISABLE_CLAUDE_CODE=1`, `OPENCODE_DISABLE_EXTERNAL_SKILLS=1`; ver <https://opencode.ai/docs/cli/#environment-variables>).
- Los `tools` se ordenan alfabéticamente por nombre antes de enviarse (`Object.entries(tools).toSorted(...)`).
- Cabeceras por proveedor (`request.ts`): siempre `x-opencode-session-id`; solo si `providerID` empieza por `opencode`
  añade `x-opencode-session`, `x-opencode-request`, `x-opencode-client` y `User-Agent: opencode/<version>`.

### (d) Headless y captura

- `opencode run --format json` emite líneas JSON con `{type, timestamp, sessionID, ...}`. Tipos en el código
  (`packages/opencode/src/cli/cmd/run.ts`): `text`, `reasoning`, `tool_use`, `step_start`, `step_finish`, `error`.
  `--thinking` es necesario para que aparezcan los bloques `reasoning`. En 1.19.3 **no** existe el evento `user`
  que sí espera Harbor; el prompt hay que recuperarlo de `export` o de la base.
- `tool_use` lleva el `part` completo, es decir `state.input` y `state.output`; `step_finish` lleva
  `tokens{input,output,reasoning,cache}` y `cost`.
- El flag documentado es `--auto`; el código acepta además los alias ocultos `--yolo` y
  `--dangerously-skip-permissions` (los tres ponen `auto=true`).
- `opencode serve` expone `GET /experimental/tool?provider=<p>&model=<m>` con los esquemas JSON que recibe el modelo,
  y `GET /doc` con el OpenAPI. Es la forma más barata de auditar el contrato exacto sin gastar tokens.

### (e) Benchmarks que soportan OpenCode

- Terminal-Bench 4.0 corre sobre Harbor: `uv tool install 'harbor[modal]'` y
  `harbor run -d terminal-bench/terminal-bench@4.0.0 -e modal -a claude-code -m anthropic/claude-sonnet-5 -k 5`
  (<https://www.tbench.ai/docs>).
- Harbor lista `opencode` entre sus agentes preintegrados y le atribuye ATIF, resume, skills, MCP, web tools y ACP
  (<https://docs.harborframework.com/agents/pre-integrated-agents.md>).
- Integración exacta (`src/harbor/agents/installed/opencode.py`): instala `npm i -g opencode-ai@<version|latest>`,
  escribe `~/.config/opencode/opencode.json`, y ejecuta
  `opencode --model=<m> run --format=json --thinking --dangerously-skip-permissions -- <instrucción> | tee opencode.txt`.
  Acepta `--ak opencode_config='{...}'` para superponer configuración y `--ak version=1.19.3`.
- Convierte el stdout JSONL a `trajectory.json` en formato ATIF v1.7 (`step_start`/`step_finish` delimitan pasos).

### (f) Paralelo

- No hay flag de timeout en `opencode run`; se usa el `timeout` del sistema. `--dir` elige el directorio de trabajo y
  `--auto` evita bloqueos por permisos.
- `OPENCODE_DB` permite apuntar a otra base (documentado en <https://opencode.ai/docs/cli/#environment-variables>).
- Cada proceso `opencode run` levanta su propio servidor; para muchas tareas conviene un `opencode serve` y
  `opencode run --attach http://127.0.0.1:4096 --dir <tarea>`.
- SQLite en WAL: un solo proceso puede escribir, muchos leer. Paralelizar contra una única DB sin servidor compartido
  produce `database is locked`; una DB por worker elimina el problema.

### (g) OpenCode Go

- Planes: Go **$10/mes**, Go Plus **$40/mes**, Go Teams **$20/miembro/mes**. Límites por modelo, no un cupo único.
- `deepseek-v4.1-flash`: $0,15 input / $0,60 output por 1M off-peak; $0,30/$1,20 peak; cached read $0,003/$0,006;
  **límite mensual $100** en Go ($200 en Go Plus). Peak = 01:00–04:00 y 06:00–10:00 UTC, lunes a viernes.
- Límites temporales: 5 h = 20% del mensual, semana = 50%, mes = 100%. Estimación publicada: 43.330 requests/5 h,
  108.330/semana, 216.670/mes para ese modelo (<https://opencode.ai/v2/docs/console/go/>).
- Endpoint nativo: `https://opencode.ai/inference/go/openai/v1/chat/completions`; `opencode models opencode-go --verbose`
  confirma `url: https://opencode.ai/inference/go/openai/v1`, `capabilities.toolcall: true`, `reasoning: true`.
- Privacidad del modelo: “Model training: Not used”, retención “0 days” (tabla Privacy del mismo doc).
- **Términos (15-ago-2026), <https://opencode.ai/legal/terms-of-service>:** prohibido (5) “automatically or
  programmatically extracts data or Output”; (7) “uses Output to develop artificial intelligence models that compete
  with the Services or any Third Party Models”; (11) “crawls, scrapes, or spiders”. El usuario es dueño del Output,
  pero el uso queda restringido por esas cláusulas. El software open-source sin hosted sí es MIT.

## Know-how

### 1. `opencode.json` mínimo para un modelo local capturado

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "gemma-loop/gemma-loop",
  "small_model": "gemma-loop/gemma-loop",
  "default_agent": "build",
  "autoupdate": false,
  "provider": {
    "gemma-loop": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Gemma Loop (local)",
      "options": { "baseURL": "http://127.0.0.1:4000", "apiKey": "sk-local",
                   "timeout": 1800000, "chunkTimeout": 120000 },
      "models": {
        "gemma-loop": { "name": "Gemma 4 E4B loop N=2", "tool_call": true,
                        "reasoning": true, "temperature": true,
                        "limit": { "context": 32768, "output": 8192 } }
      }
    }
  },
  "permission": { "external_directory": "allow", "doom_loop": "allow" }
}
```

`limit.context` debe ser el contexto **real** del servidor; si se queda corto, OpenCode compacta antes de tiempo; si se
pasa, el proveedor devuelve 400. `output` es obligatorio junto con `context`.

### 2. Proxy de registro con mitmproxy (recomendado)

```bash
pipx install mitmproxy
mitmdump --mode reverse:https://opencode.ai/inference/go/openai/v1@4000 -s capture.py
```

`capture.py` (addon propio; no copiado de ninguna fuente):

```python
import json, time
from mitmproxy import http
LOG = "capture.jsonl"

def request(flow: http.HTTPFlow):
    flow.metadata["t0"] = time.time()
    sid = flow.request.headers.get("x-opencode-session-id", "")   # siempre presente
    flow.request.headers["x-opencode-session"] = flow.request.headers.get("x-opencode-session", sid)

def response(flow: http.HTTPFlow):
    try: body = json.loads(flow.request.get_text() or "{}")
    except Exception: body = {"_raw": flow.request.get_text()}
    rec = {"ts": time.time(), "t0": flow.metadata.get("t0"), "url": flow.request.pretty_url,
           "request": body, "status": flow.response.status_code,
           "response_raw": flow.response.get_text(), "session": flow.request.headers.get("x-opencode-session-id")}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

Puntos finos:

- En modo `reverse` mitmproxy termina el TLS contra el upstream; no hay que instalar CA en Windows.
- El body de respuesta puede ser SSE (`data: {...}\n\n`). Guarda `response_raw` y reconstruye después
  `choices[0].delta.content`, `delta.reasoning_content` y `delta.tool_calls[index]`.
- Para un modelo local: `--mode reverse:http://127.0.0.1:8000/v1@4000` y `baseURL: "http://127.0.0.1:4000"`.
- Para Go, mismo proxy con `--mode reverse:https://opencode.ai/inference/go/openai/v1@4000` y este provider:
  ```jsonc
  "provider": { "go-proxy": { "npm": "@ai-sdk/openai-compatible", "name": "Go capture",
    "options": { "baseURL": "http://127.0.0.1:4000", "apiKey": "{env:OPENCODE_GO_API_KEY}" },
    "models": { "deepseek-v4.1-flash": { "tool_call": true, "reasoning": true,
      "limit": { "context": 128000, "output": 32000 } } } } }
  ```
  El `Authorization: Bearer <OPENCODE_GO_API_KEY>` lo pone OpenCode desde `apiKey`; el proxy no debe loguearlo.
- Alternativa sin proxy: `--format json --thinking` + `opencode export`. Sirve para SFT de texto/tool calls, pero **no**
  contiene el system prompt ni los esquemas de tools; hay que reconstruirlos desde el repo.

### 3. Extraer trayectorias de la DB

```bash
opencode db "SELECT id, title, directory, time_created FROM session WHERE title LIKE '%task%' ORDER BY time_created DESC LIMIT 20"
opencode db "SELECT data FROM part WHERE session_id='ses_...' AND json_extract(data,'$.type')='tool'"
opencode export ses_... > session.json
```

Join correcto: `part.message_id = message.id AND part.session_id = message.session_id`, ordenar por
`message.time_created, part.time_created, part.id`. No ordenes solo por `part.time_created`: varios parts del mismo
mensaje comparten milisegundo.

Para SFT: conserva solo sesiones con resultado exitoso (tests/exit code), entrena solo turnos `assistant`, y conserva
`tool` parts como `tool_calls` + `tool` messages. Descarta `step-start`, `step-finish`, `patch` y `file` salvo que se
quieran como supervisión auxiliar.

### 4. Correr muchas tareas en paralelo

```bash
for d in tasks/*/; do
  OPENCODE_DB="$PWD/.oc/$(basename "$d").db" \
    timeout 1800 opencode run --dir "$d" --auto --format json --thinking \
      --model gemma-loop/gemma-loop "Resuelve la tarea descrita en TASK.md" \
      > "$d/run.jsonl" 2> "$d/err.log" &
done
wait
```

- Un `OPENCODE_DB` absoluto por worker evita bloqueos; si se usa un solo `opencode serve`, no hace falta.
- `timeout` es de coreutils (WSL/Git Bash). En PowerShell: `Start-Process` + `WaitForExit(ms)` + `Kill()`.
- Aísla el entorno: `OPENCODE_DISABLE_CLAUDE_CODE=1`, `OPENCODE_DISABLE_EXTERNAL_SKILLS=1`, `HOME`/`XDG_*` por worker
  si no quieres que se mezclen auth, skills y snapshots.
- Para evaluación, fija `--agent build` explícito y `temperature` en el agente; no confíes en el default del modelo.

### 5. Reproducir el prompt exacto

`git clone --depth 1 --branch v1.19.3 https://github.com/anomalyco/opencode`; para Gemma el texto base es
`packages/opencode/src/session/prompt/default.txt`; añade el bloque `<env>` de `session/system.ts` y el `AGENTS.md`
del proyecto. Todo va en un único mensaje `system`; no hay prompts separados por herramienta.

## Trampas

- **No copiar `opencode.db` en caliente.** Hay `-wal` y `-shm`; usa `sqlite3 opencode.db ".backup backup.db"` o para el
  servidor. Una copia directa puede quedar inconsistente.
- **La DB crece muchísimo** (12,9 GB aquí con 753 sesiones y 111.672 parts). Rota por proyecto con `OPENCODE_DB` y
  limpia `tool-output/`; `OPENCODE_DISABLE_PRUNE` desactiva la poda automática.
- **`opencode export` no incluye el system prompt ni los tools.** Para SFT de agente hay que capturar el request.
- **`--format json` en 1.19.3 no emite el turno `user`.** Harbor lo sabe y lo recupera de la instrucción; si escribes tu
  propio parser, añade el prompt a mano.
- **`--thinking` es obligatorio** para que `reasoning` aparezca en el JSON; sin él se pierde la cadena de pensamiento.
- **El flag de permisos documentado es `--auto`.** `--dangerously-skip-permissions` y `--yolo` existen pero están
  ocultos en la ayuda; funcionan en 1.19.3.
- **El id de la herramienta no es el nombre del archivo.** `shell.ts` exporta `bash`; `todo.ts` exporta `todowrite`.
  Si buscas `bash.ts` no lo encontrarás.
- **`apply_patch` cambia el conjunto de tools según el modelo.** Un `modelID` con `gpt-` desactiva `edit`/`write`; si
  entrenas y evalúas con nombres distintos, el contrato de tools cambia.
- **El system prompt incluye rutas y fecha.** `Working directory: C:\Users\...` y `Today's date` se filtran al modelo;
  normalízalos o elimínalos al construir el dataset para que no aprenda el path de tu máquina.
- **`AGENTS.md` y `~/.claude/CLAUDE.md` se concatenan.** Sin `OPENCODE_DISABLE_CLAUDE_CODE=1`, el prompt de tu Claude
  Code global entra en el entrenamiento y en la evaluación.
- **`small_model` puede llamar a la nube.** Si no lo fijas, OpenCode puede usar un modelo barato del proveedor para
  títulos; en local esto rompe la reproducibilidad y filtra datos.
- **El contexto de OpenCode es caro:** system prompt (~2,1k tokens) + tools + `AGENTS.md` + historial. Con 12 GB y
  Gemma 4 E4B, trunca por turno; no metas la sesión completa.
- **OpenCode Go no es un dataset gratuito.** El cupo de $100/mes es para inferencia, pero los Términos prohíben la
  extracción programática de Output y su uso para modelos que compitan. No entrenes con `opencode-go` sin permiso escrito.
- **Harbor instala `@latest` por defecto.** Fija `--ak version=1.19.3`; si no, un cambio de OpenCode rompe la comparación.
- **Terminal-Bench corre en contenedores Linux.** Un servidor local en Windows no es alcanzable como `127.0.0.1`;
  usa `host.docker.internal` o el gateway del sandbox y `--ae`.

## Referencias

- <https://api.github.com/repos/sst/opencode> — metadatos del repo; redirige a `anomalyco/opencode`, licencia MIT.
- <https://opencode.ai/docs/config/> — formato, ubicaciones, precedencia, agentes, permisos, compaction.
- <https://opencode.ai/docs/providers/> — proveedores, `@ai-sdk/openai-compatible`, `baseURL`, `limit`, custom provider.
- <https://opencode.ai/docs/cli/> — `run`, `serve`, `export`, `import`, `session`, `db`, `debug paths`, env vars.
- <https://opencode.ai/docs/server/> — API HTTP, OpenAPI, sesiones, mensajes, tools experimentales, SSE.
- <https://opencode.ai/docs/permissions/> — `allow/ask/deny`, `--auto`, defaults, `doom_loop`.
- <https://opencode.ai/config.json> — esquema JSON; confirma `tool_call`, `limit`, `reasoning`.
- <https://opencode.ai/v2/docs/console/go/> — planes, precios, límites y modelos de OpenCode Go.
- <https://opencode.ai/legal/terms-of-service> — Términos; cláusulas 5, 7 y 11 sobre Output y extracción.
- <https://www.tbench.ai/docs> — Terminal-Bench 4.0 sobre Harbor; instalación y comando.
- <https://docs.harborframework.com/agents/pre-integrated-agents.md> — agentes soportados, `opencode`, ATIF, opciones.
- <https://raw.githubusercontent.com/harbor-framework/harbor/main/src/harbor/agents/installed/opencode.py> — integración real: instalación, config, comando y parser.
- <https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/system.ts> — selección de prompt por modelo.
- <https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/llm/request.ts> — ensamblado del system, orden de tools, cabeceras.
- <https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/prompt/default.txt> — prompt que ve Gemma.
- <https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/tool/registry.ts> — ids, selección de `apply_patch`, `websearch`.
- <https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/cli/cmd/run.ts> — eventos JSON y flags ocultos.
- <https://docs.mitmproxy.org/stable/concepts/modes/> — modo reverse y captura local.

## Sin verificar

- `[NO VERIFICADO]` Que los Términos de OpenCode permitan explícitamente entrenar un modelo derivado con salidas de
  OpenCode Go. La cláusula 7 parece prohibirlo si el modelo compite; hace falta revisión legal o permiso escrito.
- `[NO VERIFICADO]` Que Harbor acepte `--ak opencode_config='{...}'` como JSON inline en la CLI (el campo existe en el
  código y en `OpenCodeOptions`, pero no vi un ejemplo de CLI con ese kwarg).
- `[NO VERIFICADO]` Que `opencode run --attach` permita `--dir` a un directorio distinto por invocación en el mismo
  servidor; el flag existe, pero no lo probé con dos proyectos concurrentes.
