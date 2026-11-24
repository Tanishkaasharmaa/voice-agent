# LocalVoiceAgent

A fully **offline** Windows voice-command agent that runs entirely on CPU,
designed for a low-resource machine (Intel i7-8650U, 16 GB RAM, **no GPU**).

Speak a command → local Whisper transcribes it → a local Ollama agent decides
whether tools are needed and executes them → Kokoro speaks the reply. Nothing
ever leaves your computer.

---

## What it does

| Capability               | Implementation                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------- |
| Speech-to-text           | **Whisper** via `faster-whisper` (CTranslate2, int8 CPU — far lighter than torch)     |
| Voice activity detection | **Silero VAD** (ONNX Runtime), ~800 ms silence = speech end                           |
| Text-to-speech           | **Kokoro** (ONNX, CPU) — Piper as an optional fallback                                |
| Agent LLM                | **Ollama** — local model (e.g. `qwen3.5:4b`) offline; auto-switches to a **cloud model** (e.g. `deepseek-v4-flash:cloud`) when online |
| Tool calling             | Registered tools only — open/close apps, clipboard, notes, memory, time, system info  |
| Memory                   | **SQLite** (conversations, messages, memories, notes, install history)                |
| UI                       | PySide6 floating always-on-top widget + system tray                                   |
| Hotkeys                  | `CTRL+WINDOWS` push-to-talk, `CTRL+WINDOWS+SPACE` auto-stop, interrupt while speaking |
| Setup                    | Every-startup dependency + model detection with Yes/No install prompts                |

**Privacy:** microphone audio, transcripts, conversations, and the database
stay local. No telemetry, no analytics. During normal operation the only
network endpoint is the local Ollama server (`http://127.0.0.1:11434`); when
cloud models are enabled, requests route through it to Ollama's cloud.

---

## Hardware profile

The app auto-detects CPU/RAM and applies the `LOW_RESOURCE_CPU` profile:

- sequential model residency — only one model is kept loaded at a time
  (`keep_alive` management + unload before switching);
- Whisper, Ollama, and Kokoro never run concurrently;
- conservative worker-thread limits;
- when available RAM is critically low the user is warned — nothing is ever
  terminated automatically.

---

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` intentionally avoids torch and CUDA packages. `torch` is
> only needed if you switch Kokoro to its torch backend or use openai-whisper.

### One-time model setup

Run the app once. The **Setup Check** runs on every startup and offers to
install missing pieces with explicit Yes/No prompts:

1. **Ollama** — download the installer, then run it, then pull the chosen
   local model (the offline fallback):
   ```
   ollama pull qwen3.5:4b
   ```
   (or answer YES in the app and it will pull it for you with progress).
   Cloud models need no install — when online the app uses them
   automatically.
2. **Whisper** — `Systran/faster-whisper-*` snapshot →
   `%LOCALAPPDATA%\LocalVoiceAgent\models\whisper-<size>` (loaded by path,
   never from the network).
3. **Silero VAD** — `silero_vad.onnx` → `%LOCALAPPDATA%\LocalVoiceAgent\models`.
4. **Kokoro** — `kokoro-v1.0.onnx` + `voices-v1.0.bin` → same folder
   (`pip install kokoro-onnx` for the engine).

All downloads are one-time and user-confirmed. After that the app works fully
offline.

### Offline installs

If you already have the wheel/model/installer files, use **Settings → install
from local files** by placing them in the expected directories — no Internet
required.

---

## Run

```bat
.venv\Scripts\python main.py            REM normal launch (quiet terminal)
.venv\Scripts\python main.py --background   REM start in the tray
.venv\Scripts\python main.py --show-logs    REM stream live logs to the terminal
```

Logs always go to `%APPDATA%\LocalVoiceAgent\logs\localvoiceagent.log`;
`--show-logs` also streams them to the terminal for testing/fixing.

### Hotkeys

| Hotkey                      | Action                                                   |
| --------------------------- | -------------------------------------------------------- |
| `CTRL + WINDOWS`            | Hold = listen (push-to-talk); release = transcribe & act |
| `CTRL + WINDOWS + SPACE`    | Listen until ~800 ms of silence (auto-stop)              |
| Voice hotkey while speaking | Immediately interrupts TTS and starts listening          |

The Win key is swallowed only while it is part of an active hotkey combo, so
the Start menu does not pop open during voice use.

### States

The floating widget is a small glowing circle whose color shows what the
assistant is doing right now:

| Color | State | Meaning |
| ----- | ----- | ------- |
| 🩶 Slate grey | `IDLE` | Waiting — nothing happening |
| 🔵 Blue | `LISTENING` | Microphone recording (hold `CTRL+WIN`, or auto-stop mode) |
| 🟡 Yellow | `TRANSCRIBING` | Converting your speech to text (Whisper) |
| 🟣 Purple | `THINKING` | The local LLM is working out the reply (spinning arc) |
| 🟠 Orange | `EXECUTING_TOOL` | Running a tool — opening an app, clipboard, notes, etc. |
| 🟢 Green | `SPEAKING` | Kokoro is speaking the reply aloud |
| 🔴 Red | `ERROR` | Something failed; recovers to grey automatically |

Typical voice cycle: `IDLE → LISTENING → TRANSCRIBING → THINKING →
[EXECUTING_TOOL → THINKING] → SPEAKING → IDLE`.

What you'll usually see: **grey** while idle, **blue** while you hold the
keys and speak, a quick **yellow** flash while your words are transcribed,
then **purple** while the model thinks — on a CPU-only machine this is the
long stretch (it can last tens of seconds, since the LLM does all its work
locally). If a command needs a tool (e.g. "open notepad"), the circle
briefly turns **orange** between purple phases. When the reply is ready it
turns **green** while it's spoken, then back to **grey**. Any state can fall
to **red** (`ERROR`) and recover to grey.

---

## Model selection (local vs cloud)

Settings let you pick one **local model** (the offline fallback — at least one
is required) and one **cloud model**, both from dropdowns:

- **Local:** `qwen3.5:0.8b` / `2b` / `4b` / `9b` / `27b`, `gemma4:e2b` / `e4b` /
  `12b`, `nemotron-3-nano:4b`
- **Cloud:** `gemma4:cloud`, `gemma4:31b-cloud`, `deepseek-v4-flash:cloud`,
  `glm-5.2:cloud`, `glm-5.1:cloud`, `minimax-m2.7:cloud`, `minimax-m3:cloud`,
  `qwen3.5:cloud`, `qwen3.5:397b-cloud`, `nemotron-3-nano:30b-cloud`

At startup the app detects whether the machine is online: **online** (and
cloud models enabled) → the chosen cloud model is used; **offline** → the
chosen local model. `LLMResourceManager` keeps a single model resident at a
time.

---

## Project layout

```
main.py                     entry point
app/                        application, state machine, events, config, resources
setup/                      dependency checks + Yes/No installer + model manager
audio/                      microphone, speaker, capture/VAD orchestration
stt/                        Whisper engine + model info
vad/                        Silero VAD + pure speech-activity state machine
tts/                        Kokoro / Piper engines + TTS manager (interruption)
ollama/                     HTTP client, agent runtime, model router, LLM resources
tools/                      registry, permissions, windows, clipboard, browser,
                            notes, system, shell (disabled by default)
memory/                     SQLite (per-thread connections, WAL) + memory manager
ui/                         floating widget, tray, settings, history, wizard
hotkeys/                    global hotkeys (pynput, Start-menu suppression)
response/                   speech-friendly response processing
utils/                      logging (redacted), platform paths, CPU/RAM detection
tests/                      unit tests — no hardware or network required
config/defaults.json        LOW_RESOURCE_CPU profile defaults
build_windows.bat           PyInstaller packaging
```

Data lives in `%APPDATA%\LocalVoiceAgent` (config, DB, logs) and
`%LOCALAPPDATA%\LocalVoiceAgent` (models, cache) — nothing next to the exe.

---

## Security

- The LLM can only call explicitly registered tools with schema-validated
  arguments; it can **never** execute arbitrary Python.
- The shell tool is **disabled by default** and, when enabled, requires
  confirmation every time, has a blocklist of destructive commands, a
  timeout, and captured output.
- Tool results are never spoken as JSON; the response processor speaks
  short, conversational confirmations.
- Logging redacts secrets and never logs audio or clipboard content. The
  local log file records pipeline stages for debugging — your request
  (`[USER]`), model reply (`[MODEL RESPONSE]`), tool calls and TTS — and
  `--show-logs` streams them to the terminal. Debug audio recordings stay
  opt-in.

---

## Tests

```bat
.venv\Scripts\pip install pytest requests
.venv\Scripts\python -m pytest tests -q
```

The suite covers the state machine, config, SQLite memory (including
multi-threaded writes), response processing, model routing, LLM resource
management, VAD state transitions, tool validation/permissions, the agent
tool loop (mocked Ollama), dependency checks, and hotkey combos — with no
hardware, network, or model files required.

---

## Packaging

```bat
build_windows.bat
```

Produces `dist\LocalVoiceAgent\LocalVoiceAgent.exe` (one-dir). AI models are
**not** bundled — they live in `%LOCALAPPDATA%\LocalVoiceAgent\models`.
