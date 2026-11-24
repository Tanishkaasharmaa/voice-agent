# AGENTS.md

LocalVoiceAgent: fully offline, Windows-only, CPU-only voice agent (PySide6 UI). Target hardware is a 16 GB RAM laptop with no GPU — every change must respect that budget.

## Commands

```bat
REM Tests (fast, ~4 s, no hardware/network/model files required)
.venv\Scripts\python -m pytest tests -q

REM Single file
.venv\Scripts\python -m pytest tests\test_config.py -q

REM Run the app (opens Qt widget, grabs global hotkeys and mic — do not launch casually)
.venv\Scripts\python main.py --show-logs

REM Package (requires: .venv\Scripts\pip install pyinstaller — NOT in requirements.txt)
build_windows.bat
```

There is no lint/typecheck config in this repo — don't invent one.

## Hard constraints

- **Never add torch or CUDA dependencies.** The stack deliberately avoids them: faster-whisper (CTranslate2 int8) instead of openai-whisper, Silero VAD and Kokoro via ONNX Runtime. `kokoro-onnx` and `piper-tts` are also absent from `requirements.txt`.
- Heavy/optional deps (`kokoro_onnx`, `piper`) are imported lazily *inside methods*, so imports never fail and tests run without them. Keep this pattern when adding engine backends.
- Windows-only code throughout (pywin32, pynput win32 backends); paths use `%APPDATA%\LocalVoiceAgent` (config, DB, logs) and `%LOCALAPPDATA%\LocalVoiceAgent\models`. Nothing is stored next to the source or exe.

## Architecture

Entry point `main.py` → `VoiceAgentApplication` (`app/application.py`), which wires everything:

- Pipeline is driven by `EventBus` (`app/events.py`) + `StateMachine` (`app/state_machine.py`): `IDLE → LISTENING → TRANSCRIBING → THINKING → [EXECUTING_TOOL] → SPEAKING`.
- STT/VAD/TTS/audio managers are created as `None` and lazy-loaded on first use — don't assume they exist at startup.
- `LLMResourceManager` keeps exactly one model resident at a time (unload before switching). Whisper/Ollama/Kokoro never run concurrently.
- LLM can only invoke tools registered in `ToolRegistry` (`tools/registry.py`) with schema-validated args. Shell tool is disabled by default (`tools.shell_enabled`). Don't bypass permissions/confirmation.
- Config: `config/defaults.json` is deep-merged under user's `%APPDATA%\LocalVoiceAgent\config.json`; models load by local path, never from the network at runtime.

## UI layer (`ui/`)

- `FloatingWidget` is the floating island (82x24, frameless translucent `Qt.Tool`). Hover reveals gear (Settings) and X (quit) buttons; double-click emits `auto_listen_requested` → continuous auto-stop listening. Drag emits `position_changed`; hover survives spurious leave events via a 250 ms grace re-check against `QCursor.pos`.
- `CharacterWidget` (`ui/character_widget.py`) is a decorative chibi character window above the island. **It sets `WA_TransparentForMouseEvents` — never remove this**, or it steals hover/clicks from the island. `CharacterAnimator` is pure logic (Qt-free) so it stays unit-testable; hand poses ease toward per-state targets.
- Settings dialog uses a sidebar (`QListWidget#settingsSidebar` + `QStackedWidget`), not tabs.
- Cross-thread UI updates go through `_UiBridge` queued signals (`app/application.py`) — bus handlers run on the publisher's thread.

## TTS / lip sync

- `Speaker` writes playback in 100 ms chunks and emits per-chunk RMS via `on_amplitude` → `TTSManager.on_amplitude` → `TOPIC_TTS_AMPLITUDE` on the bus → `_UiBridge.amplitudeChanged` → character mouth.
- `TTSManager._pending` counts accepted-but-not-played sentences; `is_busy()` = pending or speaker playing. The pipeline thread (`_process_utterance`) stays in SPEAKING until `is_busy()` drains — don't shortcut it to IDLE when `agent.run()` returns.

## Internet / cloud detection

- `utils/system.has_internet()` probes multiple endpoints with retries (single-host single-attempt checks caused false "offline" at startup).
- If cloud models are enabled but startup detection fails, a daemon thread (`net-recheck`, stopped by `_exit`) re-probes every 2 s for ~60 s and switches to the cloud model live via `_on_online_ui`.

## VAD / auto-stop

- If the Silero model is missing or inference fails repeatedly (25 frames), `AudioManager` fails over to `EnergyVoiceGate` (`vad/base.py`) — an adaptive-RMS gate fed into the same `SpeechActivityDetector` timing machine. Never leave auto_stop mode without a working detector: it would listen until the 60 s max duration.
- `EnergyVoiceGate` classifies via threshold + envelope statistics: only STATIONARY windows (low CV *and* small max-min spread) may move the noise floor — speech never raises it, so long replies can't make it go deaf. A 1.5 s grace in `_auto_stop_utterance` suppresses calibration-time false ends.
- `SpeechActivityDetector.feed` advances its own virtual clock (`frame_duration_s` per call); pass explicit timestamps only in tests.

## Testing notes

- Root `conftest.py` adds the project root to `sys.path`, so packages import as top-level (`from app.config import ...`) even when pytest runs from elsewhere.
- Ollama interactions are mocked in agent/client tests; no live server is ever needed.
