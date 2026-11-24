# LocalVoiceAgent — Setup & Troubleshooting Guide

This guide walks you through setting up and running **LocalVoiceAgent** on a fresh Windows PC from a zip archive, along with all the common pitfalls you might encounter and how to solve them immediately.

---

## 1. System Requirements

- **OS:** Windows 10 or Windows 11 (64-bit)
- **Hardware:** CPU-only (no GPU needed); 8 GB to 16 GB RAM
- **Audio:** Working microphone and speaker / headphones
- **Software Prerequisites:**
  1. **Python 3.10, 3.11, or 3.12** ([python.org](https://www.python.org/downloads/))
     > ⚠️ **Important:** During Python installation, check the box **"Add python.exe to PATH"**.
  2. **Ollama for Windows** ([ollama.com/download](https://ollama.com/download))

---

## 2. Quick-Start Setup (5 Steps)

### Step 1: Extract the Zip
Extract the zip folder to a location on your computer (e.g. `C:\Projects\LocalVoiceAgent` or your `Documents` folder).

### Step 2: Open a Terminal & Create the Virtual Environment
Open **PowerShell** or **Command Prompt** inside the project folder:
```bat
python -m venv .venv
```

Activate the virtual environment:
- In Command Prompt (`cmd.exe`):
  ```bat
  .venv\Scripts\activate.bat
  ```
- In PowerShell:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(If PowerShell blocks scripts, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Python Dependencies
Install the required CPU packages and TTS engines:
```bat
pip install -r requirements.txt kokoro-onnx soundfile
```

### Step 4: Start Ollama & Download the Local Model
Make sure the Ollama app is running (you should see its llama icon in your Windows taskbar tray, or run `ollama serve` in a separate window).

Download the recommended lightweight local model:
```bat
ollama pull qwen2.5:1.5b
```
> 💡 *Why `qwen2.5:1.5b`?* It only takes ~1.2 GB of RAM, runs fast on CPUs without GPU, and has native function/tool-calling support. Larger models (like 9B) require 8+ GB of free RAM and may fail to allocate on laptops.

### Step 5: Download the Offline AI Models
The app requires 3 small offline model files (Whisper speech-to-text, Silero voice activity detection, and Kokoro text-to-speech). Run the included downloader:
```bat
python scripts/download_models.py
```
This downloads the weights (~820 MB total) directly into your local app data folder (`%LOCALAPPDATA%\LocalVoiceAgent\models`). Once downloaded, the agent is **100% offline**.

---

## 3. Running the Agent

Run the app with live logs enabled:
```bat
.venv\Scripts\python main.py --show-logs
```

### Hotkeys & Controls

| Action | Control | What it does |
|---|---|---|
| **Push-to-Talk** | **Hold `Ctrl + Win`** | Microphone records while held. Release to transcribe & execute. |
| **Auto-Stop Listening** | **`Ctrl + Win + Space`** | Starts listening and automatically stops when you pause (~800 ms of silence). |
| **Interrupt Speech** | **`Ctrl + Win` while speaking** | Immediately cuts off the assistant's voice and starts listening to you. |
| **Settings** | **Hover over island → ⚙️** | Opens settings (change assistant name, microphone, local model, theme). |
| **Quit** | **Hover over island → ❌** | Exits the application cleanly. |

### Visual Feedback (Status Colors)
The floating widget displays its current state with glowing colors:
- 🩶 **Grey:** `IDLE` (waiting for you)
- 🔵 **Blue:** `LISTENING` (microphone recording)
- 🟡 **Yellow:** `TRANSCRIBING` (Whisper converting audio to text)
- 🟣 **Purple:** `THINKING` (LLM reasoning / deciding on tools)
- 🟠 **Orange:** `EXECUTING_TOOL` (running a tool like opening an app or searching notes)
- 🟢 **Green:** `SPEAKING` (Kokoro speaking the response aloud)
- 🔴 **Red:** `ERROR` (recovers to grey automatically)

---

## 4. Problems Faced & How to Fix Them

### Problem 1: The assistant doesn't hear me (`[USER] ` is empty in logs)
- **Symptom:** You speak while holding `Ctrl+Win`, but the terminal log shows `[USER] ` with blank text, and the assistant never replies.
- **Cause:** Windows routed audio to a muted or disconnected microphone (e.g. laptop internal mic array instead of your active headset/earbuds).
- **Solution:**
  1. Hover over the floating widget and click the ⚙️ **Settings** gear icon.
  2. Select **Voice Input** on the left.
  3. Under **Microphone**, choose your active headset/mic (e.g. `Headset (OnePlus Nord Buds)` or `Microphone (Realtek)`).
  4. Click **Save**. Alternatively, set your headset as the **Default Input Device** in **Windows Settings → System → Sound**.

---

### Problem 2: Out of Memory / Ollama error 500
- **Symptom:**
  ```
  Ollama chat error (500): {"error":"model requires more system memory (7.9 GiB) than is available (5.6 GiB)"}
  ```
- **Cause:** The active model (e.g. a 9B or larger model) requires more free RAM than your PC currently has available.
- **Solution:**
  Switch to `qwen2.5:1.5b`:
  1. Run `ollama pull qwen2.5:1.5b` in your terminal.
  2. In the app's **Settings → LLM**, select `qwen2.5:1.5b` under **Local model**, or edit `%APPDATA%\LocalVoiceAgent\config\config.json` to have `"local_model": "qwen2.5:1.5b"`.

---

### Problem 3: Setup Check dialog shows a red cross for `qwen3.5:4b`
- **Symptom:** On first launch, a dialog appears stating `Required component missing: qwen3.5:4b` with `[YES]` and `[NO]` buttons.
- **Cause:** The default configuration template checked for `qwen3.5:4b`, which requires a bleeding-edge Ollama release.
- **Solution:**
  - Click **`NO`** (or **`Skip`** on the setup window).
  - Open **Settings** → change the local model to `qwen2.5:1.5b` (or whichever local model you pulled).
  - Once set, all checks will turn green and the prompt will never appear again.

---

### Problem 4: Rapid hotkey presses crash the app (`InvalidTransition`)
- **Symptom:**
  ```
  InvalidTransition: TRANSCRIBING -> LISTENING is not allowed
  ```
- **Cause:** In older versions, pressing the listen hotkey again while Whisper was still finishing transcription threw an unhandled transition error.
- **Solution:**
  This is **already patched** in this codebase. The state machine now recognizes rapid hotkey presses as valid user interruptions and settles cleanly.

---

### Problem 5: Ollama connection refused (`127.0.0.1:11434`)
- **Symptom:**
  ```
  Ollama not detected / Connection refused
  ```
- **Cause:** The Ollama service is not running in the background.
- **Solution:**
  Start Ollama from the Windows Start menu, or run `ollama serve` in a terminal before starting LocalVoiceAgent.

---

### Problem 6: Can't click through the character window
- **Symptom:** Clicks near the floating island get blocked by the character.
- **Solution:**
  The character window sets `Qt.WA_TransparentForMouseEvents` so all clicks pass straight through to windows behind it. Never disable this attribute in `ui/character_widget.py`.

---

## 5. File & Folder Locations on Windows

All user data and models are kept strictly separate from the source code:
- **Settings, Database, and Logs:**
  `%APPDATA%\LocalVoiceAgent\` (`config\config.json`, `data\localvoiceagent.db`, `logs\localvoiceagent.log`)
- **AI Models & Cache:**
  `%LOCALAPPDATA%\LocalVoiceAgent\models\` (`whisper-small\`, `silero_vad.onnx`, `kokoro-v1.0.onnx`, `voices-v1.0.bin`)
