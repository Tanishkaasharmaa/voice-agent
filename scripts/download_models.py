"""Helper script to pre-download all required offline model weights.

Downloads to %LOCALAPPDATA%/LocalVoiceAgent/models:
1. Silero VAD (voice activity detection)
2. Kokoro ONNX + voices.bin (text-to-speech)
3. Faster-Whisper Small (speech-to-text)
"""

import os
import sys
from pathlib import Path
import requests
from huggingface_hub import snapshot_download

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.platform import ensure_directories

dirs = ensure_directories()
models_dir = dirs["models"]
print(f"Target models folder: {models_dir}")


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already installed: {dest.name} ({round(dest.stat().st_size / (1024*1024), 1)} MB)")
        return
    print(f"Downloading {dest.name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        pct = (done / total) * 100
                        print(f"\r{dest.name}: {pct:.1f}% ({done / (1024*1024):.1f}/{total / (1024*1024):.1f} MB)", end="", flush=True)
    tmp.replace(dest)
    print(f"\nDone: {dest.name}")


def main() -> None:
    print("=" * 60)
    print("LocalVoiceAgent — Model Downloader")
    print("=" * 60)

    # 1. Silero VAD
    print("\n[1/4] Silero VAD (~2 MB)...")
    download(
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        models_dir / "silero_vad.onnx",
    )

    # 2. Kokoro ONNX
    print("\n[2/4] Kokoro ONNX model (~310 MB)...")
    download(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        models_dir / "kokoro-v1.0.onnx",
    )

    # 3. Kokoro voices
    print("\n[3/4] Kokoro voice definitions (~27 MB)...")
    download(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        models_dir / "voices-v1.0.bin",
    )

    # 4. Whisper Small
    print("\n[4/4] Faster-Whisper Small snapshot (~485 MB)...")
    whisper_dir = models_dir / "whisper-small"
    if (whisper_dir / "model.bin").exists():
        print(f"Already installed: whisper-small at {whisper_dir}")
    else:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        snapshot_download(
            repo_id="Systran/faster-whisper-small",
            local_dir=whisper_dir,
        )
        print("Done: whisper-small")

    print("\n" + "=" * 60)
    print("All offline model weights are downloaded and ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
