from setup import dependency_checks as checks
from setup.dependency_checks import DependencyStatus
from setup.dependency_manager import DependencyManager


def test_python_and_sqlite_always_present():
    assert checks.check_python().present
    assert checks.check_sqlite().present


def test_ollama_detection_offline(monkeypatch):
    import requests

    class FakeResp:
        status_code = 200

        def json(self):
            return {"version": "0.5.0"}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    status = checks.check_ollama("http://127.0.0.1:11434")
    assert status.present is True
    assert status.version == "0.5.0"


def test_ollama_model_check_with_client(monkeypatch):
    class FakeClient:
        def model_installed(self, name):
            return name == "qwen3.5:4b"

        def model_info(self, name):
            return {"size": 2 * 1024**3} if name == "qwen3.5:4b" else {}

    present = checks.check_ollama_model(FakeClient(), "qwen3.5:4b")
    assert present.present is True
    assert "2.0 GB" in present.version

    missing = checks.check_ollama_model(FakeClient(), "qwen3.5:9b")
    assert missing.present is False


def test_missing_statuses_filters():
    statuses = [
        DependencyStatus("a", checks.CORE, True),
        DependencyStatus("b", checks.CORE, False),
        DependencyStatus("c", checks.OPTIONAL, False),
    ]
    missing = checks.missing_statuses(statuses)
    assert [s.name for s in missing] == ["b", "c"]


def test_resolve_skips_when_no_prompter(tmp_path):
    # Headless mode: without a prompt callback nothing is installed.
    manager = DependencyManager(models_dir=tmp_path)
    statuses = [
        DependencyStatus("PySide6", checks.CORE, False, installable=True),
    ]
    result = manager.resolve(statuses)
    assert result is False
    assert "PySide6" in manager._skip_this_startup


def test_dependency_manager_plans(tmp_path):
    manager = DependencyManager(models_dir=tmp_path)
    plan = manager._plan_for(checks.check_ollama())
    assert plan is not None
    assert plan.internet_required is True


def test_dependency_manager_local_model_plan_and_size(tmp_path):
    manager = DependencyManager(models_dir=tmp_path)
    plan = manager._plan_for(checks.check_ollama_model(None, "qwen3.5:4b"))
    assert plan is not None
    assert plan.approx_size_mb == 4096

    plan = manager._plan_for(checks.check_ollama_model(None, "gemma4:e4b"))
    assert plan is not None
    assert plan.approx_size_mb == 4096

    plan = manager._plan_for(checks.check_ollama_model(None, "qwen3.5:27b"))
    assert plan is not None
    assert plan.approx_size_mb == 27 * 1024


def test_dependency_manager_ignores_cloud_models(tmp_path):
    manager = DependencyManager(models_dir=tmp_path)
    plan = manager._plan_for(checks.check_ollama_model(None, "deepseek-v4-flash:cloud"))
    assert plan is None


def test_run_all_checks_cloud_mode_does_not_require_local_ollama(tmp_path):
    from app.config import Settings

    settings = Settings(path=tmp_path / "config.json")
    statuses = checks.run_all_checks(settings, client=None,
                                     cloud_model="deepseek-v4-flash:cloud")
    ollama = [s for s in statuses if s.name == "Ollama"]
    assert ollama and ollama[0].present is True
    # The local server must not appear as a required missing dependency.
    missing = checks.missing_statuses(statuses)
    assert all(s.name != "Ollama" for s in missing)
    # The cloud model itself is reported as satisfied.
    assert any(s.name.startswith("Cloud model") and s.present for s in statuses)


def test_check_ollama_cloud_signin_status():
    class FakeProbe:
        def __init__(self, status):
            self._status = status

        def probe_cloud(self, model):
            return self._status

    ok = checks.check_ollama_cloud(FakeProbe("ok"), "deepseek-v4-flash:cloud")
    assert ok.present is True
    assert "signed in" in ok.message.lower()

    unauth = checks.check_ollama_cloud(FakeProbe("unauthorized"), "deepseek-v4-flash:cloud")
    assert unauth.present is False
    assert "signin" in unauth.message.lower()

    err = checks.check_ollama_cloud(FakeProbe("error"), "deepseek-v4-flash:cloud")
    assert err.present is False


def test_run_all_checks_reports_cloud_signin(tmp_path):
    from app.config import Settings

    class SignedOutClient:
        def probe_cloud(self, model):
            return "unauthorized"

        def model_installed(self, name):
            return False

        def model_info(self, name):
            return {}

    settings = Settings(path=tmp_path / "config.json")
    statuses = checks.run_all_checks(settings, client=SignedOutClient(),
                                     cloud_model="deepseek-v4-flash:cloud")
    cloud = [s for s in statuses if s.name.startswith("Cloud model")]
    assert cloud and cloud[0].present is False
    assert "signin" in cloud[0].message.lower()


def test_category_values():
    assert checks.CORE == "core"
    assert checks.AI_CORE == "ai_core"
    assert checks.VOICE_INPUT == "voice_input"


def test_check_whisper_model_uses_local_models_dir(tmp_path):
    missing = checks.check_whisper_model(models_dir=tmp_path, model="small")
    assert missing.present is False

    model_dir = tmp_path / "whisper-small"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"fake")

    present = checks.check_whisper_model(models_dir=tmp_path, model="small")
    assert present.present is True


def test_check_whisper_model_ignores_incomplete_download(tmp_path):
    # A partial snapshot (dir exists, model.bin missing) is still "missing".
    (tmp_path / "whisper-small").mkdir()
    status = checks.check_whisper_model(models_dir=tmp_path, model="small")
    assert status.present is False


def test_check_kokoro_requires_backend_package(tmp_path, monkeypatch):
    # Model files alone are not enough — the TTS backend must be installed
    # or KokoroEngine fails silently at runtime.
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"x")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"x")

    monkeypatch.setattr(checks, "_module_present", lambda m: m == "onnxruntime")
    status = checks.check_kokoro(tmp_path)
    assert status.present is False
    assert "backend" in status.message.lower()

    monkeypatch.setattr(checks, "_module_present", lambda m: m in ("kokoro_onnx", "onnxruntime"))
    assert checks.check_kokoro(tmp_path).present is True
