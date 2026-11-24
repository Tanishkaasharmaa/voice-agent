"""Tests for the internet detection used to decide cloud vs local models."""

import sys
import types

import utils.system as system_mod


class FakeSession:
    """Stands in for requests.Session; records probed URLs and can fail
    selected endpoints or whole rounds."""

    def __init__(self, fail_urls=(), fail_forever=False):
        self.fail_urls = set(fail_urls)
        self.fail_forever = fail_forever
        self.probed = []
        self.closed = False

    def head(self, url, timeout=None, allow_redirects=False):
        self.probed.append(url)
        if self.fail_forever or url in self.fail_urls:
            raise OSError("unreachable")
        return True  # any HTTP response counts as online

    def close(self):
        self.closed = True


def install_fake_session(monkeypatch, fake):
    module = types.ModuleType("requests")
    module.Session = lambda: fake
    monkeypatch.setitem(sys.modules, "requests", module)


def test_first_endpoint_success(monkeypatch):
    fake = FakeSession()
    install_fake_session(monkeypatch, fake)
    assert system_mod.has_internet(rounds=2) is True


def test_falls_through_to_second_endpoint(monkeypatch):
    first, second = system_mod._INTERNET_PROBES[0], system_mod._INTERNET_PROBES[1]
    fake = FakeSession(fail_urls={first})
    install_fake_session(monkeypatch, fake)
    assert system_mod.has_internet(endpoints=(first, second), rounds=1) is True
    assert second in fake.probed


def test_all_endpoints_down_is_offline(monkeypatch):
    fake = FakeSession(fail_forever=True)
    install_fake_session(monkeypatch, fake)
    assert system_mod.has_internet(rounds=2) is False
    assert fake.closed is True


def test_retry_round_succeeds_after_transient_failure(monkeypatch):
    # All endpoints fail on the first sweep, succeed afterwards.
    calls = {"n": 0}

    class FlakySession(FakeSession):
        def head(self, url, timeout=None, allow_redirects=False):
            calls["n"] += 1
            if calls["n"] <= len(system_mod._INTERNET_PROBES):
                raise OSError("not ready yet")
            return True

    fake = FlakySession()
    install_fake_session(monkeypatch, fake)
    assert system_mod.has_internet(rounds=3) is True


def test_missing_requests_library_is_offline(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def no_requests(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("no requests")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_requests)
    assert system_mod.has_internet() is False
