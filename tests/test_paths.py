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
