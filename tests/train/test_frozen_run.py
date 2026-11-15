"""Frozen-run identity checks on a real temporary git repository (Task 14b)."""

from __future__ import annotations

import subprocess

import pytest

from gemmaloop.train.frozen import (
    FrozenRunError,
    check_frozen_run,
    git_state,
    prereg_sha256,
    run_config_sha256,
    write_run_identity,
)


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "file.txt").write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "file.txt")
    _git(tmp_path, "commit", "-m", "initial")
    _git(tmp_path, "tag", "v1")
    return tmp_path


def test_clean_tagged_head_is_accepted(repo):
    state = check_frozen_run(repo, require_tag="v1")
    assert not state.dirty
    assert state.sha == git_state(repo).sha


def test_dirty_worktree_is_refused_unless_allowed(repo):
    (repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(FrozenRunError, match="dirty"):
        check_frozen_run(repo)
    state = check_frozen_run(repo, allow_dirty=True)
    assert state.dirty


def test_require_tag_refuses_a_different_commit(repo):
    (repo / "file.txt").write_text("second\n", encoding="utf-8")
    _git(repo, "commit", "-am", "second")
    with pytest.raises(FrozenRunError, match="require-tag|HEAD is"):
        check_frozen_run(repo, require_tag="v1")


def test_missing_tag_is_refused(repo):
    with pytest.raises(FrozenRunError, match="does not exist"):
        check_frozen_run(repo, require_tag="nope")


def test_prereg_and_config_hashes(tmp_path):
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text("training: {max_length: 8192}\n", encoding="utf-8")
    assert len(prereg_sha256(prereg)) == 64
    assert prereg_sha256(None) is None
    assert run_config_sha256({"a": 1, "b": 2}) == run_config_sha256({"b": 2, "a": 1})
    with pytest.raises(FrozenRunError):
        prereg_sha256(tmp_path / "missing.yaml")


def test_run_identity_is_written_atomically(tmp_path):
    path = write_run_identity(tmp_path, {"git_sha": "abc", "git_dirty": False})
    assert path.name == "run_identity.json"
    assert '"git_sha": "abc"' in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_tag_and_allow_dirty_are_incompatible(repo):
    with pytest.raises(FrozenRunError, match="incompatible"):
        check_frozen_run(repo, allow_dirty=True, require_tag="v1")


def test_prereg_requires_a_tag(repo):
    prereg = repo / "prereg.yaml"
    prereg.write_text("x: 1\n", encoding="utf-8")
    _git(repo, "add", "prereg.yaml")
    _git(repo, "commit", "-m", "prereg")
    _git(repo, "tag", "-f", "v1")
    with pytest.raises(FrozenRunError, match="requires --require-tag"):
        check_frozen_run(repo, prereg=prereg)
    assert check_frozen_run(repo, prereg=prereg, require_tag="v1")


def test_imported_code_must_live_in_the_checked_worktree(repo):
    from gemmaloop.train.frozen import assert_code_under

    source = repo / "src" / "gemmaloop" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    assert assert_code_under(repo, module_file=source) == source.resolve()
    outside = repo / "elsewhere" / "gemmaloop" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    with pytest.raises(FrozenRunError, match="outside this worktree"):
        assert_code_under(repo, module_file=outside)


def test_run_identity_hash_depends_on_all_four_fields():
    from gemmaloop.train.frozen import run_identity_hash

    base = dict(
        run_config_sha256="cfg",
        data_manifest_sha256="data",
        git_sha="sha",
        seed=0,
    )
    value = run_identity_hash(**base)
    assert run_identity_hash(**base) == value
    for field, replacement in (
        ("run_config_sha256", "cfg2"),
        ("data_manifest_sha256", "data2"),
        ("git_sha", "sha2"),
        ("seed", 1),
    ):
        changed = dict(base)
        changed[field] = replacement
        assert run_identity_hash(**changed) != value
