"""The Modal image must provide every third-party import in the remote closure.

Incident 2026-11-05: the remote script imported ``gemmaloop.stage0.pool`` at
module level, which reached ``gemmaloop.data.decontam`` and ``rapidfuzz``.  The
image never installed rapidfuzz, so every container died during startup while
the launcher waited.  This test walks the script's AST (never executing the
remote path) and checks module-level imports, their transitive ``gemmaloop``
imports, and the imports inside the remote function bodies.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "modal_stage0.py"
SRC = ROOT / "src"

# Kept local so the test itself still runs against the pre-fix module (which has
# only ``sample_shard``); preflight is the CPU-only remote function added by the
# fix and must be covered once it exists.
REMOTE_FUNCTION_NAMES = ("sample_shard", "preflight_check")

# Modal image pip specs use distribution names; Python imports use module names.
DISTRIBUTION_TO_IMPORT = {
    "huggingface_hub": "huggingface_hub",
    "vllm": "vllm",
    "transformers": "transformers",
}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "modal_stage0_image_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _module_file(module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    for candidate in (
        SRC / relative.with_suffix(".py"),
        SRC / relative / "__init__.py",
    ):
        if candidate.exists():
            return candidate
    return None


def _resolve_relative(
    current_module: str, level: int, module: str | None, *, is_package: bool
) -> str:
    package = current_module.split(".")
    # For level=1, a normal module drops its own name; a package __init__ drops
    # nothing because the current module is already the package.
    drop = level - (1 if is_package else 0)
    if drop:
        package = package[:-drop]
    if module:
        package.extend(module.split("."))
    return ".".join(part for part in package if part)


class _ModuleLevelImports(ast.NodeVisitor):
    """Collect imports that execute when the module is imported.

    Function and class bodies are skipped: they only run when called.
    """

    def __init__(
        self,
        current_module: str,
        *,
        is_package: bool = False,
        include_functions: bool = False,
    ) -> None:
        self.current_module = current_module
        self.is_package = is_package
        self.include_functions = include_functions
        self.modules: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.modules.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = _resolve_relative(
                self.current_module,
                node.level,
                node.module,
                is_package=self.is_package,
            )
        else:
            base = node.module or ""
        if base:
            self.modules.add(base)
        # ``from . import sources`` imports submodules named by the aliases.
        if node.module is None and base:
            for alias in node.names:
                self.modules.add(f"{base}.{alias.name}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if self.include_functions:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        if self.include_functions:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        # A class body executes at import time; methods are still skipped by
        # visit_FunctionDef.
        self.generic_visit(node)


def _imports_in_file(
    path: Path, module_name: str, *, include_functions: bool = False
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    visitor = _ModuleLevelImports(
        module_name,
        is_package=path.name == "__init__.py",
        include_functions=include_functions,
    )
    visitor.visit(tree)
    return visitor.modules


def _package_chain(module_name: str) -> list[str]:
    parts = module_name.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts) + 1)]


def _transitive_gemmaloop_imports(roots: set[str]) -> tuple[set[str], set[str]]:
    seen: set[str] = set()
    every_import: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        for name in _package_chain(module) if module.startswith("gemmaloop") else [module]:
            if name in seen or not name.startswith("gemmaloop"):
                continue
            path = _module_file(name)
            if path is None:
                continue
            seen.add(name)
            imported = _imports_in_file(path, name, include_functions=True)
            every_import |= imported
            pending.extend(
                child
                for child in imported
                if child.startswith("gemmaloop") and child not in seen
            )
    return seen, every_import


def _function_defs(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_functions(node: ast.AST, known: set[str]) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            if child.func.id in known:
                names.add(child.func.id)
    return names


def _closure_imports(tree: ast.Module, roots: set[str]) -> set[str]:
    """Follow module-level helper calls transitively from the remote functions."""

    functions = _function_defs(tree)
    pending = [name for name in roots if name in functions]
    seen: set[str] = set()
    imports: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        node = functions[name]
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                imports.update(alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    # The script is not a package; remote imports are absolute.
                    continue
                if child.module:
                    imports.add(child.module)
        pending.extend(_called_functions(node, set(functions)) - seen)
    return imports


def _remote_closure_imports(tree: ast.Module) -> set[str]:
    return _closure_imports(tree, set(REMOTE_FUNCTION_NAMES))


def _third_party(modules: set[str]) -> set[str]:
    stdlib = set(sys.stdlib_module_names)
    # Modal's own client is provided by the Modal runtime, not by pip.
    return {
        module.split(".")[0]
        for module in modules
        if module.split(".")[0] not in stdlib
        and module.split(".")[0] not in {"gemmaloop", "__future__", "modal"}
    }


def _image_import_names(script) -> set[str]:
    pip_specs: list[str] = []

    class FakeImage:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        @classmethod
        def from_registry(cls, tag, **kwargs):
            return cls()

        def apt_install(self, *args):
            return self

        def pip_install(self, *args):
            pip_specs.extend(args)
            return self

        def env(self, values):
            return self

        def add_local_python_source(self, *modules):
            return self

    class FakeApp:
        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            return lambda fn: fn

    script._build_modal_app(SimpleNamespace(App=FakeApp, Image=FakeImage))

    names: set[str] = set()
    for spec in pip_specs:
        distribution = re.split(r"[\[<>=!~; ]", spec, maxsplit=1)[0]
        normalized = distribution.lower().replace("-", "_")
        names.add(DISTRIBUTION_TO_IMPORT.get(normalized, normalized))
    return names


def _third_party_for_tree(
    tree: ast.Module, roots: set[str], *, module_level: set[str] | None = None
) -> set[str]:
    imports = set(module_level or set()) | _closure_imports(tree, roots)
    _, transitive = _transitive_gemmaloop_imports(set(imports))
    return _third_party(imports | transitive)


def test_remote_import_closure_is_provided_by_the_image():
    script = _load_script()
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    module_level = _ModuleLevelImports("modal_stage0")
    module_level.visit(tree)

    remote_imports = _remote_closure_imports(tree)
    imported = set(module_level.modules) | remote_imports
    _, transitive = _transitive_gemmaloop_imports(imported)
    third_party = _third_party(imported | transitive)
    image_imports = _image_import_names(script)
    missing = sorted(third_party - image_imports)

    assert not missing, (
        "remote import closure needs packages missing from the Modal image: "
        f"{missing}; image provides {sorted(image_imports)}"
    )
    assert "rapidfuzz" not in third_party


def test_preflight_imports_exactly_the_gpu_closure():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    module_level = _ModuleLevelImports("modal_stage0")
    module_level.visit(tree)
    level = set(module_level.modules)
    gpu = _third_party_for_tree(
        tree, {"sample_shard"}, module_level=level
    )
    preflight = _third_party_for_tree(
        tree, {"preflight_check"}, module_level=level
    )
    assert preflight == gpu
    assert gpu == {"huggingface_hub", "transformers", "vllm"}


def test_ast_walker_catches_lazy_remote_imports_and_local_pool():
    source = SCRIPT.read_text(encoding="utf-8")
    needle = (
        "def _cached_engine(LLM: Any, model_path: str, max_model_len: int):\n"
    )
    mutated = source.replace(
        needle,
        needle
        + "    import flash_attn\n"
        + "    from gemmaloop.stage0.pool import PART1_CREATIVE_BRIEFS\n",
        1,
    )
    assert mutated != source
    tree = ast.parse(mutated)
    third_party = _third_party_for_tree(tree, {"sample_shard"})
    image_imports = _image_import_names(_load_script())
    missing = third_party - image_imports
    assert "flash_attn" in missing
    assert "rapidfuzz" in missing


def test_gemmaloop_function_body_lazy_import_is_followed(tmp_path, monkeypatch):
    fake = tmp_path / "modal_plan.py"
    fake.write_text(
        "def remote_helper():\n"
        "    import rapidfuzz\n"
        "    return rapidfuzz\n",
        encoding="utf-8",
    )
    original = _module_file

    def resolver(name: str) -> Path | None:
        if name == "gemmaloop.stage0.modal_plan":
            return fake
        return original(name)

    monkeypatch.setattr(
        sys.modules[__name__], "_module_file", resolver
    )
    _, imported = _transitive_gemmaloop_imports(
        {"gemmaloop.stage0.modal_plan"}
    )
    assert "rapidfuzz" in imported
