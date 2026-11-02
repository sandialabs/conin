"""The import rules of ``doc/sphinx/architecture.rst`` (section Layering).

The imports are read from the source with ``ast`` (tests are not checked), so
these tests see every import, including the ones behind ``try_import``.
"""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import conin

PACKAGE_DIR = pathlib.Path(conin.__file__).parent

# Each unit may import the units of the layers below its own.  The units of
# the base and model layers may also import each other: the model layer's
# packages refer to each other at package level, though not in a module-level
# cycle (see test_no_module_level_import_cycles).
LAYERS = [
    {"exceptions", "util", "config", "__about__"},
    {"graphical_models", "constraints"},
    {"pgmpy"},
    {"constrained_models", "representations", "io"},
    {"algorithms.util"},
    {"algorithms.feasibility"},
    {"algorithms.scoring"},
    {"algorithms.inference"},
    {"algorithms.sampling"},
    {"algorithms.learning"},
    {"examples", "algorithms", "conin"},
]
SHARED_LAYERS = (0, 3)

# The modules that need torch.  Only they may be imported inside a function,
# so that ``import conin`` does not import torch.
TORCH_MODULES = (
    "conin.algorithms.util.mvr_kernels",
    "conin.algorithms.inference.mvr.",
    "conin.algorithms.learning.mvr.",
    "conin.algorithms.sampling.mvr.",
)


def _modules():
    """Map each non-test module of conin to its source file."""
    modules = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
        if "tests" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


MODULES = _modules()


def _is_package(name):
    return MODULES[name].name == "__init__.py"


def _imports(name):
    """Yield ``(target, lineno, function_level)`` for each conin import of a module.

    ``from X import y`` imports ``X.y`` if that is a module, else ``X``.
    """
    tree = ast.parse(MODULES[name].read_text(), filename=str(MODULES[name]))
    package = name if _is_package(name) else name.rpartition(".")[0]
    in_function = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_function.update(id(n) for n in ast.walk(node) if n is not node)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            targets = [
                (
                    f"{module}.{alias.name}"
                    if f"{module}.{alias.name}" in MODULES
                    else module
                )
                for alias in node.names
            ]
        else:
            continue
        for target in targets:
            if target == "conin" or target.startswith("conin."):
                yield target, node.lineno, id(node) in in_function


def _unit(name):
    """``"graphical_models"`` for ``conin.graphical_models.x``; ``algorithms`` by subpackage."""
    parts = name.split(".")
    if len(parts) == 1:
        return "conin"
    if parts[1] == "algorithms" and len(parts) > 2:
        return f"algorithms.{parts[2]}"
    return parts[1]


def _layer(unit):
    for index, layer in enumerate(LAYERS):
        if unit in layer:
            return index
    raise AssertionError(f"{unit!r} is in no layer of LAYERS; add it")


def test_every_unit_is_in_a_layer():
    for name in MODULES:
        _layer(_unit(name))


def test_imports_follow_the_layers():
    violations = []
    for name in sorted(MODULES):
        importer = _unit(name)
        for target, lineno, _ in _imports(name):
            imported = _unit(target)
            if imported == importer:
                continue
            a, b = _layer(importer), _layer(imported)
            if b < a or (a == b and a in SHARED_LAYERS):
                continue
            violations.append(f"{name}:{lineno} ({importer}) imports {target}")
    assert violations == []


def test_function_level_imports_are_only_for_torch():
    violations = [
        f"{name}:{lineno} imports {target} inside a function"
        for name in sorted(MODULES)
        for target, lineno, function_level in _imports(name)
        if function_level and not target.startswith(TORCH_MODULES)
    ]
    assert violations == []


def test_no_module_level_import_cycles():
    # Importing a.b.c also runs the packages a and a.b, unless the importer
    # is inside them (then they are already being imported).
    graph = {}
    for name in MODULES:
        own = {name.rsplit(".", k)[0] for k in range(1, name.count(".") + 1)}
        edges = set()
        for target, _, function_level in _imports(name):
            if function_level:
                continue
            parts = target.split(".")
            for k in range(1, len(parts) + 1):
                prefix = ".".join(parts[:k])
                if prefix != name and prefix not in own and prefix in MODULES:
                    edges.add(prefix)
        graph[name] = edges

    # Tarjan's strongly connected components; a component with more than one
    # module is a cycle.
    index, low, stack, on_stack, cycles = {}, {}, [], set(), []

    def visit(v):
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(graph)))
    try:
        for v in sorted(graph):
            if v not in index:
                visit(v)
    finally:
        sys.setrecursionlimit(limit)
    assert cycles == []


def test_import_conin_does_not_import_torch():
    # pgmpy imports torch itself when it is installed, so it is blocked here
    # (conin imports it with try_import).  The MVR packages are imported too:
    # they load their algorithms lazily.
    script = textwrap.dedent("""
        import sys

        class BlockPgmpy:
            def find_spec(self, name, path=None, target=None):
                if name == "pgmpy" or name.startswith("pgmpy."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, BlockPgmpy())

        import conin
        import conin.algorithms.inference.mvr
        import conin.algorithms.learning.mvr
        import conin.algorithms.sampling.mvr
        import conin.algorithms.util
        import conin.representations.mvr

        loaded = sorted(
            name
            for name in sys.modules
            if name == "torch"
            or name.startswith("torch.")
            or name == "conin.algorithms.util.mvr_kernels"
        )
        print(loaded)
        """)
    env = dict(os.environ, PYTHONPATH=str(PACKAGE_DIR.parent))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=PACKAGE_DIR.parent,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
