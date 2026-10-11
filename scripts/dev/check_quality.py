#!/usr/bin/env python3
"""Check source syntax and maintained API contracts without importing backends."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Iterator, Union

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "configs/development/api.json"
TEXT_SUFFIXES = {".py", ".md", ".html", ".ini", ".txt", ".yml", ".yaml"}
AUDIENCE_TERMS = re.compile(r"\bstudents?\b|\bteaching\b|\binstructors?\b", re.I)
FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def load_scope(path: Path) -> dict:
    """Load and validate the maintained API and language-check scope.

    Parameters
    ----------
    path : pathlib.Path
        JSON scope file to read; no runtime modules are imported.

    Returns
    -------
    dict
        Validated module, syntax-root and active-language path lists.

    Raises
    ------
    ValueError
        If the schema, lists, uniqueness or repository-relative paths are invalid.
    """
    scope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scope, dict) or scope.get("schema_version") != "fakect.development-api/1":
        raise ValueError("Expected fakect.development-api/1 documentation scope")
    for field in ("modules", "syntax_roots", "language_paths"):
        values = scope.get(field)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value for value in values)):
            raise ValueError(f"{field} must be a nonempty list of relative paths")
        if len(values) != len(set(values)):
            raise ValueError(f"{field} contains duplicate paths")
        for value in values:
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"{field} path must stay inside the repository: {value}")
    if any(not value.endswith(".py") for value in scope["modules"]):
        raise ValueError("API modules must name Python source files")
    return scope


def iter_functions(tree: ast.Module) -> Iterator[tuple[str, FunctionNode]]:
    """Yield top-level functions and direct class methods, excluding local closures.

    Parameters
    ----------
    tree : ast.Module
        Parsed Python module; executable statements are not evaluated.

    Returns
    -------
    iterator of tuple
        Qualified function name and its FunctionDef or AsyncFunctionDef node.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{method.name}", method


def doc_sections(docstring: str) -> dict[str, str]:
    """Extract underline-delimited NumPy-style sections from a function docstring.

    Parameters
    ----------
    docstring : str
        Cleaned documentation text returned by ast.get_docstring.

    Returns
    -------
    dict of str
        Section heading to stripped content; preamble is stored as Purpose.
    """
    lines = docstring.splitlines()
    headers = [(index, line.strip()) for index, line in enumerate(lines[:-1])
               if line.strip() and re.fullmatch(r"-{3,}", lines[index + 1].strip())]
    result = {"Purpose": "\n".join(lines[:headers[0][0]] if headers else lines).strip()}
    for number, (index, heading) in enumerate(headers):
        end = headers[number + 1][0] if number + 1 < len(headers) else len(lines)
        result[heading] = "\n".join(lines[index + 2:end]).strip()
    return result


def contract_errors(node: FunctionNode) -> list[str]:
    """Check a function's purpose, return contract and named parameter documentation.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Function syntax node whose first statement is inspected as documentation.

    Returns
    -------
    list of str
        Actionable contract defects; an empty list indicates structural compliance.

    Notes
    -----
    This checks documentation structure, not the truth of scientific statements.
    self and cls are implicit; local callbacks are outside the maintained scope.
    """
    docstring = ast.get_docstring(node)
    if not docstring:
        return ["missing function docstring"]
    sections = doc_sections(docstring)
    errors = []
    for section in ("Purpose", "Parameters", "Returns"):
        if not sections.get(section):
            errors.append(f"missing or empty {section} section")
    names = []
    for line in sections.get("Parameters", "").splitlines():
        if line and not line[0].isspace() and ":" in line:
            names.extend(name.strip().lstrip("*") for name in line.split(":", 1)[0].split(","))
    arguments = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
    arguments += [arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
    arguments = [name for name in arguments if name not in {"self", "cls"}]
    for name in arguments:
        if name not in names:
            errors.append(f"Parameters does not document {name}")
    for name in names:
        if name not in arguments:
            errors.append(f"Parameters documents nonexistent argument {name}")
    return errors


def collect_files(root: Path, paths: list[str], suffixes: set[str]) -> list[Path]:
    """Enumerate selected repository files without traversing artifact directories.

    Parameters
    ----------
    root : pathlib.Path
        Repository root used to resolve the explicitly configured paths.
    paths : list of str
        Relative files or directories to inspect.
    suffixes : set of str
        Permitted filename suffixes; suffixless explicit files are also included.

    Returns
    -------
    list of pathlib.Path
        Sorted unique source paths; missing scope paths raise FileNotFoundError.
    """
    files = set()
    for relative in paths:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing scope path: {relative}")
        if path.is_file():
            files.add(path)
        else:
            files.update(candidate for candidate in path.rglob("*")
                         if candidate.is_file() and candidate.suffix in suffixes
                         and "__pycache__" not in candidate.parts)
    return sorted(files)


def check_repository(root: Path, scope: dict) -> dict:
    """Validate Python syntax, declared function contracts and active project language.

    Parameters
    ----------
    root : pathlib.Path
        Repository checkout to inspect, without writing files or importing code.
    scope : dict
        Validated configuration from load_scope.

    Returns
    -------
    dict
        Counts, error records and a passed flag suitable for CLI or CI use.
    """
    errors, trees = [], {}
    files = collect_files(root, scope["syntax_roots"], {".py"})
    files = sorted(set(files) | {root / relative for relative in scope["modules"]})
    for path in files:
        relative = str(path.relative_to(root))
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        except (SyntaxError, UnicodeError, OSError) as error:
            errors.append({"path": relative, "line": getattr(error, "lineno", 1),
                           "message": f"Python syntax/read failure: {error}"})
    count = 0
    for relative in scope["modules"]:
        tree = trees.get(root / relative)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                errors.append({"path": relative, "line": node.lineno,
                               "message": f"{node.name}: missing class purpose docstring"})
        for name, node in iter_functions(tree):
            count += 1
            for error in contract_errors(node):
                errors.append({"path": relative, "line": node.lineno,
                               "message": f"{name}: {error}"})
    for path in collect_files(root, scope["language_paths"], TEXT_SUFFIXES):
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            match = AUDIENCE_TERMS.search(line)
            if match:
                errors.append({"path": str(path.relative_to(root)), "line": number,
                               "message": f"Use generic developer language instead of {match.group()!r}"})
    return {"passed": not errors, "syntax_files": len(files),
            "api_modules": len(scope["modules"]), "documented_functions": count, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    """Run static repository checks and print a concise or JSON diagnostic.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments excluding the executable; None uses process arguments.

    Returns
    -------
    int
        Zero for passing checks, one for defects, or two for an invalid scope.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scope", type=Path, default=SCOPE)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable counts and diagnostics")
    args = parser.parse_args(argv)
    try:
        result = check_repository(args.root.resolve(), load_scope(args.scope))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for error in result["errors"]:
            print(f'{error["path"]}:{error["line"]}: {error["message"]}')
        print(f'{"PASS" if result["passed"] else "FAIL"}: {result["syntax_files"]} Python files; '
              f'{result["documented_functions"]} function contracts in {result["api_modules"]} modules')
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
