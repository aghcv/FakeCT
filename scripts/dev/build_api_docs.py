#!/usr/bin/env python3
"""Generate the offline maintained API reference directly from Python syntax."""
from __future__ import annotations

import argparse
import ast
import html
from pathlib import Path
import re

from check_quality import FunctionNode, ROOT, SCOPE, contract_errors, iter_functions, load_scope


def function_signature(name: str, node: FunctionNode) -> str:
    """Render a qualified Python signature without importing its defining module.

    Parameters
    ----------
    name : str
        Module-local function name or Class.method qualified name.
    node : ast.FunctionDef or ast.AsyncFunctionDef
        Source syntax containing arguments, defaults and return annotation.

    Returns
    -------
    str
        Signature including positional/keyword boundaries and optional annotations.
    """
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    result = f"{prefix}{name}({ast.unparse(node.args)})"
    if node.returns is not None:
        result += " -> " + ast.unparse(node.returns)
    return result


def render_reference(root: Path, scope: dict) -> str:
    """Render escaped, searchable HTML from the declared module contracts.

    Parameters
    ----------
    root : pathlib.Path
        Repository checkout whose maintained source files are read only.
    scope : dict
        Validated API scope containing repository-relative module filenames.

    Returns
    -------
    str
        Complete standalone HTML document; no external fonts, scripts or backends.

    Raises
    ------
    ValueError
        If a maintained function lacks the required documentation contract.
    """
    navigation, sections = [], []
    count = 0
    for relative in scope["modules"]:
        tree = ast.parse((root / relative).read_text(encoding="utf-8-sig"), filename=relative)
        identifier = "module-" + re.sub(r"[^a-zA-Z0-9_-]", "-", relative)
        navigation.append(f'<a href="#{identifier}">{html.escape(relative)}</a>')
        parts = [f'<section id="{identifier}" class="module"><h2>{html.escape(relative)}</h2>']
        module_doc = ast.get_docstring(tree) or ""
        parts.append(f'<p class="module-purpose">{html.escape(module_doc)}</p>')
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = html.escape(ast.get_docstring(node) or "")
                parts.append(f'<div class="class-purpose"><h3>Class {html.escape(node.name)}</h3>'
                             f'<pre>{doc}</pre></div>')
        for name, node in iter_functions(tree):
            defects = contract_errors(node)
            if defects:
                raise ValueError(f'{relative}:{node.lineno}: {name}: {"; ".join(defects)}')
            count += 1
            anchor = identifier + "-" + name.replace(".", "-")
            signature = html.escape(function_signature(name, node))
            doc = html.escape(ast.get_docstring(node) or "")
            parts.append(f'<article class="contract" id="{anchor}"><h3><code>{signature}</code></h3>'
                         f'<p class="source">Source: {html.escape(relative)}:{node.lineno}</p>'
                         f'<pre>{doc}</pre></article>')
        parts.append("</section>")
        sections.append("\n".join(parts))
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>syncoCT · API reference</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#18283b;font:17px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:32px 24px 60px}h1{font-size:36px;line-height:1.2}h2{font-size:24px;overflow-wrap:anywhere;margin-top:36px}h3{font-size:17px;overflow-wrap:anywhere}a{color:#08698b}nav{display:flex;flex-wrap:wrap;gap:18px}.modules{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:14px}.contract{background:white;border:1px solid #d6e3eb;border-left:4px solid #08698b;border-radius:6px;margin:18px 0;padding:14px 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;margin:0}.source{font-size:13px;color:#536478}input{width:100%;max-width:700px;padding:12px;font:inherit;border:1px solid #b8c9d6;border-radius:6px}.module-purpose{white-space:pre-wrap;color:#536478}code{font-family:ui-monospace,monospace}.count{color:#536478}section{scroll-margin-top:16px}[hidden]{display:none!important}
</style></head><body><main>
<header><h1>syncoCT API reference</h1><nav><a href="index.html">Workflow guide</a><a href="contributing.html">Developer conventions</a><a href="testing.html">Testing guide</a><a href="../reference/index.html">Method references</a></nav></header>
<p>Purpose, inputs, outputs and side effects for the maintained workflow API.
Generated from source syntax; no runtime modules or optional learning backends are imported.
Experimental research modules and local callbacks are outside this declared API scope.</p>
<p class="count">''' + str(len(scope["modules"])) + " modules · " + str(count) + ''' function contracts</p>
<label for="search">Find a function, module, parameter or unit</label><br>
<input id="search" type="search" placeholder="e.g. tube_mask, spacing_ijk_mm, registry">
<p id="matches" aria-live="polite"></p>
<details><summary>Module index</summary><p class="modules">''' + "\n".join(navigation) + '''</p></details>
''' + "\n".join(sections) + '''
<footer><p>Regenerate with <code>python scripts/dev/build_api_docs.py</code>.
Scope: <code>configs/development/api.json</code>.</p></footer>
</main><script>
const input=document.getElementById('search'), modules=[...document.querySelectorAll('.module')];
input.addEventListener('input',()=>{const term=input.value.trim().toLowerCase();let found=0;
for(const module of modules){let visible=0;const moduleMatch=module.querySelector('h2').textContent.toLowerCase().includes(term);
for(const article of module.querySelectorAll('.contract')){const show=!term||moduleMatch||article.textContent.toLowerCase().includes(term);article.hidden=!show;if(show){visible++;found++;}}
module.hidden=Boolean(term)&&!visible;}
document.getElementById('matches').textContent=term?found+' matching function contracts':'';});
</script></body></html>
'''


def build_reference(root: Path, scope: dict, output: Path) -> dict:
    """Generate and write one reproducible offline API artifact.

    Parameters
    ----------
    root : pathlib.Path
        Repository source root to inspect without executing code.
    scope : dict
        Validated module scope from load_scope.
    output : pathlib.Path
        Destination HTML file; missing parent directories are created.

    Returns
    -------
    dict
        Output path, encoded byte count and maintained module count.

    Notes
    -----
    Replaces the documentation artifact only; source files are never modified.
    Output excludes clocks and absolute source paths for reproducible builds.
    """
    document = render_reference(root, scope)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"output": str(output), "bytes": len(document.encode("utf-8")),
            "api_modules": len(scope["modules"])}


def main(argv: list[str] | None = None) -> int:
    """Build the source-derived reference and print its artifact path and size.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments excluding the executable; None uses process arguments.

    Returns
    -------
    int
        Zero on success; argparse exits with two for scope/source errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scope", type=Path, default=SCOPE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/developer/api.html")
    args = parser.parse_args(argv)
    try:
        result = build_reference(args.root.resolve(), load_scope(args.scope), args.output)
    except (OSError, SyntaxError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f'{result["output"]} ({result["bytes"]:,} bytes)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
