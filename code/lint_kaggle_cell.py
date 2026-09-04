#!/usr/bin/env python3
"""Catch names used before assignment at module level in kaggle/KAGGLE_CELL.py.

    python3 code/lint_kaggle_cell.py kaggle/KAGGLE_CELL.py

RUN THIS AFTER EVERY EDIT TO THE CELL. It is pasted into a notebook and executed top to
bottom, so a name whose definition lived in a block that was later replaced fails at
RUNTIME -- and on Kaggle that means a NameError after the inputs are mounted, sometimes
after the GPU is already warm. Three separate bugs of this shape have each cost a run:
a trainer flag that was not in the mounted dataset, an unflatten() that globbed one
dataset directory, and WORKOUT defined inside a superseded block.

Module level only: function bodies resolve names at call time, and comprehensions have
their own scope, so both are handled rather than reported.

The cell is pasted into a notebook and run top-to-bottom, so a name defined only in a
block I later replaced fails at RUNTIME, hours into a session, after the GPU is warm.
Three of those have now cost a run each (--seed missing, wrist glob, WORKOUT). Static
check instead of eyeballing.
"""
import ast, builtins, sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)
defined = set(dir(builtins))
problems = []

class Scope(ast.NodeVisitor):
    """Only walks module level; function bodies get their names at call time."""
    def visit_FunctionDef(self, node):
        defined.add(node.name)          # body deliberately not walked
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, node):
        defined.add(node.name)
    def visit_Import(self, node):
        for a in node.names:
            defined.add((a.asname or a.name).split(".")[0])
    def visit_ImportFrom(self, node):
        for a in node.names:
            defined.add(a.asname or a.name)
    def visit_ListComp(self, node):
        self._comp(node)
    visit_SetComp = visit_GeneratorExp = visit_DictComp = visit_ListComp

    def _comp(self, node):
        # comprehensions have their own scope: bind the targets, then walk the body
        local = set()
        for gen in node.generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    local.add(n.id)
        defined.update(local)

    def generic_visit(self, node):
        # record loads before assignment, in source order
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load) and child.id not in defined:
                    problems.append((child.lineno, child.id))
                elif isinstance(child.ctx, (ast.Store, ast.Del)):
                    defined.add(child.id)
        super().generic_visit(node)

for stmt in tree.body:
    # evaluate the value side before binding targets
    Scope().visit(stmt)
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defined.add(n.id)
        elif isinstance(n, (ast.Import,)):
            for a in n.names: defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: defined.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)

seen = set()
out = [(l, n) for l, n in problems if not (n in seen or seen.add(n))]
if out:
    for l, n in out:
        print(f"  line {l}: {n!r} used before assignment")
    sys.exit(1)
print("  no module-level name used before assignment")
