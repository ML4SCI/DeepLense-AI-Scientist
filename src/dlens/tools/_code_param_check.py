# tools/_code_param_check.py
"""AST-based verification that generated code used the validated parameters.

Closes the loop flagged in docs/PARAMETER_EXTRACTION.md: the validated
``LensParameterSet`` used to bind only via prompt. This module parses the
lenstronomy input dictionaries back OUT of the generated script (``ast``, not
regex) and diffs them against the validated set, field by field.

Resolution rules (in order):
* literals (including unary minus and constant-folding of + - * / ** on
  resolved operands, so ``10**3`` and ``-0.1`` work);
* names, through a symbol table of ``name = value`` assignments;
* constant subscripts of resolved dicts and lists (gpt-5.2 routinely routes
  literals through kwargs dicts, e.g. ``kwargs_data["exposure_time"]``), and
  ``**`` expansions of resolved dicts, applied in order;
* the lenstronomy objects a name holds: ``LensModel``, ``LightModel``, ``PSF``
  and ``ImageModel`` calls are followed through assignments.
Anything else (other calls, loop variables, attribute reads) is reported as
UNRESOLVED — never guessed, and unresolved values do NOT pass.

The script is followed the way it runs. A branch whose condition is known
statically (``if False:``, ``if DEBUG:`` with ``DEBUG = True``, ``if __name__ ==
"__main__":``) contributes only the side that runs; a function body counts
only where the function is called, in its own scope. When the path depends on
runtime values (another condition, a loop, an exception), a value that differs
between the possible paths is UNRESOLVED.

Where each field is looked for:
* ``kwargs_lens``/``kwargs_source``: the arguments of the last
  ``ImageModel.image(...)`` call, positional or by keyword, since that is what
  is rendered (an argument the call leaves out is missing); without such a
  call, assignments to those names.
* ``lens_model_list``/``source_model_list``: the ``LensModel``/``LightModel``
  passed to the ``ImageModel`` that renders, else the only one the script
  builds, else an assignment to that name (``light_model_list`` is accepted for
  the source too).
* ``numPix``/``deltaPix``/``exposure_time``/``background_rms``: the arguments of
  ``data_configure_simple(...)``, positional or by keyword in either the 1.9.2
  (``numPix``) or 1.14.1+ (``num_pix``) spelling, falling back to assignments to
  either name.
* PSF ``psf_type``/``fwhm``/``pixel_size``: the ``PSF(...)`` passed to that
  ``ImageModel``, else the only one built. lenstronomy's default ``psf_type`` is
  ``'NONE'``, so leaving it out counts as ``'NONE'``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from dlens.schemas._lens_params import CodeParamComparison, FieldComparison, LensParameterSet

UNRESOLVED = "<unresolved>"

_REL_TOL = 1e-6

# data_configure_simple's grid arguments: lenstronomy 1.9.2 spelling, 1.14+ spelling.
_GRID_ALIASES = (("numPix", "num_pix"), ("deltaPix", "delta_pix"))

# The leading parameters of the calls the check follows (identical in 1.9.2 and 1.14).
_PARAMETERS = {
    "LensModel": ("lens_model_list",),
    "LightModel": ("light_model_list",),
    "PSF": ("psf_type", "fwhm", "truncation", "pixel_size"),
    "ImageModel": ("data_class", "psf_class", "lens_model_class", "source_model_class"),
    "data_configure_simple": ("numPix", "deltaPix", "exposure_time", "background_rms"),
    "image": ("kwargs_lens", "kwargs_source"),
}
_PSF_DEFAULT_TYPE = "NONE"

# Names whose assignments stand in for a field when nothing more direct is found.
_CONVENTIONAL = {"lens_model_list": "lens_model_list", "source_model_list": "source_model_list",
                 "light_model_list": "source_model_list", "kwargs_lens": "kwargs_lens",
                 "kwargs_source": "kwargs_source"}

_ABSENT = object()  # marks a key missing on one control-flow path
_NOT_GIVEN = object()  # the rendering ImageModel was not passed that component
_MAX_CALL_DEPTH = 20


@dataclass
class _Built:
    """A lenstronomy object the script builds, with the arguments the check follows."""

    kind: str
    fields: dict[str, Any]


class _Extracted:
    """What the AST walk recovered from the script (None = not found)."""

    def __init__(self) -> None:
        self.lens_model_list: Any = None
        self.source_model_list: Any = None
        self.kwargs_lens: Any = None
        self.kwargs_source: Any = None
        self.numPix: Any = None
        self.deltaPix: Any = None
        self.exposure_time: Any = None
        self.background_rms: Any = None
        self.psf_type: Any = None
        self.psf_fwhm: Any = None
        self.psf_pixel_size: Any = None


def _resolve(node: ast.AST, symbols: dict[str, Any]) -> Any:
    """Resolve a node to a Python value, UNRESOLVED, or containers thereof."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _resolve(node.operand, symbols)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return -val if isinstance(node.op, ast.USub) else val
        return UNRESOLVED
    if isinstance(node, ast.Name):
        return symbols.get(node.id, UNRESOLVED)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        left, right = _resolve(node.left, symbols), _resolve(node.right, symbols)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (left, right)):
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                return left**right
            except (ZeroDivisionError, OverflowError):
                return UNRESOLVED
        return UNRESOLVED
    if isinstance(node, ast.Subscript):
        # Still fully static: a dict or list literal reached through a name and
        # indexed by a constant, e.g. kwargs_data["exposure_time"].
        base = _resolve(node.value, symbols)
        key = _resolve(node.slice, symbols)
        if isinstance(base, dict) and isinstance(key, (str, int, float)) and key in base:
            return base[key]
        if (
            isinstance(base, (list, tuple))
            and isinstance(key, int)
            and not isinstance(key, bool)
            and 0 <= key < len(base)
        ):
            return base[key]
        return UNRESOLVED
    if isinstance(node, ast.List):
        return [_resolve(el, symbols) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_resolve(el, symbols) for el in node.elts)
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                # **mapping: merged in order, so it can override an earlier key.
                mapping = _resolve(v, symbols)
                if not isinstance(mapping, dict):
                    return UNRESOLVED
                out.update(mapping)
                continue
            key = _resolve(k, symbols)
            if key == UNRESOLVED or not isinstance(key, (str, int, float, bool, type(None))):
                return UNRESOLVED  # an unknown key could override any of the others
            out[key] = _resolve(v, symbols)
        return out
    if isinstance(node, ast.Call):
        return _build(node, symbols)
    return UNRESOLVED


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _arguments(call: ast.Call, symbols: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """The call's arguments, resolved and bound to the callee's leading parameter ``names``.

    Keyword arguments are kept under their own names. An argument the call does
    not pass is absent, or UNRESOLVED when a ``*`` or ``**`` expansion that cannot
    be resolved might be passing it.
    """
    out: dict[str, Any] = {}
    opaque = False
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Starred):
            opaque = True
            break
        if i < len(names):
            out[names[i]] = _resolve(arg, symbols)
    for keyword in call.keywords:
        if keyword.arg is not None:
            out[keyword.arg] = _resolve(keyword.value, symbols)
            continue
        mapping = _resolve(keyword.value, symbols)
        if isinstance(mapping, dict):
            out.update({k: v for k, v in mapping.items() if isinstance(k, str)})
        else:
            opaque = True
    if opaque:
        for name in names:
            out.setdefault(name, UNRESOLVED)
    return out


def _build(call: ast.Call, symbols: dict[str, Any]) -> Any:
    """What a call returns, as far as the check follows it."""
    name = _call_name(call)
    if name not in ("LensModel", "LightModel", "PSF", "ImageModel"):
        return UNRESOLVED
    args = _arguments(call, symbols, _PARAMETERS[name])
    if name == "PSF":
        return _Built("PSF", {"psf_type": args.get("psf_type", _PSF_DEFAULT_TYPE),
                              "fwhm": args.get("fwhm"), "pixel_size": args.get("pixel_size")})
    if name == "ImageModel":
        return _Built("ImageModel", {"psf": args.get("psf_class"),
                                     "lens": args.get("lens_model_class"),
                                     "source": args.get("source_model_class")})
    return _Built(name, {"models": args.get(_PARAMETERS[name][0])})


def _truth(node: ast.AST, symbols: dict[str, Any]) -> bool | None:
    """The condition's value when it is known statically, else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _truth(node.operand, symbols)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_truth(v, symbols) for v in node.values]
        decisive = isinstance(node.op, ast.Or)  # True decides an `or`, False an `and`
        if decisive in values:
            return decisive
        return (not decisive) if all(v is not None for v in values) else None
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = node.left, node.comparators[0]
        if (isinstance(left, ast.Name) and left.id == "__name__"
                and isinstance(right, ast.Constant) and right.value == "__main__"
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
            # The check is about the script run directly.
            return isinstance(node.ops[0], ast.Eq)
        a, b = _resolve(left, symbols), _resolve(right, symbols)
        scalars = (str, int, float, bool, type(None))
        if a == UNRESOLVED or b == UNRESOLVED or not (isinstance(a, scalars) and isinstance(b, scalars)):
            return None
        op = node.ops[0]
        try:
            if isinstance(op, ast.Eq):
                return a == b
            if isinstance(op, ast.NotEq):
                return a != b
            if isinstance(op, ast.Is):
                return a is b
            if isinstance(op, ast.IsNot):
                return a is not b
            if isinstance(op, ast.Lt):
                return a < b
            if isinstance(op, ast.LtE):
                return a <= b
            if isinstance(op, ast.Gt):
                return a > b
            if isinstance(op, ast.GtE):
                return a >= b
        except TypeError:
            return None
        return None
    value = _resolve(node, symbols)
    if value == UNRESOLVED or isinstance(value, _Built):
        return None
    return bool(value)


def _names_bound(target: ast.AST) -> list[str]:
    """Every plain name an assignment target or loop variable binds."""
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def _calls_in_order(node: ast.AST) -> list[ast.Call]:
    """The calls inside an expression or simple statement, in evaluation order."""
    out: list[ast.Call] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Lambda):
            return  # its body runs only when called
        for child in ast.iter_child_nodes(n):
            visit(child)
        if isinstance(n, ast.Call):
            out.append(n)

    visit(node)
    return out


class _State:
    """The names bound on one control-flow path, and what that path built and rendered."""

    def __init__(self, symbols: dict[str, Any] | None = None,
                 found: dict[str, Any] | None = None) -> None:
        self.symbols: dict[str, Any] = dict(symbols or {})
        self.found: dict[str, Any] = dict(found or {})

    def copy(self) -> _State:
        return _State(self.symbols, self.found)

    def take(self, other: _State) -> None:
        self.symbols, self.found = other.symbols, other.found


def _merge(paths: list[_State]) -> _State:
    """Join control-flow paths: anything that differs between them is UNRESOLVED."""
    merged = _State()
    for attr in ("symbols", "found"):
        tables = [getattr(p, attr) for p in paths]
        out = getattr(merged, attr)
        for key in {k for t in tables for k in t}:
            values = [t.get(key, _ABSENT) for t in tables]
            same = values[0] is not _ABSENT and all(v == values[0] for v in values[1:])
            out[key] = values[0] if same else UNRESOLVED
    return merged


class _Walker:
    """Follows the script statement by statement, the way it runs."""

    def __init__(self) -> None:
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.active: list[str] = []

    def block(self, stmts: list[ast.stmt], state: _State) -> None:
        for stmt in stmts:
            self.statement(stmt, state)

    def statement(self, node: ast.stmt, state: _State) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.functions[node.name] = node  # the body runs only where it is called
            state.symbols[node.name] = UNRESOLVED
            return
        if isinstance(node, ast.ClassDef):
            state.symbols[node.name] = UNRESOLVED
            return
        if isinstance(node, ast.If):
            self.calls(node.test, state)
            value = _truth(node.test, state.symbols)
            if value is not None:
                self.block(node.body if value else node.orelse, state)
                return
            taken, skipped = state.copy(), state.copy()
            self.block(node.body, taken)
            self.block(node.orelse, skipped)
            state.take(_merge([taken, skipped]))
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            self.loop(node, state)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.calls(item.context_expr, state)
                if item.optional_vars is not None:
                    self.bind(item.optional_vars, UNRESOLVED, state)
            self.block(node.body, state)
            return
        if isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            self.attempt(node, state)
            return
        if isinstance(node, ast.Match):
            self.calls(node.subject, state)
            paths = [state.copy()]  # no case matched
            for case in node.cases:
                path = state.copy()
                for name in _names_bound(case.pattern):
                    path.symbols[name] = UNRESOLVED
                self.block(case.body, path)
                paths.append(path)
            state.take(_merge(paths))
            return
        self.calls(node, state)
        if isinstance(node, ast.Assign):
            value = _resolve(node.value, state.symbols)
            for target in node.targets:
                self.bind(target, value, state)

    def loop(self, node: ast.For | ast.AsyncFor | ast.While, state: _State) -> None:
        """The body runs zero or more times: iterate until the joined state is stable."""
        if isinstance(node, ast.While):
            self.calls(node.test, state)
            if _truth(node.test, state.symbols) is False:
                self.block(node.orelse, state)
                return
        else:
            self.calls(node.iter, state)
        entry = state.copy()
        for _ in range(3):
            body = entry.copy()
            if not isinstance(node, ast.While):
                self.bind(node.target, UNRESOLVED, body)
            self.block(node.body, body)
            joined = _merge([state, body])
            if joined.symbols == entry.symbols and joined.found == entry.found:
                break
            entry = joined
        state.take(entry)
        self.block(node.orelse, state)

    def attempt(self, node: ast.AST, state: _State) -> None:
        """try/except: a handler may start from any point in the body."""
        before = state.copy()
        self.block(node.body, state)
        self.block(node.orelse, state)
        paths = [state.copy()]
        for handler in node.handlers:
            path = _merge([before, state])
            if handler.name:
                path.symbols[handler.name] = UNRESOLVED
            self.block(handler.body, path)
            paths.append(path)
        state.take(_merge(paths))
        self.block(node.finalbody, state)

    def bind(self, target: ast.AST, value: Any, state: _State) -> None:
        if isinstance(target, ast.Name):
            state.symbols[target.id] = value
            if target.id in _CONVENTIONAL:
                state.found[f"assigned:{_CONVENTIONAL[target.id]}"] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            parts = list(value) if isinstance(value, (list, tuple)) else None
            starred = any(isinstance(el, ast.Starred) for el in target.elts)
            if parts is None or starred or len(parts) != len(target.elts):
                parts = [UNRESOLVED] * len(target.elts)
            for el, part in zip(target.elts, parts):
                self.bind(el.value if isinstance(el, ast.Starred) else el, part, state)
        # Attribute and subscript targets (in-place edits) are not followed.

    def calls(self, node: ast.AST, state: _State) -> None:
        for call in _calls_in_order(node):
            self.call(call, state)

    def call(self, call: ast.Call, state: _State) -> None:
        name, found = _call_name(call), state.found
        if name in ("LensModel", "LightModel", "PSF", "ImageModel"):
            key = f"built:{name}"
            previous = found.get(key, ())
            found[key] = UNRESOLVED if previous == UNRESOLVED else previous + (_build(call, state.symbols),)
        elif name == "data_configure_simple":
            args = _arguments(call, state.symbols, _PARAMETERS[name])
            for camel, snake in _GRID_ALIASES:
                if snake in args:
                    args[camel] = args.pop(snake)
            found["data"] = {k: args.get(k) for k in _PARAMETERS[name]}
        elif name == "image" and isinstance(call.func, ast.Attribute):
            args = _arguments(call, state.symbols, _PARAMETERS["image"])
            found["rendered"] = {"kwargs_lens": args.get("kwargs_lens"),
                                 "kwargs_source": args.get("kwargs_source"),
                                 "image_model": _resolve(call.func.value, state.symbols)}
        if isinstance(call.func, ast.Name) and call.func.id in self.functions:
            self.run(self.functions[call.func.id], call, state)

    def run(self, fn: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call, state: _State) -> None:
        """Run a called function's body in its own scope; what it builds still counts."""
        if fn.name in self.active or len(self.active) >= _MAX_CALL_DEPTH:
            return
        a = fn.args
        positional = [p.arg for p in (*a.posonlyargs, *a.args)]
        given = _arguments(call, state.symbols, tuple(positional))
        local = _State(state.symbols, state.found)
        defaults = dict(zip(positional[len(positional) - len(a.defaults):], a.defaults))
        defaults.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
        for p in (*positional, *(k.arg for k in a.kwonlyargs)):
            if p in given:
                local.symbols[p] = given[p]
            elif p in defaults:
                local.symbols[p] = _resolve(defaults[p], state.symbols)
            else:
                local.symbols[p] = UNRESOLVED
        for extra in (a.vararg, a.kwarg):
            if extra is not None:
                local.symbols[extra.arg] = UNRESOLVED
        shared = {n for node in ast.walk(fn) if isinstance(node, ast.Global) for n in node.names}
        self.active.append(fn.name)
        self.block(fn.body, local)
        self.active.pop()
        state.found = local.found
        for name in shared & set(local.symbols):
            state.symbols[name] = local.symbols[name]


def _only(built: Any) -> Any:
    """The one object of a kind the script builds: None if none, UNRESOLVED if several differ."""
    if built is None or built == UNRESOLVED:
        return built
    distinct: list[Any] = []
    for b in built:
        if b not in distinct:
            distinct.append(b)
    return distinct[0] if len(distinct) == 1 else UNRESOLVED


def _component(image_model: Any, role: str, kind: str, found: dict[str, Any]) -> Any:
    """The ``kind`` object the rendering ImageModel was given as ``role``, most direct evidence first."""
    if isinstance(image_model, _Built):
        given = image_model.fields[role]
        if isinstance(given, _Built) and given.kind == kind:
            return given
        if given is None:
            return _NOT_GIVEN
    return _only(found.get(f"built:{kind}"))


def extract_lenstronomy_inputs(code: str) -> _Extracted:
    """Walk the script and recover the lenstronomy input values."""
    tree = ast.parse(code)
    state = _State()
    _Walker().block(tree.body, state)
    symbols, f = state.symbols, state.found
    found = _Extracted()

    rendered = f.get("rendered")
    image_model: Any = None
    if rendered == UNRESOLVED:
        found.kwargs_lens = found.kwargs_source = image_model = UNRESOLVED
    elif rendered is not None:
        # The arguments of the last image() call outrank same-named assignments.
        found.kwargs_lens, found.kwargs_source = rendered["kwargs_lens"], rendered["kwargs_source"]
        image_model = rendered["image_model"]
    else:
        found.kwargs_lens, found.kwargs_source = f.get("assigned:kwargs_lens"), f.get("assigned:kwargs_source")
    if not (isinstance(image_model, _Built) and image_model.kind == "ImageModel"):
        image_model = _only(f.get("built:ImageModel"))

    for attr, role, kind in (("lens_model_list", "lens", "LensModel"),
                             ("source_model_list", "source", "LightModel")):
        model = _component(image_model, role, kind, f)
        if isinstance(model, _Built):
            setattr(found, attr, model.fields["models"])
        elif model == UNRESOLVED:
            setattr(found, attr, UNRESOLVED)
        elif model is None:
            setattr(found, attr, f.get(f"assigned:{attr}"))

    psf = _component(image_model, "psf", "PSF", f)
    if isinstance(psf, _Built):
        found.psf_type, found.psf_fwhm = psf.fields["psf_type"], psf.fields["fwhm"]
        found.psf_pixel_size = psf.fields["pixel_size"]
    elif psf == UNRESOLVED:
        found.psf_type = found.psf_fwhm = found.psf_pixel_size = UNRESOLVED

    data = f.get("data")
    for attr, alias in (*_GRID_ALIASES, ("exposure_time", None), ("background_rms", None)):
        value = data.get(attr) if isinstance(data, dict) else data
        if value is None:
            # Fallback to plain assignments for the data block.
            value = symbols.get(attr, symbols.get(alias) if alias else None)
        setattr(found, attr, value)
    return found


def _numbers_match(expected: float, actual: Any) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    return abs(float(actual) - float(expected)) <= _REL_TOL * max(1.0, abs(float(expected)))


def compare_code_to_params(code: str, params: LensParameterSet) -> CodeParamComparison:
    """Diff the script's lenstronomy inputs against the validated parameter set.

    Only the validated set's own fields are checked (extra keys the script adds
    are fine). ``passed`` requires every checked field to match — diverged,
    missing, and unresolved all block, so a script that computes values at
    runtime cannot silently pass.
    """
    try:
        found = extract_lenstronomy_inputs(code)
    except SyntaxError as exc:
        return CodeParamComparison(
            passed=False, matched=[], diverged=[], missing=["<script>"],
            unresolved=[], messages=[f"script is not parseable: {exc}"],
        )

    matched: list[str] = []
    diverged: list[FieldComparison] = []
    missing: list[str] = []
    unresolved: list[str] = []

    def record(field: str, expected: Any, actual: Any) -> None:
        if actual is None:
            missing.append(field)
        elif actual == UNRESOLVED:
            unresolved.append(field)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if _numbers_match(expected, actual):
                matched.append(field)
            else:
                diverged.append(
                    FieldComparison(field=field, expected=expected, actual=actual)
                )
        elif expected == actual:
            matched.append(field)
        else:
            diverged.append(FieldComparison(field=field, expected=expected, actual=actual))

    record("lens_model_list", params.lens_model_list, found.lens_model_list)
    record("source_model_list", params.source_model_list, found.source_model_list)

    for label, expected_kwargs, actual_kwargs in (
        ("kwargs_lens", params.kwargs_lens, found.kwargs_lens),
        ("kwargs_source", params.kwargs_source, found.kwargs_source),
    ):
        if actual_kwargs is None:
            missing.append(label)
            continue
        if not isinstance(actual_kwargs, list):
            unresolved.append(label)
            continue
        for i, exp_dict in enumerate(expected_kwargs):
            if i < len(actual_kwargs) and actual_kwargs[i] == UNRESOLVED:
                unresolved.append(f"{label}[{i}]")
                continue
            if i >= len(actual_kwargs) or not isinstance(actual_kwargs[i], dict):
                missing.append(f"{label}[{i}]")
                continue
            for key, exp_val in exp_dict.items():
                field = f"{label}[{i}].{key}"
                record(field, exp_val, actual_kwargs[i].get(key))

    data = params.kwargs_data
    record("numPix", data.numPix, found.numPix)
    record("deltaPix", data.deltaPix, found.deltaPix)
    if data.exposure_time is not None:
        record("exposure_time", data.exposure_time, found.exposure_time)
    if data.background_rms is not None:
        record("background_rms", data.background_rms, found.background_rms)
    psf = params.kwargs_psf
    for field, expected, actual in (("psf_type", psf.psf_type, found.psf_type),
                                    ("psf_fwhm", psf.fwhm, found.psf_fwhm),
                                    ("psf_pixel_size", psf.pixel_size, found.psf_pixel_size)):
        if expected is not None:
            record(field, expected, actual)

    messages: list[str] = []
    for d in diverged:
        messages.append(
            f"{d.field}: script uses {d.actual!r} but the validated value is "
            f"{d.expected!r} — use the validated value exactly."
        )
    for f in missing:
        messages.append(
            f"{f}: not found in the script — write it explicitly (raw lenstronomy "
            "with the validated values, not a wrapper that hides them)."
        )
    for f in unresolved:
        messages.append(
            f"{f}: value is computed at runtime and cannot be verified — write the "
            "validated value as a literal."
        )

    return CodeParamComparison(
        passed=not (diverged or missing or unresolved),
        matched=sorted(matched),
        diverged=diverged,
        missing=sorted(missing),
        unresolved=sorted(unresolved),
        messages=messages,
    )
