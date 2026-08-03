# tools/_code_param_check.py
"""AST-based verification that generated code used the validated parameters.

Closes the loop flagged in docs/PARAMETER_EXTRACTION.md: the validated
``LensParameterSet`` used to bind only via prompt. This module parses the
lenstronomy input dictionaries back OUT of the generated script (``ast``, not
regex) and diffs them against the validated set, field by field.

Resolution rules (in order):
* literals (including unary minus and constant-folding of + - * / ** on
  resolved operands, so ``10**3`` and ``-0.1`` work);
* names resolved through a symbol table built from simple ``name = value``
  assignments in source order.
Anything else (function calls, loop variables, attribute reads) is reported as
UNRESOLVED — never guessed, and unresolved values do NOT pass.

Where each field is looked for:
* ``lens_model_list``/``source_model_list``: a ``lens_model_list=`` /
  ``light_model_list=`` call keyword, or an assignment to that name
  (``source_model_list`` also accepted).
* ``kwargs_lens``/``kwargs_source``: assignments to those names.
* ``numPix``/``deltaPix``: the first two arguments of a
  ``data_configure_simple(...)`` call, falling back to same-named assignments.
* ``exposure_time``/``background_rms``: keywords of that call, falling back to
  same-named assignments.
* PSF ``fwhm``: the ``fwhm=`` keyword of a ``PSF(...)`` call.
"""

from __future__ import annotations

import ast
from typing import Any, Optional

from dlens.schemas._lens_params import CodeParamComparison, FieldComparison, LensParameterSet

UNRESOLVED = "<unresolved>"

_REL_TOL = 1e-6


class _Extracted:
    """What the AST walk recovered from the script (None = not found)."""

    def __init__(self) -> None:
        self.lens_model_list: Optional[Any] = None
        self.source_model_list: Optional[Any] = None
        self.kwargs_lens: Optional[Any] = None
        self.kwargs_source: Optional[Any] = None
        self.numPix: Optional[Any] = None
        self.deltaPix: Optional[Any] = None
        self.exposure_time: Optional[Any] = None
        self.background_rms: Optional[Any] = None
        self.psf_fwhm: Optional[Any] = None


def _resolve(node: ast.AST, symbols: dict[str, Any]) -> Any:
    """Resolve a node to a Python value, UNRESOLVED, or containers thereof."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = _resolve(node.operand, symbols)
        if isinstance(val, (int, float)):
            return -val if isinstance(node.op, ast.USub) else val
        return UNRESOLVED
    if isinstance(node, ast.Name):
        return symbols.get(node.id, UNRESOLVED)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        left, right = _resolve(node.left, symbols), _resolve(node.right, symbols)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
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
        # Still fully static: dict/list literal reached through a name, indexed
        # by a constant (gpt-5.2 routinely routes literals through kwargs
        # dicts, e.g. kwargs_data["exposure_time"]).
        base = _resolve(node.value, symbols)
        key = _resolve(node.slice, symbols)
        if isinstance(base, dict) and not isinstance(key, ast.AST) and key in base:
            return base[key]
        if (
            isinstance(base, list)
            and isinstance(key, int)
            and not isinstance(key, bool)
            and 0 <= key < len(base)
        ):
            return base[key]
        return UNRESOLVED
    if isinstance(node, ast.List):
        return [_resolve(el, symbols) for el in node.elts]
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values):
            key = _resolve(k, symbols) if k is not None else UNRESOLVED
            out[key] = _resolve(v, symbols)
        return out
    return UNRESOLVED


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def extract_lenstronomy_inputs(code: str) -> _Extracted:
    """Walk the script and recover the lenstronomy input values."""
    tree = ast.parse(code)
    symbols: dict[str, Any] = {}
    found = _Extracted()

    for node in ast.walk(tree):
        # Symbol table: simple single-target assignments, source order is fine
        # for generated scripts (ast.walk is close enough; shadowing is rare
        # in single-shot generated code).
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            name = node.targets[0].id
            value = _resolve(node.value, symbols)
            symbols[name] = value
            if name == "lens_model_list":
                found.lens_model_list = value
            elif name in ("source_model_list", "light_model_list"):
                found.source_model_list = value
            elif name == "kwargs_lens":
                found.kwargs_lens = value
            elif name == "kwargs_source":
                found.kwargs_source = value

        if isinstance(node, ast.Call):
            cname = _call_name(node)
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if "lens_model_list" in kw and found.lens_model_list is None:
                found.lens_model_list = _resolve(kw["lens_model_list"], symbols)
            if "light_model_list" in kw and found.source_model_list is None:
                found.source_model_list = _resolve(kw["light_model_list"], symbols)
            if cname == "data_configure_simple":
                if len(node.args) >= 2:
                    found.numPix = _resolve(node.args[0], symbols)
                    found.deltaPix = _resolve(node.args[1], symbols)
                if "exposure_time" in kw:
                    found.exposure_time = _resolve(kw["exposure_time"], symbols)
                if "background_rms" in kw:
                    found.background_rms = _resolve(kw["background_rms"], symbols)
            if cname == "PSF" and "fwhm" in kw:
                found.psf_fwhm = _resolve(kw["fwhm"], symbols)

    # Fallbacks to plain assignments for the data block.
    for attr in ("numPix", "deltaPix", "exposure_time", "background_rms"):
        if getattr(found, attr) is None and attr in symbols:
            setattr(found, attr, symbols[attr])
    return found


def _numbers_match(expected: float, actual: Any) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    return abs(float(actual) - float(expected)) <= _REL_TOL * max(
        1.0, abs(float(expected))
    )


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
    if params.kwargs_psf.fwhm is not None:
        record("psf_fwhm", params.kwargs_psf.fwhm, found.psf_fwhm)

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
