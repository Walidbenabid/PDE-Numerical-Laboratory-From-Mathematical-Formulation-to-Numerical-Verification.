# -*- coding: utf-8 -*-
"""
PDE NUMERICAL LABORATORY
Interactive Environment for Numerical Analysis of Partial Differential Equations

Laboratoire numérique interactif pour l'analyse et la résolution d'EDP linéaires :
définition et classification, formulations forte et faible, espaces de Sobolev,
Lax–Milgram, différences finies, éléments finis P1, schémas temporels,
stabilité de von Neumann / spectrale, consistance, convergence, visualisation.

Lancement      :  pip install -r requirements.txt
                  streamlit run main.py
Auto-tests     :  python main.py --selftest

Organisation du fichier (sections) :
  1. Imports, constantes, limites
  2. Analyse d'expressions sécurisée (liste blanche, sans eval/exec) + compilation numérique
  3. Catalogue des EDP, configuration, modèle de problème, solutions manufacturées
  4. Classification mathématique
  5. Formulations forte / faible, espaces fonctionnels, Lax–Milgram
  6. Quadratures, maillages, fonctions de forme P1
  7. Assemblage éléments finis P1 et opérateurs différences finies
  8. Solveurs (stationnaires, θ-schéma, schémas pour les ondes, transport)
  9. Erreurs, stabilité (von Neumann, spectrale), convergence
 10. Interprétation scientifique, validation automatique
 11. Visualisation (Plotly)
 12. Interface Streamlit
"""
from __future__ import annotations

import copy
import datetime as _dt
import difflib
import hashlib
import io
import json
import math
import platform
import re
import sys
import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import scipy
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import streamlit as st
import sympy
from plotly.subplots import make_subplots
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)
# Hors exécution Streamlit (auto-tests en ligne de commande), les décorateurs de cache avertissent
# qu'aucun runtime n'est actif : avertissement sans objet ici.
import logging as _logging  # noqa: E402

for _name in ("streamlit.runtime.caching.cache_data_api", "streamlit.runtime.caching.cache_resource_api"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

# =============================================================================
# 1. CONSTANTES ET LIMITES
# =============================================================================
APP_NAME = "PDE NUMERICAL LABORATORY"
APP_TITLE = "PDE Numerical Laboratory"
APP_SUBTITLE = "Interactive Environment for Numerical Analysis of Partial Differential Equations"
APP_VERSION = "1.0.0"

LIMITS: dict[str, float] = {
    "n1d_max": 4096,            # nombre maximal d'intervalles en 1D
    "n2d_max": 256,             # nombre maximal de mailles par direction en 2D
    "fem2d_elems_max": 140_000,  # nombre maximal de triangles
    "nt_max": 200_000,          # nombre maximal de pas de temps
    "work_max": 2.5e8,          # (pas de temps) x (inconnues)
    "dense_eig_max": 1200,      # taille max. pour un calcul dense de spectre symétrique
    "dense_eig_nonsym_max": 700,
    "matrix_table_max": 14,     # taille max. d'une matrice affichée intégralement
    "dense_csv_max": 400,       # taille max. pour un export CSV dense
    "frames_max": 120,
    "frame_values_max": 4_000_000,
    "blowup_factor": 1e6,       # |U| > 1e6 × échelle des données ⇒ explosion numérique
}

SIDE_LABELS = {"left": "gauche", "right": "droite", "bottom": "bas", "top": "haut", "boundary": "∂Ω"}
SIDE_NORMALS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}
BC_LABELS = {"dirichlet": "Dirichlet", "neumann": "Neumann", "robin": "Robin (Fourier)"}


class ExpressionError(ValueError):
    """Erreur de syntaxe ou de sémantique dans une expression saisie par l'utilisateur."""


class ConfigError(ValueError):
    """Configuration numérique invalide ou hors des limites autorisées."""


# =============================================================================
# 2. ANALYSE D'EXPRESSIONS SÉCURISÉE
#    Tokeniseur + analyseur récursif descendant construisant directement des objets
#    SymPy à partir d'une liste blanche. Aucune chaîne utilisateur n'est jamais
#    transmise à eval, exec, sympify ou parse_expr.
# =============================================================================
X_SYM, Y_SYM, T_SYM = sympy.symbols("x y t", real=True)
SYMBOLS = {"x": X_SYM, "y": Y_SYM, "t": T_SYM}


def _heaviside(a):
    return sympy.Heaviside(a, sympy.Rational(1, 2))


# nom -> (constructeur SymPy, nb. min. d'arguments, nb. max. d'arguments)
FUNCTION_WHITELIST: dict[str, tuple[Callable, int, int]] = {
    "sin": (sympy.sin, 1, 1), "cos": (sympy.cos, 1, 1), "tan": (sympy.tan, 1, 1),
    "exp": (sympy.exp, 1, 1), "log": (sympy.log, 1, 1), "ln": (sympy.log, 1, 1),
    "sqrt": (sympy.sqrt, 1, 1), "abs": (sympy.Abs, 1, 1),
    "sinh": (sympy.sinh, 1, 1), "cosh": (sympy.cosh, 1, 1), "tanh": (sympy.tanh, 1, 1),
    "atan": (sympy.atan, 1, 1), "arctan": (sympy.atan, 1, 1),
    "asin": (sympy.asin, 1, 1), "arcsin": (sympy.asin, 1, 1),
    "acos": (sympy.acos, 1, 1), "arccos": (sympy.acos, 1, 1),
    "atan2": (sympy.atan2, 2, 2), "sign": (sympy.sign, 1, 1), "heaviside": (_heaviside, 1, 1),
    "min": (sympy.Min, 2, 8), "max": (sympy.Max, 2, 8),
}
CONSTANT_WHITELIST = {"pi": sympy.pi, "e": sympy.E}
_MAX_EXPR_LEN = 400
_MAX_TOKENS = 400
_MAX_DEPTH = 40
_MAX_ABS_EXPONENT = 64

_TOKEN_RE = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>\*\*|[+\-*/^(),])"
)


class _Token:
    """Jeton lexical : kind ∈ {'num', 'id', 'op', 'end'}."""
    __slots__ = ("kind", "text", "pos")

    def __init__(self, kind: str, text: str, pos: int):
        self.kind = kind
        self.text = text
        self.pos = pos


def _tokenize(src: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise ExpressionError(f"caractère non autorisé « {src[pos]} » (position {pos + 1})")
        kind = m.lastgroup
        text = m.group(kind)
        if kind != "ws":
            if kind == "op" and text == "**":
                text = "^"
            tokens.append(_Token(kind, text, pos + 1))
        pos = m.end()
        if len(tokens) > _MAX_TOKENS:
            raise ExpressionError("expression trop longue")
    # multiplication implicite : 2x, 2(x+1), (x+1)(x-1), pi x, x(1-x)
    out: list[_Token] = []
    for i, tok in enumerate(tokens):
        if out:
            prev = out[-1]
            prev_is_value = prev.kind == "num" or (prev.kind == "op" and prev.text == ")") or (
                prev.kind == "id" and prev.text not in FUNCTION_WHITELIST)
            cur_starts_value = tok.kind in ("num", "id") or (tok.kind == "op" and tok.text == "(")
            if prev_is_value and cur_starts_value:
                if prev.kind == "num" and tok.kind == "num":
                    raise ExpressionError(f"deux nombres consécutifs sans opérateur (position {tok.pos})")
                out.append(_Token("op", "*", tok.pos))
        out.append(tok)
    out.append(_Token("end", "", len(src) + 1))
    return out


def _make_number(text: str, pos: int) -> sympy.Expr:
    try:
        val = float(text)
    except ValueError as exc:  # pragma: no cover - le tokeniseur garantit le format
        raise ExpressionError(f"nombre invalide « {text} »") from exc
    if not math.isfinite(val) or abs(val) > 1e100:
        raise ExpressionError(f"nombre hors limites « {text} » (position {pos})")
    if re.fullmatch(r"\d+", text):
        return sympy.Integer(int(text))
    return sympy.Rational(repr(val)) if "e" in text.lower() else sympy.Rational(text)


class _Parser:
    """Analyseur récursif descendant :
        expr  := term (('+'|'-') term)*
        term  := unary (('*'|'/') unary)*
        unary := ('+'|'-') unary | power
        power := atom ('^' unary)?
        atom  := NOMBRE | CONSTANTE | VARIABLE | FONCTION '(' expr (',' expr)* ')' | '(' expr ')'
    """

    def __init__(self, tokens: list[_Token], allowed: tuple[str, ...]):
        self.toks = tokens
        self.i = 0
        self.allowed = allowed
        self.depth = 0

    def peek(self) -> _Token:
        return self.toks[self.i]

    def take(self) -> _Token:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def expect(self, text: str) -> None:
        tok = self.take()
        if tok.text != text:
            found = "fin de l'expression" if tok.kind == "end" else f"« {tok.text} »"
            raise ExpressionError(f"« {text} » attendu, {found} trouvé (position {tok.pos})")

    def parse(self) -> sympy.Expr:
        if self.peek().kind == "end":
            raise ExpressionError("expression vide")
        node = self.expr()
        tok = self.peek()
        if tok.kind != "end":
            raise ExpressionError(f"symbole inattendu « {tok.text} » (position {tok.pos})")
        return node

    def expr(self) -> sympy.Expr:
        node = self.term()
        while self.peek().kind == "op" and self.peek().text in "+-":
            op = self.take().text
            rhs = self.term()
            node = node + rhs if op == "+" else node - rhs
        return node

    def term(self) -> sympy.Expr:
        node = self.unary()
        while self.peek().kind == "op" and self.peek().text in "*/":
            op = self.take()
            rhs = self.unary()
            if op.text == "*":
                node = node * rhs
            else:
                if rhs.is_number and rhs == 0:
                    raise ExpressionError(f"division par zéro (position {op.pos})")
                node = node / rhs
        return node

    def unary(self) -> sympy.Expr:
        tok = self.peek()
        if tok.kind == "op" and tok.text in "+-":
            self.take()
            self._enter(tok)
            val = self.unary()
            self.depth -= 1
            return -val if tok.text == "-" else val
        return self.power()

    def power(self) -> sympy.Expr:
        base = self.atom()
        tok = self.peek()
        if tok.kind == "op" and tok.text == "^":
            self.take()
            self._enter(tok)
            expo = self.unary()
            self.depth -= 1
            if expo.is_number:
                ev = complex(sympy.N(expo))
                if abs(ev.imag) > 0 or abs(ev.real) > _MAX_ABS_EXPONENT:
                    raise ExpressionError(f"exposant hors limites (|p| ≤ {_MAX_ABS_EXPONENT}) (position {tok.pos})")
                if base.is_number and base == 0 and ev.real < 0:
                    raise ExpressionError(f"0 élevé à une puissance négative (position {tok.pos})")
            return sympy.Pow(base, expo)
        return base

    def _enter(self, tok: _Token) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise ExpressionError(f"imbrication trop profonde (position {tok.pos})")

    def atom(self) -> sympy.Expr:
        tok = self.take()
        if tok.kind == "num":
            return _make_number(tok.text, tok.pos)
        if tok.kind == "id":
            name = tok.text
            if name in FUNCTION_WHITELIST:
                ctor, nmin, nmax = FUNCTION_WHITELIST[name]
                nxt = self.peek()
                if not (nxt.kind == "op" and nxt.text == "("):
                    raise ExpressionError(f"la fonction « {name} » doit être suivie de parenthèses (position {tok.pos})")
                self.take()
                self._enter(tok)
                args = [self.expr()]
                while self.peek().kind == "op" and self.peek().text == ",":
                    self.take()
                    args.append(self.expr())
                self.expect(")")
                self.depth -= 1
                if not (nmin <= len(args) <= nmax):
                    nb = f"{nmin}" if nmin == nmax else f"{nmin} à {nmax}"
                    raise ExpressionError(f"« {name} » attend {nb} argument(s), {len(args)} fourni(s)")
                return ctor(*args)
            if name in CONSTANT_WHITELIST:
                return CONSTANT_WHITELIST[name]
            if name in SYMBOLS:
                if name not in self.allowed:
                    perm = ", ".join(self.allowed) if self.allowed else "aucune (constante attendue)"
                    raise ExpressionError(f"la variable « {name} » n'est pas autorisée ici — variables permises : {perm}")
                return SYMBOLS[name]
            known = list(FUNCTION_WHITELIST) + list(CONSTANT_WHITELIST) + list(self.allowed)
            hint = difflib.get_close_matches(name, known, n=1)
            extra = f" — vouliez-vous dire « {hint[0]} » ?" if hint else ""
            if len(name) > 1 and all(ch in "xyt" for ch in name):
                extra = " — utilisez « * » pour un produit, par exemple « x*y »"
            raise ExpressionError(f"symbole inconnu « {name} » (position {tok.pos}){extra}")
        if tok.kind == "op" and tok.text == "(":
            self._enter(tok)
            node = self.expr()
            self.expect(")")
            self.depth -= 1
            return node
        if tok.kind == "end":
            raise ExpressionError("expression incomplète")
        raise ExpressionError(f"symbole inattendu « {tok.text} » (position {tok.pos})")


def parse_expression(src: Any, allowed: tuple[str, ...] = ("x", "y", "t")) -> sympy.Expr:
    """Analyse sûre d'une expression mathématique (liste blanche stricte)."""
    if src is None:
        raise ExpressionError("expression vide")
    text = str(src).strip()
    if not text:
        raise ExpressionError("expression vide")
    if len(text) > _MAX_EXPR_LEN:
        raise ExpressionError(f"expression trop longue (maximum {_MAX_EXPR_LEN} caractères)")
    expr = _Parser(_tokenize(text), allowed).parse()
    if expr.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo):
        raise ExpressionError("l'expression contient une valeur infinie ou indéterminée")
    return expr


def parse_constant(src: Any) -> float:
    """Analyse une constante numérique (aucune variable autorisée)."""
    expr = parse_expression(src, allowed=())
    val = complex(sympy.N(expr))
    if abs(val.imag) > 1e-14 or not math.isfinite(val.real):
        raise ExpressionError("la constante doit être un réel fini")
    return float(val.real)


# --- compilation numérique d'un arbre SymPy validé (aucun exec) ------------------
_NP_UNARY: dict[Any, Callable] = {
    sympy.sin: np.sin, sympy.cos: np.cos, sympy.tan: np.tan, sympy.exp: np.exp,
    sympy.log: np.log, sympy.Abs: np.abs, sympy.sinh: np.sinh, sympy.cosh: np.cosh,
    sympy.tanh: np.tanh, sympy.atan: np.arctan, sympy.asin: np.arcsin, sympy.acos: np.arccos,
    sympy.sign: np.sign, sympy.cot: lambda v: 1.0 / np.tan(v), sympy.sec: lambda v: 1.0 / np.cos(v),
    sympy.csc: lambda v: 1.0 / np.sin(v),
}


def _compile_node(e: sympy.Expr) -> Callable[[dict], Any]:
    if e.is_number:
        val = complex(sympy.N(e))
        if abs(val.imag) > 1e-14 * max(1.0, abs(val.real)):
            raise ExpressionError(f"valeur complexe rencontrée : {e}")
        v = float(val.real)
        return lambda env, v=v: v
    if e.is_Symbol:
        name = e.name
        return lambda env, name=name: env[name]
    if e.is_Add:
        parts = [_compile_node(a) for a in e.args]

        def _add(env, parts=parts):
            out = parts[0](env)
            for g in parts[1:]:
                out = out + g(env)
            return out
        return _add
    if e.is_Mul:
        parts = [_compile_node(a) for a in e.args]

        def _mul(env, parts=parts):
            out = parts[0](env)
            for g in parts[1:]:
                out = out * g(env)
            return out
        return _mul
    if e.is_Pow:
        fb = _compile_node(e.base)
        if e.exp.is_number:
            p = float(sympy.N(e.exp))
            if p == 0.5:
                return lambda env, fb=fb: np.sqrt(fb(env))
            if p == -0.5:
                return lambda env, fb=fb: 1.0 / np.sqrt(fb(env))
            if p == -1.0:
                return lambda env, fb=fb: 1.0 / fb(env)
            if p == 2.0:
                return lambda env, fb=fb: fb(env) * fb(env)
            if float(p).is_integer():
                return lambda env, fb=fb, p=p: np.power(fb(env), int(p)) if int(p) >= 0 else 1.0 / np.power(fb(env), -int(p))
            return lambda env, fb=fb, p=p: np.power(fb(env), p)
        fe = _compile_node(e.exp)
        return lambda env, fb=fb, fe=fe: np.power(fb(env), fe(env))
    if isinstance(e, sympy.Heaviside):
        fa = _compile_node(e.args[0])
        h0 = float(sympy.N(e.args[1])) if len(e.args) > 1 else 0.5
        return lambda env, fa=fa, h0=h0: np.heaviside(fa(env), h0)
    if isinstance(e, sympy.atan2):
        fa, fb2 = _compile_node(e.args[0]), _compile_node(e.args[1])
        return lambda env, fa=fa, fb2=fb2: np.arctan2(fa(env), fb2(env))
    if isinstance(e, sympy.Min):
        parts = [_compile_node(a) for a in e.args]
        return lambda env, parts=parts: _reduce_np(np.minimum, parts, env)
    if isinstance(e, sympy.Max):
        parts = [_compile_node(a) for a in e.args]
        return lambda env, parts=parts: _reduce_np(np.maximum, parts, env)
    if e.func in _NP_UNARY and len(e.args) == 1:
        fa = _compile_node(e.args[0])
        npf = _NP_UNARY[e.func]
        return lambda env, fa=fa, npf=npf: npf(fa(env))
    raise ExpressionError(
        f"terme non évaluable numériquement : {e.func.__name__} "
        "(par ex. une dérivée de fonction discontinue produisant une masse de Dirac)")


def _reduce_np(op: Callable, parts: list[Callable], env: dict) -> Any:
    out = parts[0](env)
    for g in parts[1:]:
        out = op(out, g(env))
    return out


class NumFunc:
    """Fonction numérique vectorisée f(x, y, t) compilée à partir d'une expression SymPy validée."""

    def __init__(self, expr: sympy.Expr):
        if not isinstance(expr, sympy.Basic):  # jamais de conversion de chaîne : arbre SymPy validé uniquement
            raise TypeError("NumFunc attend une expression SymPy construite par l'analyseur sécurisé")
        self.expr = expr
        self._f = _compile_node(self.expr)
        self.symbols = {s.name for s in self.expr.free_symbols}
        self.is_constant = len(self.symbols) == 0
        self.depends_t = "t" in self.symbols
        self.const_value = float(self._f({})) if self.is_constant else None

    def __call__(self, x: Any, y: Any = None, t: Any = 0.0) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.zeros_like(xa) if y is None else np.asarray(y, dtype=float)
        env = {"x": xa, "y": ya, "t": t}
        with np.errstate(all="ignore"):
            val = self._f(env)
        val = np.asarray(val, dtype=float)
        shape = np.broadcast_shapes(xa.shape, ya.shape, np.shape(t))
        if val.shape != shape:
            val = np.broadcast_to(val, shape).copy()
        return val

    def latex(self) -> str:
        return sympy.latex(self.expr)


def zero_func() -> NumFunc:
    return NumFunc(sympy.Integer(0))


def soft_simplify(e: sympy.Expr, max_ops: int = 160) -> sympy.Expr:
    """Simplification symbolique bornée (évite les temps de calcul excessifs)."""
    try:
        if sympy.count_ops(e) > max_ops:
            return e
        s = sympy.simplify(e)
        return s if sympy.count_ops(s) <= sympy.count_ops(e) else e
    except Exception:  # noqa: BLE001
        return e


def latex_of(e: Any) -> str:
    try:
        return sympy.latex(e)
    except Exception:  # noqa: BLE001
        return str(e)


def is_smooth_expr(e: sympy.Expr) -> bool:
    """Vrai si l'expression n'utilise que des fonctions analytiques (pas |.|, sign, H, min, max)."""
    return not e.has(sympy.Abs, sympy.sign, sympy.Heaviside, sympy.Min, sympy.Max)


# =============================================================================
# 3. CATALOGUE DES EDP, CONFIGURATION, MODÈLE DE PROBLÈME
# =============================================================================
CLASS_LABELS = {
    "elliptic": "Elliptique",
    "parabolic": "Parabolique",
    "hyperbolic": "Hyperbolique (second ordre)",
    "transport": "Hyperbolique du premier ordre",
}

FAMILIES: dict[str, dict[str, Any]] = {
    "poisson": {
        "label": "Équation de Poisson", "class": "elliptic", "time_order": 0,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("f",),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Problème stationnaire modèle : potentiel électrostatique, déformation d'une membrane, "
                   "température d'équilibre.",
    },
    "laplace": {
        "label": "Équation de Laplace", "class": "elliptic", "time_order": 0,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": (),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Fonctions harmoniques : seules les conditions aux limites pilotent la solution.",
    },
    "diffusion_reaction": {
        "label": "Diffusion–réaction stationnaire", "class": "elliptic", "time_order": 0,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("k", "c", "f"),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Diffusion hétérogène k(x,y) et réaction (absorption) c(x,y).",
    },
    "heat": {
        "label": "Équation de la chaleur", "class": "parabolic", "time_order": 1,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("kappa", "f", "u0"),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Diffusion thermique homogène de diffusivité κ.",
    },
    "parabolic_dr": {
        "label": "Diffusion–réaction instationnaire", "class": "parabolic", "time_order": 1,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("k", "c", "f", "u0"),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Évolution d'une concentration soumise à diffusion hétérogène et réaction.",
    },
    "advection_diffusion": {
        "label": "Advection–diffusion", "class": "parabolic", "time_order": 1,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("k", "b", "f", "u0"),
        "bc_types": ("dirichlet", "neumann", "robin"),
        "summary": "Transport par un champ de vitesse constant b et diffusion k(x,y).",
    },
    "wave": {
        "label": "Équation des ondes", "class": "hyperbolic", "time_order": 2,
        "dims": (1, 2), "methods": ("fdm", "fem"), "uses": ("cw", "f", "u0", "v0"),
        "bc_types": ("dirichlet", "neumann"),
        "summary": "Propagation d'ondes à célérité c (corde vibrante, membrane).",
    },
    "transport": {
        "label": "Équation de transport", "class": "transport", "time_order": 1,
        "dims": (1,), "methods": ("fdm",), "uses": ("a", "u0"),
        "bc_types": (),
        "summary": "Convection pure à vitesse constante a : u(x,t) = u₀(x − a t).",
    },
}

FAMILY_SHORT = {"poisson": "Poisson", "laplace": "Laplace", "diffusion_reaction": "Diffusion–réaction",
                "heat": "Chaleur", "parabolic_dr": "Diff.–réaction (t)", "advection_diffusion": "Advection–diffusion",
                "wave": "Ondes", "transport": "Transport"}
METHOD_LABELS = {"fdm": "Différences finies (FDM)", "fem": "Éléments finis P1 (FEM)"}
PATTERN_LABELS = {"right": "Diagonale montante", "left": "Diagonale descendante",
                  "alternate": "Diagonales alternées", "crossed": "Croisée (4 triangles / maille)"}
GRADING_LABELS = {"uniform": "Uniforme", "perturbed": "Perturbation aléatoire (graine fixe)",
                  "chebyshev": "Resserré aux bords (Tchebychev)"}
TRANSPORT_SCHEMES = {
    "upwind": "Décentré amont (upwind)",
    "ftcs": "Centré explicite (FTCS)",
    "lax_friedrichs": "Lax–Friedrichs",
    "lax_wendroff": "Lax–Wendroff",
    "btcs": "Centré implicite (Euler implicite)",
}
WAVE_SCHEMES = {"leapfrog": "Saute-mouton centré explicite (β = 0)",
                "newmark": "Newmark accélération moyenne (β = 1/4, implicite)"}
SOLVER_LABELS = {"direct": "Direct creux (SuperLU)", "cg": "Gradient conjugué (CG)",
                 "gmres": "GMRES(50)", "bicgstab": "BiCGSTAB"}
PRECOND_LABELS = {"none": "Aucun", "jacobi": "Jacobi (diagonal)", "ilu": "ILU incomplète"}

DEFAULT_BC = {"type": "dirichlet", "g": "0", "beta": "1"}

DEFAULT_CFG: dict[str, Any] = {
    "schema": "pdelab-config",
    "version": APP_VERSION,
    "name": "Poisson 2D — test de référence",
    "family": "poisson",
    "dim": 2,
    "domain": {"shape": "rectangle", "xa": 0.0, "xb": 1.0, "ya": 0.0, "yb": 1.0},
    "coef": {"k": "1", "c": "0", "kappa": "1", "bx": "1", "by": "0", "cw": "1", "a": "1"},
    "f": "2*pi^2*sin(pi*x)*sin(pi*y)",
    "u0": "0",
    "v0": "0",
    "exact": "sin(pi*x)*sin(pi*y)",
    "manufactured": False,
    "bc": {s: dict(DEFAULT_BC) for s in ("left", "right", "bottom", "top", "boundary")},
    "transport_bc": "periodic",
    "method": "fem",
    "mesh": {"nx": 16, "ny": 16, "pattern": "right", "grading": "uniform", "perturb": 0.25,
             "seed": 7, "refine": 0},
    "time": {"T": 0.5, "dt_mode": "number", "number": 0.4, "dt": 0.001, "nt": 200, "theta": 0.5,
             "lumped": False, "wave_scheme": "leapfrog", "transport_scheme": "upwind", "frames": 60},
    "adv_scheme": "centered",
    "solver": {"kind": "direct", "precond": "none", "tol": 1e-10, "maxiter": 5000},
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CFG)


def deep_merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (upd or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def config_json(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(config_json(cfg).encode("utf-8")).hexdigest()[:12]


def sides_of(cfg: dict) -> list[str]:
    if cfg["dim"] == 1:
        return ["left", "right"]
    if cfg["domain"].get("shape") == "lshape":
        return ["boundary"]
    return ["left", "right", "bottom", "top"]


def _clip_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        iv = default
    return int(min(max(iv, lo), hi))


def _clip_float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        fv = float(v)
        if not math.isfinite(fv):
            fv = default
    except (TypeError, ValueError):
        fv = default
    return float(min(max(fv, lo), hi))


def normalize_config(cfg_in: dict) -> tuple[dict, list[str]]:
    """Complète et rend cohérente une configuration ; renvoie (configuration, notes)."""
    notes: list[str] = []
    cfg = deep_merge(DEFAULT_CFG, cfg_in or {})
    fam = cfg.get("family")
    if fam not in FAMILIES:
        notes.append(f"Famille inconnue « {fam} » remplacée par Poisson.")
        cfg["family"] = fam = "poisson"
    meta = FAMILIES[fam]
    if cfg.get("dim") not in meta["dims"]:
        cfg["dim"] = meta["dims"][0]
        notes.append(f"Dimension ajustée à {cfg['dim']}D (seule dimension prise en charge pour cette famille).")
    if cfg.get("method") not in meta["methods"]:
        cfg["method"] = meta["methods"][0]
        notes.append(f"Méthode ajustée : {METHOD_LABELS[cfg['method']]}.")
    dom = cfg["domain"]
    if cfg["dim"] == 1:
        dom["shape"] = "interval"
    elif dom.get("shape") not in ("rectangle", "lshape"):
        dom["shape"] = "rectangle"
    if dom["shape"] == "lshape":
        if meta["class"] != "elliptic":
            dom["shape"] = "rectangle"
            notes.append("Domaine en L réservé aux problèmes elliptiques : rectangle rétabli.")
        else:
            if cfg["method"] != "fem":
                cfg["method"] = "fem"
                notes.append("Domaine en L : seule la méthode des éléments finis est disponible.")
            dom.update({"xa": -1.0, "xb": 1.0, "ya": -1.0, "yb": 1.0})
            cfg["bc"]["boundary"]["type"] = "dirichlet"
    for key in ("xa", "xb", "ya", "yb"):
        dom[key] = _clip_float(dom.get(key), -1e3, 1e3, DEFAULT_CFG["domain"][key])
    if dom["xb"] - dom["xa"] < 1e-6:
        dom["xa"], dom["xb"] = 0.0, 1.0
        notes.append("Intervalle en x invalide (x_b ≤ x_a) : [0, 1] rétabli.")
    if dom["yb"] - dom["ya"] < 1e-6:
        dom["ya"], dom["yb"] = 0.0, 1.0
        notes.append("Intervalle en y invalide (y_b ≤ y_a) : [0, 1] rétabli.")
    allowed_bc = meta["bc_types"]
    for side, bc in cfg["bc"].items():
        if not isinstance(bc, dict):
            cfg["bc"][side] = dict(DEFAULT_BC)
            bc = cfg["bc"][side]
        bc.setdefault("g", "0")
        bc.setdefault("beta", "1")
        if allowed_bc and bc.get("type") not in allowed_bc:
            bc["type"] = "dirichlet"
    if cfg.get("transport_bc") not in ("periodic", "inflow"):
        cfg["transport_bc"] = "periodic"
    mesh = cfg["mesh"]
    if cfg["dim"] == 1:
        mesh["nx"] = _clip_int(mesh.get("nx"), 2, int(LIMITS["n1d_max"]), 32)
    else:
        mesh["nx"] = _clip_int(mesh.get("nx"), 2, int(LIMITS["n2d_max"]), 16)
        mesh["ny"] = _clip_int(mesh.get("ny"), 2, int(LIMITS["n2d_max"]), 16)
        if dom["shape"] == "lshape" and mesh["nx"] % 2:
            mesh["nx"] += 1
            notes.append("Domaine en L : nombre de mailles rendu pair pour que l'angle rentrant soit un nœud.")
    if mesh.get("pattern") not in PATTERN_LABELS:
        mesh["pattern"] = "right"
    if mesh.get("grading") not in GRADING_LABELS:
        mesh["grading"] = "uniform"
    mesh["perturb"] = _clip_float(mesh.get("perturb"), 0.0, 0.3, 0.25)
    mesh["seed"] = _clip_int(mesh.get("seed"), 0, 10_000, 7)
    mesh["refine"] = _clip_int(mesh.get("refine"), 0, 4, 0)
    if cfg["method"] == "fdm":
        mesh["refine"] = 0
        if mesh["grading"] != "uniform":
            mesh["grading"] = "uniform"
            notes.append("Différences finies : grille uniforme imposée.")
    tm = cfg["time"]
    tm["T"] = _clip_float(tm.get("T"), 1e-6, 1e4, 0.5)
    if tm.get("dt_mode") not in ("number", "dt", "nt"):
        tm["dt_mode"] = "number"
    tm["number"] = _clip_float(tm.get("number"), 1e-6, 1e4, 0.4)
    tm["dt"] = _clip_float(tm.get("dt"), 1e-9, 1e3, 1e-3)
    tm["nt"] = _clip_int(tm.get("nt"), 1, int(LIMITS["nt_max"]), 100)
    tm["theta"] = _clip_float(tm.get("theta"), 0.0, 1.0, 0.5)
    tm["lumped"] = bool(tm.get("lumped", False))
    if tm.get("wave_scheme") not in WAVE_SCHEMES:
        tm["wave_scheme"] = "leapfrog"
    if tm.get("transport_scheme") not in TRANSPORT_SCHEMES:
        tm["transport_scheme"] = "upwind"
    tm["frames"] = _clip_int(tm.get("frames"), 10, int(LIMITS["frames_max"]), 60)
    if cfg.get("adv_scheme") not in ("centered", "upwind"):
        cfg["adv_scheme"] = "centered"
    sol = cfg["solver"]
    if sol.get("kind") not in SOLVER_LABELS:
        sol["kind"] = "direct"
    if sol.get("precond") not in PRECOND_LABELS:
        sol["precond"] = "none"
    sol["tol"] = _clip_float(sol.get("tol"), 1e-14, 1e-2, 1e-10)
    sol["maxiter"] = _clip_int(sol.get("maxiter"), 10, 100_000, 5000)
    cfg["manufactured"] = bool(cfg.get("manufactured", False))
    for key in ("f", "u0", "v0", "exact"):
        cfg[key] = "" if cfg.get(key) is None else str(cfg.get(key))
    cfg["schema"] = "pdelab-config"
    cfg["version"] = APP_VERSION
    return cfg, notes


def _normal_derivative(u: sympy.Expr, side: str) -> sympy.Expr:
    if side == "left":
        return -sympy.diff(u, X_SYM)
    if side == "right":
        return sympy.diff(u, X_SYM)
    if side == "bottom":
        return -sympy.diff(u, Y_SYM)
    if side == "top":
        return sympy.diff(u, Y_SYM)
    raise ValueError(side)


class Problem:
    """Problème aux dérivées partielles entièrement résolu (expressions SymPy + fonctions numériques)."""

    def __init__(self, cfg_in: dict):
        self.cfg, self.notes = normalize_config(cfg_in)
        cfg = self.cfg
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.family = cfg["family"]
        self.meta = FAMILIES[self.family]
        self.cls = self.meta["class"]
        self.time_order = self.meta["time_order"]
        self.time_dependent = self.time_order > 0
        self.dim = cfg["dim"]
        self.method = cfg["method"]
        dom = cfg["domain"]
        self.shape = dom["shape"]
        self.xa, self.xb, self.ya, self.yb = dom["xa"], dom["xb"], dom["ya"], dom["yb"]
        self.sides = sides_of(cfg)
        self.space_vars: tuple[str, ...] = ("x",) if self.dim == 1 else ("x", "y")
        self.all_vars: tuple[str, ...] = self.space_vars + (("t",) if self.time_dependent else ())
        self.T = cfg["time"]["T"] if self.time_dependent else 0.0
        uses = self.meta["uses"]
        zero = sympy.Integer(0)
        # ---------------- coefficients ----------------
        self.k_expr: sympy.Expr = sympy.Integer(1)
        self.c_expr: sympy.Expr = zero
        self.bx = 0.0
        self.by = 0.0
        self.kappa = None
        self.cw = None
        self.a = None
        coef = cfg["coef"]
        self.k_expr = self._parse("k", coef.get("k"), self.space_vars) if "k" in uses else sympy.Integer(1)
        if "kappa" in uses:
            self.kappa = self._const("kappa", coef.get("kappa"))
            if self.kappa is not None:
                if self.kappa <= 0:
                    self.errors.append("κ doit être strictement positif (κ ≤ 0 : équation rétrograde, mal posée).")
                self.k_expr = sympy.nsimplify(self.kappa, rational=True) if self.kappa > 0 else sympy.Integer(1)
        if "cw" in uses:
            self.cw = self._const("c (célérité)", coef.get("cw"))
            if self.cw is not None and self.cw <= 0:
                self.errors.append("La célérité c doit être strictement positive.")
            if self.cw is not None and self.cw > 0:
                self.k_expr = sympy.nsimplify(self.cw, rational=True) ** 2
        if "c" in uses:
            self.c_expr = self._parse("c", coef.get("c"), self.space_vars)
        if "b" in uses:
            bx = self._const("b_x", coef.get("bx"))
            by = self._const("b_y", coef.get("by")) if self.dim == 2 else 0.0
            self.bx = bx if bx is not None else 0.0
            self.by = by if by is not None else 0.0
        if "a" in uses:
            self.a = self._const("a", coef.get("a"))
            if self.a is not None and abs(self.a) < 1e-14:
                self.errors.append("La vitesse de transport a doit être non nulle.")
        # ---------------- solution exacte ----------------
        self.exact_expr: sympy.Expr | None = None
        if cfg["exact"].strip():
            self.exact_expr = self._parse("solution exacte", cfg["exact"], self.all_vars)
        self.manufactured = bool(cfg["manufactured"]) and self.exact_expr is not None
        if cfg["manufactured"] and self.exact_expr is None:
            self.warnings.append("Solution manufacturée demandée sans solution exacte : option ignorée.")
        # ---------------- données ----------------
        self.f_expr: sympy.Expr = zero
        self.u0_expr: sympy.Expr = zero
        self.v0_expr: sympy.Expr = zero
        self.bc: dict[str, dict[str, Any]] = {}
        fvars = self.all_vars
        if self.manufactured and not self.errors:
            u = self.exact_expr
            if self.cls == "transport":
                res = soft_simplify(sympy.diff(u, T_SYM) + sympy.nsimplify(self.a or 0.0, rational=True) * sympy.diff(u, X_SYM))
                if res != 0:
                    self.warnings.append("La solution exacte fournie ne vérifie pas symboliquement u_t + a u_x = 0 "
                                         "(l'équation de transport traitée est homogène).")
            elif "f" in uses:
                self.f_expr = soft_simplify(self.apply_operator(u))
            elif self.family == "laplace":
                res = soft_simplify(self.apply_operator(u))
                if res != 0:
                    self.warnings.append("La solution fournie n'est pas harmonique (−Δu ≠ 0) : "
                                         "utilisez l'équation de Poisson pour une solution manufacturée générale.")
            if self.time_dependent:
                self.u0_expr = soft_simplify(u.subs(T_SYM, 0))
                if self.time_order == 2:
                    self.v0_expr = soft_simplify(sympy.diff(u, T_SYM).subs(T_SYM, 0))
        else:
            if "f" in uses:
                self.f_expr = self._parse("f", cfg["f"], fvars)
            if "u0" in uses:
                self.u0_expr = self._parse("u₀", cfg["u0"], self.space_vars)
            if "v0" in uses:
                self.v0_expr = self._parse("v₀", cfg["v0"], self.space_vars)
        # conditions aux limites
        if self.cls != "transport":
            for side in self.sides:
                bcc = cfg["bc"][side]
                typ = bcc["type"]
                beta = zero
                if typ == "robin":
                    beta = self._parse(f"β ({SIDE_LABELS[side]})", bcc.get("beta", "1"), self.space_vars)
                if self.manufactured and not self.errors:
                    if typ == "dirichlet":
                        g = self.exact_expr
                    else:
                        g = self.k_expr * _normal_derivative(self.exact_expr, side)
                        if typ == "robin":
                            g = g + beta * self.exact_expr
                        g = soft_simplify(g)
                else:
                    g = self._parse(f"g ({SIDE_LABELS[side]})", bcc.get("g", "0"), fvars)
                self.bc[side] = {"type": typ, "g": g, "beta": beta}
        else:
            self.transport_bc = cfg["transport_bc"]
            if self.transport_bc == "inflow":
                if self.exact_expr is not None:
                    self.inflow_expr = self.exact_expr
                else:
                    self.inflow_expr = None  # données entrantes : u0 transporté (voir solveur)
        # ---------------- compilation numérique ----------------
        self.k = self._compile(self.k_expr, "k")
        self.c = self._compile(self.c_expr, "c")
        self.f = self._compile(self.f_expr, "f")
        self.u0 = self._compile(self.u0_expr, "u₀")
        self.v0 = self._compile(self.v0_expr, "v₀")
        self.exact = self._compile(self.exact_expr, "solution exacte") if self.exact_expr is not None else None
        self.exact_dx = self.exact_dy = self.exact_dt = None
        if self.exact is not None:
            self.exact_dx = self._compile(sympy.diff(self.exact_expr, X_SYM), "∂u/∂x")
            self.exact_dy = self._compile(sympy.diff(self.exact_expr, Y_SYM), "∂u/∂y")
            if self.time_dependent:
                self.exact_dt = self._compile(sympy.diff(self.exact_expr, T_SYM), "∂u/∂t")
        self.bcf: dict[str, dict[str, Any]] = {}
        for side, b in self.bc.items():
            self.bcf[side] = {"type": b["type"], "g": self._compile(b["g"], "g"),
                              "beta": self._compile(b["beta"], "β")}
        self.inflow = None
        if self.cls == "transport" and getattr(self, "inflow_expr", None) is not None:
            self.inflow = self._compile(self.inflow_expr, "donnée entrante")
        # ---------------- propriétés dérivées ----------------
        self.kernel_constants = False
        self.has_dirichlet = any(b["type"] == "dirichlet" for b in self.bc.values())
        self.has_robin = any(b["type"] == "robin" for b in self.bc.values())
        if not self.errors:
            self._check_data()

    # ------------------------------------------------------------------ helpers
    def _parse(self, name: str, src: Any, allowed: tuple[str, ...]) -> sympy.Expr:
        try:
            return parse_expression(src, allowed)
        except ExpressionError as exc:
            self.errors.append(f"{name} : {exc}")
            return sympy.Integer(0)

    def _const(self, name: str, src: Any) -> float | None:
        try:
            return parse_constant(src)
        except ExpressionError as exc:
            self.errors.append(f"{name} : {exc}")
            return None

    def _compile(self, expr: sympy.Expr | None, name: str) -> NumFunc | None:
        if expr is None:
            return None
        try:
            return NumFunc(expr)
        except ExpressionError as exc:
            self.errors.append(f"{name} : {exc}")
            return zero_func()

    # ----------------------------------------------------------- symbolic tools
    def apply_operator(self, u: sympy.Expr) -> sympy.Expr:
        """L(u) = ∂ₜu (ou ∂ₜₜu) − ∇·(k∇u) + b·∇u + c u (transport : ∂ₜu + a ∂ₓu)."""
        if self.cls == "transport":
            return sympy.diff(u, T_SYM) + sympy.nsimplify(self.a or 0.0, rational=True) * sympy.diff(u, X_SYM)
        k = self.k_expr
        expr = -sympy.diff(k * sympy.diff(u, X_SYM), X_SYM)
        if self.dim == 2:
            expr += -sympy.diff(k * sympy.diff(u, Y_SYM), Y_SYM)
        if self.bx:
            expr += sympy.nsimplify(self.bx, rational=True) * sympy.diff(u, X_SYM)
        if self.by:
            expr += sympy.nsimplify(self.by, rational=True) * sympy.diff(u, Y_SYM)
        expr += self.c_expr * u
        if self.time_order == 1:
            expr += sympy.diff(u, T_SYM)
        elif self.time_order == 2:
            expr += sympy.diff(u, T_SYM, 2)
        return expr

    def sample_points(self, n: int = 61) -> tuple[np.ndarray, np.ndarray]:
        """Points d'échantillonnage couvrant Ω̄ (grille fine, domaine en L respecté)."""
        if self.dim == 1:
            return np.linspace(self.xa, self.xb, max(n * 4, 101)), np.zeros(max(n * 4, 101))
        xs = np.linspace(self.xa, self.xb, n)
        ys = np.linspace(self.ya, self.yb, n)
        XX, YY = np.meshgrid(xs, ys)
        XX, YY = XX.ravel(), YY.ravel()
        if self.shape == "lshape":
            keep = ~((XX > 0) & (YY < 0))
            XX, YY = XX[keep], YY[keep]
        return XX, YY

    def coefficient_range(self, which: str = "k") -> tuple[float, float, bool]:
        """(min, max, exact) d'un coefficient : exact si constant, sinon échantillonnage fin."""
        f = {"k": self.k, "c": self.c}[which]
        if f.is_constant:
            return f.const_value, f.const_value, True
        xs, ys = self.sample_points(121)
        vals = f(xs, ys)
        return float(np.nanmin(vals)), float(np.nanmax(vals)), False

    def _check_data(self) -> None:
        xs, ys = self.sample_points(41)
        times = np.linspace(0.0, self.T, 5) if self.time_dependent else [0.0]
        checks = [("k", self.k), ("c", self.c), ("f", self.f), ("u₀", self.u0), ("v₀", self.v0)]
        if self.exact is not None:
            checks.append(("solution exacte", self.exact))
        for name, fn in checks:
            if fn is None:
                continue
            for tt in times:
                vals = fn(xs, ys, tt)
                if not np.all(np.isfinite(vals)):
                    self.errors.append(f"{name} prend des valeurs non finies (NaN/Inf) sur le domaine "
                                       f"(t = {tt:g}) : vérifiez l'expression (log, sqrt, division…).")
                    break
        if self.cls in ("elliptic", "parabolic"):
            kmin, _, _ = self.coefficient_range("k")
            if not kmin > 0:
                self.errors.append(f"Le coefficient de diffusion doit vérifier k ≥ k₀ > 0 sur Ω̄ "
                                   f"(min observé : {kmin:.3g}). Le problème n'est alors plus uniformément "
                                   "elliptique et sort du cadre traité.")
        for side, b in self.bcf.items():
            if b["type"] == "robin":
                bxs, bys = self.boundary_samples(side)
                vals = b["beta"](bxs, bys)
                if np.nanmin(vals) < 0:
                    self.warnings.append(f"β < 0 sur le bord {SIDE_LABELS[side]} : la coercivité n'est plus garantie.")
        if self.cls == "transport" and self.exact is not None and not self.manufactured:
            xx = np.linspace(self.xa, self.xb, 101)
            for tt in np.linspace(0, self.T, 4):
                ut = self.exact_dt(xx, None, tt)
                ux = self.exact_dx(xx, None, tt)
                if np.max(np.abs(ut + (self.a or 0.0) * ux)) > 1e-8 * (1 + np.max(np.abs(ux))):
                    self.warnings.append("La solution exacte fournie ne vérifie pas u_t + a u_x = 0.")
                    break

    def boundary_samples(self, side: str, n: int = 41) -> tuple[np.ndarray, np.ndarray]:
        if self.dim == 1:
            xv = self.xa if side == "left" else self.xb
            return np.array([xv]), np.array([0.0])
        s = np.linspace(0, 1, n)
        if side == "left":
            return np.full(n, self.xa), self.ya + s * (self.yb - self.ya)
        if side == "right":
            return np.full(n, self.xb), self.ya + s * (self.yb - self.ya)
        if side == "bottom":
            return self.xa + s * (self.xb - self.xa), np.full(n, self.ya)
        if side == "top":
            return self.xa + s * (self.xb - self.xa), np.full(n, self.yb)
        # bord du domaine en L
        pts = [(-1, -1), (0, -1), (0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1)]
        xs_l, ys_l = [], []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            xs_l.append(x0 + s * (x1 - x0))
            ys_l.append(y0 + s * (y1 - y0))
        return np.concatenate(xs_l), np.concatenate(ys_l)

    def exact_residuals(self) -> dict[str, float] | None:
        """Vérifie numériquement que la solution exacte satisfait l'EDP, les CL et les CI."""
        if self.exact_expr is None or self.errors:
            return None
        out: dict[str, float] = {}
        try:
            res_expr = self.apply_operator(self.exact_expr) - (self.f_expr if self.cls != "transport" else 0)
            res = NumFunc(res_expr)
            xs, ys = self.sample_points(31)
            if self.shape == "lshape":
                keep = (xs ** 2 + ys ** 2) > 1e-6
                xs, ys = xs[keep], ys[keep]
            times = np.linspace(0, self.T, 4) if self.time_dependent else [0.0]
            scale = 1.0 + max(float(np.max(np.abs(self.f(xs, ys, tt)))) for tt in times)
            out["EDP"] = max(float(np.nanmax(np.abs(res(xs, ys, tt)))) for tt in times) / scale
            for side, b in self.bc.items():
                bxs, bys = self.boundary_samples(side)
                if b["type"] == "dirichlet":
                    expr = self.exact_expr - b["g"]
                else:
                    expr = self.k_expr * _normal_derivative(self.exact_expr, side) - b["g"]
                    if b["type"] == "robin":
                        expr = expr + b["beta"] * self.exact_expr
                fn = NumFunc(expr)
                gscale = 1.0 + max(float(np.max(np.abs(NumFunc(b["g"])(bxs, bys, tt)))) for tt in times)
                out[f"CL {SIDE_LABELS[side]}"] = max(float(np.nanmax(np.abs(fn(bxs, bys, tt)))) for tt in times) / gscale
            if self.time_dependent:
                xs, ys = self.sample_points(31)
                out["CI u(·,0)"] = float(np.nanmax(np.abs(self.exact(xs, ys, 0.0) - self.u0(xs, ys))))
                if self.time_order == 2:
                    out["CI ∂ₜu(·,0)"] = float(np.nanmax(np.abs(self.exact_dt(xs, ys, 0.0) - self.v0(xs, ys))))
        except ExpressionError as exc:
            out["erreur"] = float("nan")
            self.warnings.append(f"Vérification de la solution exacte impossible : {exc}")
        return out

    # -------------------------------------------------------------- properties
    @property
    def ok(self) -> bool:
        return not self.errors

    def dirichlet_sides(self) -> list[str]:
        return [s for s, b in self.bc.items() if b["type"] == "dirichlet"]

    def is_pure_neumann(self) -> bool:
        """Problème stationnaire sans Dirichlet, sans Robin et sans réaction : noyau = constantes."""
        if self.cls != "elliptic" or self.has_dirichlet:
            return False
        if self.has_robin:
            for side, b in self.bc.items():
                if b["type"] == "robin" and not (b["beta"].is_number and b["beta"] == 0):
                    return False
        return bool(self.c_expr.is_number and self.c_expr == 0)

    def domain_measure(self) -> float:
        if self.dim == 1:
            return self.xb - self.xa
        if self.shape == "lshape":
            return 3.0
        return (self.xb - self.xa) * (self.yb - self.ya)

    def k_max(self) -> float:
        return self.coefficient_range("k")[1]

    def describe(self) -> str:
        geo = (f"[{self.xa:g}, {self.xb:g}]" if self.dim == 1 else
               ("domaine en L (−1,1)² \\ [0,1]×[−1,0]" if self.shape == "lshape" else
                f"[{self.xa:g}, {self.xb:g}] × [{self.ya:g}, {self.yb:g}]"))
        return f"{self.meta['label']} — {self.dim}D — {geo}"


# =============================================================================
# 4. CLASSIFICATION MATHÉMATIQUE
# =============================================================================
def classify_problem(prob: Problem) -> dict[str, Any]:
    """Classification d'une EDP linéaire du second ordre par la matrice des coefficients principaux."""
    out: dict[str, Any] = {"family": prob.family, "label": prob.meta["label"]}
    out["stationary"] = not prob.time_dependent
    out["n_space"] = prob.dim
    out["linear"] = True
    out["linear_reason"] = ("Les coefficients ne dépendent que des variables indépendantes (x, y, t), "
                            "jamais de u ni de ses dérivées : l'opérateur est linéaire et le principe de "
                            "superposition s'applique.")
    kmin, kmax, kexact = prob.coefficient_range("k")
    out["k_range"] = (kmin, kmax, kexact)
    smooth = is_smooth_expr(prob.k_expr) and is_smooth_expr(prob.c_expr)
    out["regularity"] = (
        "Coefficients analytiques (C^∞ sur Ω̄) : la formulation forte a un sens classique dès que u ∈ C²."
        if smooth else
        "Coefficients seulement bornés (|·|, sign, H, min ou max présents) : ∇·(k∇u) n'a pas de sens classique "
        "partout ; seule la formulation faible (k ∈ L^∞, k ≥ k₀ > 0) est pertinente.")
    out["smooth"] = smooth
    if prob.cls == "transport":
        out.update(order_space=1, order_time=1, order=1, type="Hyperbolique du premier ordre",
                   variables=["t", "x"], principal=None,
                   note=("La classification par la matrice des coefficients principaux concerne les EDP du "
                         "second ordre. L'équation de transport est du premier ordre ; elle est dite "
                         "hyperbolique car elle possède une famille de courbes caractéristiques réelles "
                         "x − a t = constante, le long desquelles u est constante."),
                   discriminant=None)
        return out
    k = prob.k_expr if kexact else sympy.Symbol("k(x,y)" if prob.dim == 2 else "k(x)", positive=True)
    ksym = sympy.Symbol("k", positive=True)
    variables: list[str] = []
    entries: list[sympy.Expr] = []
    if prob.time_order == 1:
        variables.append("t")
        entries.append(sympy.Integer(0))
    elif prob.time_order == 2:
        variables.append("t")
        entries.append(sympy.Integer(1))
    variables += list(prob.space_vars)
    entries += [-k] * prob.dim
    mat = sympy.diag(*entries)
    out["variables"] = variables
    out["principal"] = mat
    out["principal_latex"] = sympy.latex(mat)
    out["order_space"] = 2
    out["order_time"] = prob.time_order
    out["order"] = 2
    # signes des valeurs propres (k > 0 vérifié)
    signs = []
    for v, ent in zip(variables, entries):
        if ent == 0:
            signs.append(0)
        elif v == "t":
            signs.append(1)
        else:
            signs.append(-1 if kmin > 0 else (1 if kmax < 0 else None))
    out["eigen_signs"] = signs
    if None in signs:
        typ = "Type mixte ou dégénéré (hors classification classique)"
    elif all(s == signs[0] and s != 0 for s in signs):
        typ = "Elliptique"
    elif signs.count(0) == 1 and len({s for s in signs if s != 0}) == 1 and prob.time_order == 1:
        typ = "Parabolique"
    elif 0 not in signs and (signs.count(1) == 1 or signs.count(-1) == 1) and len(signs) >= 2:
        typ = "Hyperbolique"
    else:
        typ = "Hors classification classique"
    out["type"] = typ
    if len(variables) == 2:
        A_, C_ = mat[0, 0], mat[1, 1]
        B_ = 2 * mat[0, 1]
        disc = sympy.simplify(B_ ** 2 - 4 * A_ * C_)
        out["discriminant"] = {
            "vars": variables, "A": sympy.latex(A_), "B": sympy.latex(B_), "C": sympy.latex(C_),
            "value": sympy.latex(disc),
            "sign": ("< 0" if (disc.is_negative or (not kexact and typ == "Elliptique")) else
                     ("= 0" if disc == 0 else "> 0")),
        }
    else:
        out["discriminant"] = None
    out["note"] = {
        "Elliptique": "Toutes les valeurs propres de la matrice principale sont non nulles et de même signe.",
        "Parabolique": "Une valeur propre nulle (direction t, où seul ∂ₜu apparaît) et les autres de même signe.",
        "Hyperbolique": "Valeurs propres non nulles, une de signe opposé aux autres (signature (1, d)).",
    }.get(typ, "La matrice principale n'est pas de signe constant : aucune classification n'est attribuée.")
    _ = ksym
    return out


# =============================================================================
# 5. FORMULATIONS FORTE / FAIBLE, ESPACES FONCTIONNELS, LAX–MILGRAM
# =============================================================================
STATUS_META = {
    "applicable": ("Théorème applicable", "green"),
    "ok_exact": ("Hypothèse satisfaite (exacte)", "green"),
    "ok_numeric": ("Hypothèse vérifiée numériquement", "blue"),
    "not_verified": ("Hypothèse non vérifiée", "orange"),
    "not_treated": ("Cas non traité", "red"),
    "info": ("Information", "gray"),
}

GAMMA = {"left": r"\Gamma_1", "right": r"\Gamma_2", "bottom": r"\Gamma_3", "top": r"\Gamma_4",
         "boundary": r"\partial\Omega"}


def fmt(v: Any, digits: int = 4) -> str:
    """Formatage compact d'un nombre pour l'affichage."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v):
        return "∞" if v > 0 else ("−∞" if v < 0 else "NaN")
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.{digits}g}"
    return f"{v:.{max(digits - 1, 1)}e}"


def ltx_num(v: float, digits: int = 4) -> str:
    s = fmt(v, digits)
    if "e" in s:
        mant, ex = s.split("e")
        return rf"{mant}\times 10^{{{int(ex)}}}"
    return s.replace("−", "-")


def _paren(lx: str) -> str:
    return rf"\left({lx}\right)"


def _expr_lx(e: sympy.Expr, simplify: bool = True) -> str:
    return latex_of(soft_simplify(e) if simplify else e)


def _coef_lx(e: sympy.Expr) -> str:
    lx = _expr_lx(e)
    return _paren(lx) if (isinstance(e, sympy.Add) or lx.startswith("-")) else lx


def domain_latex(prob: Problem) -> str:
    if prob.dim == 1:
        dom = rf"\Omega = ({ltx_num(prob.xa)},\ {ltx_num(prob.xb)})"
    elif prob.shape == "lshape":
        dom = r"\Omega = (-1,1)^2 \setminus [0,1]\times[-1,0]\quad\text{(domaine en L, angle rentrant } 3\pi/2\text{)}"
    else:
        dom = (rf"\Omega = ({ltx_num(prob.xa)},\ {ltx_num(prob.xb)})\times"
               rf"({ltx_num(prob.ya)},\ {ltx_num(prob.yb)})")
    if prob.time_dependent:
        dom += rf",\qquad t\in(0,\ T],\ T = {ltx_num(prob.T)}"
    return dom


def side_set_latex(prob: Problem, side: str) -> str:
    if prob.dim == 1:
        return rf"x = {ltx_num(prob.xa)}" if side == "left" else rf"x = {ltx_num(prob.xb)}"
    if side == "boundary":
        return r"\partial\Omega"
    xa, xb, ya, yb = (ltx_num(v) for v in (prob.xa, prob.xb, prob.ya, prob.yb))
    return {"left": rf"\{{x={xa}\}}\times({ya},{yb})", "right": rf"\{{x={xb}\}}\times({ya},{yb})",
            "bottom": rf"({xa},{xb})\times\{{y={ya}\}}", "top": rf"({xa},{xb})\times\{{y={yb}\}}"}[side]


def pde_latex(prob: Problem, generic: bool = True) -> str:
    """Équation aux dérivées partielles (forme générique ou instanciée avec les données)."""
    fam, d = prob.family, prob.dim
    if generic:
        k, c, f = "k", "c", "f"
        kap = r"\kappa"
        cw2 = "c^2"
        bvec = r"\mathbf b" if d == 2 else "b"
        a = "a"
    else:
        k = _coef_lx(prob.k_expr)
        c = _coef_lx(prob.c_expr)
        f = _expr_lx(prob.f_expr) if prob.cls != "transport" else "0"
        kap = "" if (prob.kappa is None or prob.kappa == 1) else ltx_num(prob.kappa)
        cw2 = "" if (prob.cw or 1.0) == 1 else ltx_num((prob.cw or 1.0) ** 2)
        bvec = (rf"\begin{{pmatrix}}{ltx_num(prob.bx)}\\ {ltx_num(prob.by)}\end{{pmatrix}}" if d == 2 else ltx_num(prob.bx))
        a = ltx_num(prob.a if prob.a is not None else 1.0)

    def _signed(coef_val: float | None, coef_lx: str, term: str) -> str:
        """« + a ∂u » écrit proprement (signe, coefficient unité)."""
        if generic or coef_val is None:
            return rf" + {coef_lx}\,{term}"
        if coef_val == 1:
            return rf" + {term}"
        if coef_val == -1:
            return rf" - {term}"
        if coef_val < 0:
            return rf" - {ltx_num(-coef_val)}\,{term}"
        return rf" + {coef_lx}\,{term}"
    c_zero = (not generic) and prob.c_expr == 0
    k_one = (not generic) and prob.k_expr == 1
    if prob.time_order == 0:
        if d == 1:
            if fam in ("poisson", "laplace") or (k_one and c_zero):
                lhs = "-u''(x)"
            else:
                lhs = (r"-u''(x)" if k_one else rf"-\left({k}\,u'(x)\right)'")
                if not c_zero:
                    lhs += rf" + {c}\,u(x)"
            rhs = "0" if fam == "laplace" else (f if generic else f)
            return rf"{lhs} = {rhs}\quad\text{{dans }}\Omega"
        if fam in ("poisson", "laplace") or (k_one and c_zero):
            lhs = r"-\Delta u"
        else:
            lhs = r"-\Delta u" if k_one else rf"-\nabla\cdot\left({k}\,\nabla u\right)"
            if not c_zero:
                lhs += rf" + {c}\,u"
        rhs = "0" if fam == "laplace" else f
        return rf"{lhs} = {rhs}\quad\text{{dans }}\Omega"
    if fam == "transport":
        return (r"\partial_t u" + _signed(prob.a, a, r"\partial_x u")
                + r" = 0\quad\text{dans }\Omega\times(0,T]")
    if fam == "wave":
        lap = r"\partial_{xx}u" if d == 1 else r"\Delta u"
        return rf"\partial_{{tt}} u - {cw2}\,{lap} = {f}\quad\text{{dans }}\Omega\times(0,T]"
    lap = r"\partial_{xx}u" if d == 1 else r"\Delta u"
    divk = (rf"\partial_x\left({k}\,\partial_x u\right)" if d == 1 else rf"\nabla\cdot\left({k}\,\nabla u\right)")
    if fam == "heat":
        return rf"\partial_t u - {kap}\,{lap} = {f}\quad\text{{dans }}\Omega\times(0,T]"
    if fam == "parabolic_dr":
        mid = (lap if k_one else divk)
        cterm = "" if c_zero else rf" + {c}\,u"
        return rf"\partial_t u - {mid}{cterm} = {f}\quad\text{{dans }}\Omega\times(0,T]"
    adv = _signed(prob.bx, bvec, r"\partial_x u") if d == 1 else rf" + {bvec}\cdot\nabla u"
    mid = (lap if k_one else divk)
    return rf"\partial_t u{adv} - {mid} = {f}\quad\text{{dans }}\Omega\times(0,T]"


def flux_symbol(prob: Problem) -> str:
    if prob.family in ("poisson", "laplace"):
        return ""
    if prob.family == "heat":
        return r"\kappa\,"
    if prob.family == "wave":
        return r"c^2\,"
    return r"k\,"


def bc_latex(prob: Problem, side: str, generic: bool = False) -> str:
    """Condition aux limites sur un bord donné."""
    bc = prob.bc[side]
    typ = bc["type"]
    g = "g" if generic else _expr_lx(bc["g"])
    beta = r"\beta" if generic else _coef_lx(bc["beta"])
    kf = flux_symbol(prob)
    where = rf"\quad\text{{sur }} {GAMMA[side]}" if prob.dim == 2 else ""
    if prob.dim == 1:
        xv = ltx_num(prob.xa if side == "left" else prob.xb)
        sgn = "-" if side == "left" else ""
        ku = rf"{sgn}{kf}u'({xv})" if prob.time_order == 0 else rf"{sgn}{kf}\partial_x u({xv},t)"
        uu = rf"u({xv})" if prob.time_order == 0 else rf"u({xv},t)"
        if typ == "dirichlet":
            return rf"{uu} = {g}"
        if typ == "neumann":
            return rf"{ku} = {g}"
        return rf"{ku} + {beta}\,{uu} = {g}"
    if typ == "dirichlet":
        return rf"u = {g}{where}"
    if typ == "neumann":
        return rf"{kf}\dfrac{{\partial u}}{{\partial n}} = {g}{where}"
    return rf"{kf}\dfrac{{\partial u}}{{\partial n}} + {beta}\,u = {g}{where}"


def strong_formulation(prob: Problem) -> dict[str, Any]:
    out: dict[str, Any] = {"pde_generic": pde_latex(prob, True), "pde": pde_latex(prob, False),
                           "domain": domain_latex(prob)}
    bcs = []
    if prob.cls == "transport":
        if prob.transport_bc == "periodic":
            bcs.append(("Périodicité", rf"u({ltx_num(prob.xa)},t) = u({ltx_num(prob.xb)},t)"))
        else:
            inflow = "left" if (prob.a or 1) > 0 else "right"
            xv = ltx_num(prob.xa if inflow == "left" else prob.xb)
            gl = _expr_lx(prob.inflow_expr.subs(X_SYM, prob.xa if inflow == "left" else prob.xb)) \
                if prob.inflow is not None else r"u_0(x - a t)\ \text{prolongée}"
            bcs.append((f"Entrée ({'gauche' if inflow == 'left' else 'droite'})", rf"u({xv},t) = {gl}"))
            bcs.append(("Sortie", r"\text{aucune condition : les caractéristiques sortent du domaine}"))
    else:
        for side in prob.sides:
            label = f"{BC_LABELS[prob.bc[side]['type']]} — bord {SIDE_LABELS[side]}"
            if prob.dim == 2 and side != "boundary":
                label += f" ({side_set_latex(prob, side).replace(chr(92), '')})"
            bcs.append((label, bc_latex(prob, side)))
    out["bcs"] = bcs
    ics = []
    if prob.time_dependent:
        ics.append(rf"u(\mathbf x, 0) = u_0(\mathbf x) = {_expr_lx(prob.u0_expr)}")
        if prob.time_order == 2:
            ics.append(rf"\partial_t u(\mathbf x, 0) = v_0(\mathbf x) = {_expr_lx(prob.v0_expr)}")
    out["ics"] = ics
    # interprétation des termes
    terms: list[tuple[str, str, str]] = []
    fam = prob.family
    if prob.time_order == 1:
        terms.append((r"\partial_t u", "Terme d'évolution",
                      "Variation temporelle de la quantité u (capacité, accumulation). Rend le problème parabolique "
                      "ou hyperbolique d'ordre 1 : il faut une condition initiale."))
    if prob.time_order == 2:
        terms.append((r"\partial_{tt} u", "Inertie",
                      "Accélération (principe fondamental de la dynamique) : il faut deux conditions initiales, "
                      "u(·,0) et ∂ₜu(·,0)."))
    if fam in ("poisson", "laplace"):
        terms.append((r"-\Delta u", "Diffusion homogène",
                      "Opérateur de Laplace (k = 1) : moyenne locale, effet régularisant. Opposé de la divergence "
                      "du gradient — le signe « − » rend l'opérateur positif."))
    elif fam == "heat":
        terms.append((r"-\kappa\Delta u", "Diffusion thermique",
                      "Divergence du flux de Fourier q = −κ∇u : la chaleur va des zones chaudes vers les zones "
                      "froides ; lisse les données (effet régularisant)."))
    elif fam == "wave":
        terms.append((r"-c^2\Delta u", "Rappel élastique",
                      "Tension de la corde ou de la membrane ; c est la célérité des ondes. Pas de dissipation : "
                      "l'énergie est conservée."))
    elif fam == "transport":
        terms.append((r"a\,\partial_x u", "Transport",
                      "Advection à vitesse constante a : u est constante le long des caractéristiques "
                      "x − a t = cste. Aucun effet régularisant : les discontinuités sont transportées."))
    else:
        terms.append((r"-\nabla\cdot(k\nabla u)", "Diffusion hétérogène",
                      "Divergence du flux q = −k∇u (loi de Fourier ou de Fick) avec conductivité k(x, y) > 0."))
    if "c" in prob.meta["uses"]:
        terms.append((r"c\,u", "Réaction",
                      "Absorption (puits) si c > 0, production si c < 0. Terme d'ordre 0, symétrique."))
    if "b" in prob.meta["uses"]:
        terms.append((r"\mathbf b\cdot\nabla u", "Advection",
                      "Transport par un écoulement de vitesse b (terme d'ordre 1, non symétrique). Le nombre de "
                      "Péclet |b|L/k mesure son importance relative à la diffusion."))
    if "f" in prob.meta["uses"]:
        terms.append(("f", "Source", "Terme source volumique (donnée) : apport ou prélèvement de la quantité u."))
    out["terms"] = terms
    # inconnues, données, paramètres
    xv = "(x)" if prob.dim == 1 else "(x, y)"
    xvt = "(x, t)" if prob.dim == 1 else "(x, y, t)"
    out["unknowns"] = [f"u{xvt if prob.time_dependent else xv} : champ inconnu (scalaire)"]
    data = []
    if "f" in prob.meta["uses"]:
        data.append(f"f{xvt if prob.time_dependent else xv} = {prob.f_expr}")
    if prob.time_dependent:
        data.append(f"u₀{xv} = {prob.u0_expr}")
    if prob.time_order == 2:
        data.append(f"v₀{xv} = {prob.v0_expr}")
    for side, b in prob.bc.items():
        data.append(f"g sur le bord {SIDE_LABELS[side]} ({BC_LABELS[b['type']]}) = {b['g']}")
    out["data"] = data
    params = []
    if "k" in prob.meta["uses"]:
        params.append(f"k{xv} = {prob.k_expr} (diffusion)")
    if "c" in prob.meta["uses"]:
        params.append(f"c{xv} = {prob.c_expr} (réaction)")
    if prob.kappa is not None:
        params.append(f"κ = {fmt(prob.kappa)} (diffusivité)")
    if "b" in prob.meta["uses"]:
        params.append(f"b = ({fmt(prob.bx)}, {fmt(prob.by)})" if prob.dim == 2 else f"b = {fmt(prob.bx)}")
    if prob.cw is not None:
        params.append(f"c = {fmt(prob.cw)} (célérité)")
    if prob.a is not None:
        params.append(f"a = {fmt(prob.a)} (vitesse de transport)")
    for side, b in prob.bc.items():
        if b["type"] == "robin":
            params.append(f"β sur le bord {SIDE_LABELS[side]} = {b['beta']} (coefficient d'échange)")
    if prob.time_dependent:
        params.append(f"T = {fmt(prob.T)} (temps final)")
    out["params"] = params
    out["space_vars"] = "x" if prob.dim == 1 else "x, y"
    out["time_var"] = "t ∈ (0, T]" if prob.time_dependent else "aucune (problème stationnaire)"
    # hypothèses mathématiques
    hyps = [r"\Omega\ \text{borné, à frontière lipschitzienne (polygonale)}"]
    if prob.cls in ("elliptic", "parabolic"):
        hyps.append(r"k\in L^\infty(\Omega),\quad k(\mathbf x)\ \ge k_0 > 0\ \text{p.p.}")
    if "c" in prob.meta["uses"]:
        hyps.append(r"c\in L^\infty(\Omega)\ \ (c\ge 0\ \text{pour la coercivité immédiate})")
    if prob.cls != "transport":
        hyps.append(r"f\in L^2(\Omega)" + (r"\ \text{(pour presque tout } t)" if prob.time_dependent else ""))
        if prob.has_dirichlet:
            hyps.append(r"g_D\in H^{1/2}(\Gamma_D)\ \ \text{(trace d'une fonction de } H^1)")
        if any(b["type"] == "neumann" for b in prob.bc.values()):
            hyps.append(r"g_N\in L^2(\Gamma_N)")
        if prob.has_robin:
            hyps.append(r"\beta\in L^\infty(\Gamma_R),\ \beta\ge 0,\quad g_R\in L^2(\Gamma_R)")
    if prob.cls == "parabolic":
        hyps.append(r"u_0\in L^2(\Omega),\quad f\in L^2(0,T;L^2(\Omega))")
    if prob.cls == "hyperbolic":
        hyps.append(r"u_0\in H^1(\Omega),\ v_0\in L^2(\Omega),\ f\in L^2(0,T;L^2(\Omega))")
    if prob.cls == "transport":
        hyps.append(r"u_0\in C^1\ \text{(solution classique)}\ \text{ou}\ u_0\in L^2\ \text{(solution faible)}")
    out["hypotheses"] = hyps
    return out


# --------------------------------------------------------------------------- weak
def _robin_sides(prob: Problem) -> list[str]:
    return [s for s, b in prob.bc.items() if b["type"] == "robin"]


def _neumann_sides(prob: Problem) -> list[str]:
    return [s for s, b in prob.bc.items() if b["type"] == "neumann"]


def _dirichlet_sides(prob: Problem) -> list[str]:
    return [s for s, b in prob.bc.items() if b["type"] == "dirichlet"]


def _bnd_int(prob: Problem, side: str, integrand: str) -> str:
    if prob.dim == 1:
        xv = ltx_num(prob.xa if side == "left" else prob.xb)
        return integrand.replace("@", xv)
    return rf"\int_{{{GAMMA[side]}}} {integrand.replace('(@)', '').replace('@', '')}\,ds"


def function_spaces(prob: Problem) -> dict[str, str]:
    """Espaces de recherche et de test adaptés aux conditions aux limites."""
    dset = _dirichlet_sides(prob)
    all_d = set(dset) == set(prob.sides) and len(dset) > 0
    homog = all(prob.bc[s]["g"] == 0 for s in dset)
    gd = r"\Gamma_D" if not all_d else r"\partial\Omega"
    out: dict[str, str] = {}
    if prob.is_pure_neumann():
        out["test"] = r"V = \left\{ v\in H^1(\Omega)\ :\ \int_\Omega v\,d\mathbf x = 0 \right\}\ \simeq H^1(\Omega)/\mathbb R"
        out["trial"] = out["test"]
        out["case"] = "neumann_pur"
        out["comment"] = ("Problème de Neumann pur sans réaction : la forme bilinéaire s'annule sur les constantes. "
                          "On travaille dans l'espace quotient (ou à moyenne nulle) et la solution n'est définie "
                          "qu'à une constante additive près.")
    elif not dset:
        out["test"] = r"V = H^1(\Omega)"
        out["trial"] = r"V = H^1(\Omega)"
        out["case"] = "h1"
        out["comment"] = ("Aucune condition de Dirichlet : toutes les conditions sont naturelles (Neumann/Robin), "
                          "elles apparaissent dans les formes a et L. L'espace est H¹(Ω) entier.")
    elif all_d and homog:
        out["test"] = r"V = H_0^1(\Omega)"
        out["trial"] = r"V = H_0^1(\Omega)"
        out["case"] = "h10"
        out["comment"] = ("Dirichlet homogène sur tout le bord : la condition essentielle est incluse dans l'espace "
                          "H₀¹(Ω) ; espace de recherche et espace des fonctions tests coïncident.")
    else:
        if all_d:
            out["test"] = r"V_0 = H_0^1(\Omega)"
        else:
            out["test"] = r"V_0 = H^1_{\Gamma_D}(\Omega) = \left\{ v\in H^1(\Omega)\ :\ v|_{\Gamma_D} = 0 \right\}"
        out["trial"] = (rf"V_g = \left\{{ v\in H^1(\Omega)\ :\ v|_{{{gd}}} = g_D \right\}} = R g_D + V_0"
                        if not homog else out["test"].replace("V_0", "V_g"))
        out["case"] = "affine" if not homog else "mixed_homog"
        out["comment"] = ("Condition de Dirichlet " + ("non homogène" if not homog else "homogène") +
                          (" sur une partie du bord" if not all_d else " sur tout le bord") +
                          " : les fonctions tests s'annulent sur Γ_D ; " +
                          ("l'espace de recherche est un espace affine (translaté de V₀ par un relèvement R g_D) "
                           "et diffère de l'espace des fonctions tests." if not homog else
                           "espace de recherche et espace des tests coïncident."))
    return out


def weak_formulation(prob: Problem) -> dict[str, Any]:
    """Passage détaillé de la formulation forte à la formulation faible."""
    out: dict[str, Any] = {"applicable": prob.cls in ("elliptic", "parabolic", "hyperbolic")}
    if prob.cls == "transport":
        out["transport"] = [
            ("Formulation faible (au sens des distributions)",
             r"\int_0^T\!\!\int_\Omega u\,(\partial_t\varphi + a\,\partial_x\varphi)\,dx\,dt + "
             r"\int_\Omega u_0(x)\,\varphi(x,0)\,dx = 0\qquad\forall\varphi\in C^1_c(\overline\Omega\times[0,T))",
             "On multiplie par φ et on intègre par parties en temps et en espace : toutes les dérivées passent sur "
             "φ. Cette formulation admet des solutions discontinues (u₀ seulement L²)."),
            ("Pourquoi Lax–Milgram ne s'applique pas",
             r"\int_\Omega (a\,\partial_x v)\,v\,dx = \frac a2\int_\Omega \partial_x(v^2)\,dx = 0\quad"
             r"\text{(cas périodique)}",
             "L'opérateur a∂ₓ est antisymétrique : la forme associée n'est pas coercive. Le caractère bien posé "
             "s'obtient par la méthode des caractéristiques et l'estimation d'énergie d/dt‖u‖² = 0."),
        ]
        return out
    d = prob.dim
    nset, rset = _neumann_sides(prob), _robin_sides(prob)
    kf = flux_symbol(prob)
    kk = {"poisson": "", "laplace": "", "heat": r"\kappa\,", "wave": r"c^2\,"}.get(prob.family, r"k\,")
    has_c = "c" in prob.meta["uses"]
    has_b = "b" in prob.meta["uses"]
    dt_term = {0: "", 1: r"\partial_t u\,", 2: r"\partial_{tt} u\,"}[prob.time_order]
    steps: list[dict[str, str]] = []
    # étape 1
    steps.append({"title": "1. Formulation forte", "latex": pde_latex(prob, True),
                  "text": "Point de départ : l'équation est supposée vérifiée ponctuellement (u ∈ C²), avec les "
                          "conditions aux limites rappelées ci-dessous."})
    # étape 2 : multiplication par v
    lhs_terms = []
    if dt_term:
        lhs_terms.append(rf"{dt_term} v")
    if d == 1:
        lhs_terms.append(rf"-\left({kk}u'\right)'\,v" if prob.time_order == 0 else rf"-\partial_x\left({kk}\partial_x u\right)v")
    else:
        lhs_terms.append(rf"-\nabla\cdot\left({kk}\nabla u\right)v")
    if has_b:
        lhs_terms.append(r"(\mathbf b\cdot\nabla u)\,v" if d == 2 else r"b\,\partial_x u\,v")
    if has_c:
        lhs_terms.append(r"c\,u\,v")
    integ = r"\int_\Omega " if d == 2 else rf"\int_{{{ltx_num(prob.xa)}}}^{{{ltx_num(prob.xb)}}} "
    dx = r"\,d\mathbf x" if d == 2 else r"\,dx"
    rhs_f = rf"{integ} f\,v{dx}" if "f" in prob.meta["uses"] else "0"
    steps.append({"title": "2. Multiplication par une fonction test v et intégration sur Ω",
                  "latex": rf"{integ}\Big[{' + '.join(lhs_terms)}\Big]{dx} = {rhs_f}"
                           r"\qquad \forall v\ \text{(test)},\ v = 0\ \text{sur}\ \Gamma_D",
                  "text": "On ne demande plus l'égalité point par point mais en moyenne pondérée contre toute "
                          "fonction test v. Les fonctions tests s'annulent là où u est imposée (Γ_D) : on n'y "
                          "teste pas l'équation puisque la valeur de u y est déjà connue."})
    # étape 3 : Green
    if d == 1:
        green = (rf"-{integ}\left({kk}u'\right)'v\,dx = {integ}{kk}u'\,v'\,dx - "
                 rf"\Big[{kk}u'\,v\Big]_{{{ltx_num(prob.xa)}}}^{{{ltx_num(prob.xb)}}}")
    else:
        green = (rf"-\int_\Omega\nabla\cdot\left({kk}\nabla u\right)v\,d\mathbf x = \int_\Omega {kk}\nabla u\cdot\nabla v"
                 rf"\,d\mathbf x - \int_{{\partial\Omega}} {kk}\frac{{\partial u}}{{\partial n}}\,v\,ds")
    steps.append({"title": "3. Intégration par parties (formule de Green)", "latex": green,
                  "text": "La dérivée seconde de u est transférée : une dérivée sur u, une sur v. On ne demande "
                          "plus que u et v soient dérivables une fois (au sens faible) : c'est l'origine de l'espace "
                          "H¹. Le terme de bord fait apparaître le flux normal k ∂u/∂n."})
    # étape 4 : bord
    lines = []
    if d == 1:
        for side in prob.sides:
            typ = prob.bc[side]["type"]
            xv = ltx_num(prob.xa if side == "left" else prob.xb)
            nrm = "-1" if side == "left" else "+1"
            if typ == "dirichlet":
                lines.append(rf"x={xv}\ (n={nrm}):\ v({xv}) = 0\ \Rightarrow\ \text{{contribution nulle (condition essentielle)}}")
            elif typ == "neumann":
                lines.append(rf"x={xv}\ (n={nrm}):\ {kf}\partial_n u = g\ \Rightarrow\ +\,g({xv})\,v({xv})\ \text{{dans }} L(v)")
            else:
                lines.append(rf"x={xv}\ (n={nrm}):\ {kf}\partial_n u = g-\beta u\ \Rightarrow\ "
                             rf"+\beta u({xv})v({xv})\ \text{{dans }} a,\ +g({xv})v({xv})\ \text{{dans }} L")
    else:
        lines.append(r"\int_{\partial\Omega} " + kf + r"\frac{\partial u}{\partial n}\,v\,ds = "
                     + " + ".join(rf"\int_{{{GAMMA[s]}}}\cdots" for s in prob.sides))
        for side in prob.sides:
            typ = prob.bc[side]["type"]
            if typ == "dirichlet":
                lines.append(rf"{GAMMA[side]}:\ v=0\ \Rightarrow\ \int_{{{GAMMA[side]}}}{kf}\partial_n u\,v\,ds = 0"
                             r"\quad\text{(Dirichlet : essentielle)}")
            elif typ == "neumann":
                lines.append(rf"{GAMMA[side]}:\ {kf}\partial_n u = g\ \Rightarrow\ \int_{{{GAMMA[side]}}} g\,v\,ds"
                             r"\ \text{passe dans } L(v)\quad\text{(Neumann : naturelle)}")
            else:
                lines.append(rf"{GAMMA[side]}:\ {kf}\partial_n u = g-\beta u\ \Rightarrow\ "
                             rf"\int_{{{GAMMA[side]}}}\beta\,u\,v\,ds\ \text{{dans }} a,\ "
                             rf"\int_{{{GAMMA[side]}}} g\,v\,ds\ \text{{dans }} L\quad\text{{(Robin)}}")
    steps.append({"title": "4. Traitement des termes de bord selon les conditions aux limites",
                  "latex": r"\begin{aligned}" + r"\\ ".join(lines) + r"\end{aligned}",
                  "text": "Dirichlet est une condition essentielle : imposée dans l'espace (v = 0 sur Γ_D). "
                          "Neumann est naturelle : elle s'insère dans le second membre. Robin contribue à la fois "
                          "à la forme bilinéaire (terme βuv, symétrique et positif si β ≥ 0) et au second membre."})
    # étape 5 : formes a et L
    a_terms = []
    if d == 1:
        a_terms.append(rf"{integ}{kk}u'\,v'\,dx" if prob.time_order == 0 else rf"{integ}{kk}\partial_x u\,\partial_x v\,dx")
    else:
        a_terms.append(rf"\int_\Omega {kk}\nabla u\cdot\nabla v\,d\mathbf x")
    if has_b:
        a_terms.append(r"\int_\Omega(\mathbf b\cdot\nabla u)\,v\,d\mathbf x" if d == 2 else rf"{integ}b\,\partial_x u\,v\,dx")
    if has_c:
        a_terms.append(rf"{integ}c\,u\,v{dx}")
    for side in rset:
        if d == 1:
            xv = ltx_num(prob.xa if side == "left" else prob.xb)
            a_terms.append(rf"\beta\,u({xv})\,v({xv})")
        else:
            a_terms.append(rf"\int_{{{GAMMA[side]}}}\beta\,u\,v\,ds")
    l_terms = [rf"{integ}f\,v{dx}"] if "f" in prob.meta["uses"] else []
    for side in nset + rset:
        if d == 1:
            xv = ltx_num(prob.xa if side == "left" else prob.xb)
            l_terms.append(rf"g({xv})\,v({xv})")
        else:
            l_terms.append(rf"\int_{{{GAMMA[side]}}}g\,v\,ds")
    if not l_terms:
        l_terms = ["0"]
    a_lx = r"a(u,v) = " + " + ".join(a_terms)
    l_lx = r"L(v) = " + " + ".join(l_terms)
    out["a_latex"], out["L_latex"] = a_lx, l_lx
    spaces = function_spaces(prob)
    out["spaces"] = spaces
    if prob.time_order == 0:
        final = (r"\text{Trouver } u\in " + spaces["trial"].split("=")[0].strip() +
                 r"\ \text{tel que}\quad a(u,v) = L(v)\qquad\forall v\in " + spaces["test"].split("=")[0].strip())
    elif prob.time_order == 1:
        final = (r"\text{Pour presque tout } t:\ \ \frac{d}{dt}(u(t),v)_{L^2} + a(u(t),v) = L(t;v)\quad\forall v\in "
                 + spaces["test"].split("=")[0].strip() + r",\qquad u(0) = u_0")
    else:
        final = (r"\frac{d^2}{dt^2}(u(t),v)_{L^2} + a(u(t),v) = L(t;v)\quad\forall v\in "
                 + spaces["test"].split("=")[0].strip() + r",\quad u(0)=u_0,\ \partial_t u(0) = v_0")
    steps.append({"title": "5. Formulation variationnelle (faible)",
                  "latex": r"\begin{aligned}&" + a_lx + r"\\ &" + l_lx + r"\\[4pt] &" + final + r"\end{aligned}",
                  "text": "La formulation faible ne contient plus que des dérivées premières. Elle a un sens dès que "
                          "u, v ∈ H¹(Ω) et f ∈ L²(Ω) — hypothèses beaucoup plus faibles que u ∈ C²."})
    # données instanciées
    inst = []
    if prob.family not in ("poisson", "laplace", "heat", "wave") and not (prob.k_expr == 1):
        inst.append(rf"k(\mathbf x) = {_expr_lx(prob.k_expr)}")
    if prob.family == "heat":
        inst.append(rf"\kappa = {ltx_num(prob.kappa or 1.0)}")
    if prob.family == "wave":
        inst.append(rf"c^2 = {ltx_num((prob.cw or 1.0) ** 2)}")
    if has_c:
        inst.append(rf"c(\mathbf x) = {_expr_lx(prob.c_expr)}")
    if has_b:
        inst.append(rf"\mathbf b = ({ltx_num(prob.bx)},\ {ltx_num(prob.by)})" if d == 2 else rf"b = {ltx_num(prob.bx)}")
    if "f" in prob.meta["uses"]:
        inst.append(rf"f = {_expr_lx(prob.f_expr)}")
    for side in rset:
        inst.append(rf"\beta|_{{{GAMMA[side] if d == 2 else side}}} = {_expr_lx(prob.bc[side]['beta'])}")
    out["instantiated"] = inst
    # étape 6 : espaces
    steps.append({"title": "6. Choix des espaces fonctionnels",
                  "latex": r"\begin{aligned}&\text{Espace des tests : } " + spaces["test"] +
                           r"\\ &\text{Espace de recherche : } " + spaces["trial"] + r"\end{aligned}",
                  "text": spaces["comment"]})
    # étape 7 : relèvement / compatibilité
    if spaces["case"] == "affine":
        steps.append({"title": "7. Relèvement de la condition de Dirichlet non homogène",
                      "latex": r"u = R g_D + u_0,\quad u_0\in V_0:\qquad a(u_0,v) = L(v) - a(R g_D, v)\quad\forall v\in V_0",
                      "text": "R g_D ∈ H¹(Ω) est un relèvement (prolongement) de la donnée de bord, qui existe si "
                              "g_D ∈ H¹ᐟ²(Γ_D) par surjectivité de l'application trace. On se ramène ainsi à un "
                              "problème posé dans l'espace vectoriel V₀, où Lax–Milgram s'applique. En éléments "
                              "finis, R g_D est l'interpolé de g_D aux nœuds de Γ_D."})
    if spaces["case"] == "neumann_pur":
        steps.append({"title": "7. Condition de compatibilité (Neumann pur)",
                      "latex": r"v\equiv 1:\quad a(u,1) = 0 = L(1)\ \Longleftrightarrow\ \int_\Omega f\,d\mathbf x "
                               r"+ \int_{\partial\Omega} g\,ds = 0",
                      "text": "Le noyau de a contient les constantes : le problème n'a de solution que si la donnée "
                              "est orthogonale à ce noyau (alternative de Fredholm), et la solution est unique à une "
                              "constante près — on impose ∫u = 0."})
    steps.append({"title": "Réciproque (régularité)",
                  "latex": r"u\in H^2(\Omega)\ \text{solution faible}\ \Longrightarrow\ "
                           r"\text{EDP vérifiée p.p. et conditions naturelles au sens des traces}",
                  "text": "En réintégrant par parties avec des fonctions tests de C_c^∞(Ω), on retrouve l'EDP ; avec "
                          "des fonctions tests non nulles au bord, on retrouve les conditions de Neumann/Robin. "
                          "La solution faible généralise donc la solution forte."})
    out["steps"] = steps
    out["pedagogy"] = [
        ("Pourquoi multiplier par une fonction test ?",
         "Une fonction de L² n'a pas de valeur ponctuelle bien définie ; tester l'équation contre une famille riche "
         "de fonctions v (dense) caractérise entièrement la solution. C'est aussi le principe des travaux virtuels "
         "en mécanique : v joue le rôle d'un déplacement virtuel."),
        ("Pourquoi intégrer par parties ?",
         "Pour équilibrer la régularité exigée : au lieu de deux dérivées sur u, une sur u et une sur v. La forme "
         "a(u, v) obtenue est symétrique (si b = 0), ce qui donne un cadre hilbertien (énergie)."),
        ("Pourquoi la dérivée seconde disparaît ?",
         "La formule de Green transforme ∫(−∇·(k∇u))v en ∫k∇u·∇v plus un terme de bord : la dérivée seconde est "
         "« partagée ». Elle n'est plus exigée : u ∈ H¹ suffit, alors que u peut ne pas être dans H² (coins rentrants, "
         "coefficients discontinus)."),
        ("Pourquoi les dérivées faibles sont-elles nécessaires ?",
         "Les fonctions affines par morceaux (éléments finis P1) ou |x| ne sont pas dérivables partout. Leur dérivée "
         "faible, définie par ∫u ∂φ = −∫g φ pour toute φ ∈ C_c^∞, existe et est dans L² : elles appartiennent à H¹. "
         "Les espaces de Sobolev complètent C¹ pour la norme d'énergie et sont des espaces de Hilbert."),
        ("Pourquoi la formulation faible est-elle adaptée aux éléments finis ?",
         "Il suffit de remplacer V par un sous-espace de dimension finie V_h ⊂ V (méthode de Galerkin). Les "
         "fonctions P1 globalement continues sont dans H¹ mais pas dans C¹ : seule la formulation faible a un sens "
         "pour elles. On obtient directement un système linéaire symétrique KU = F, et le lemme de Céa garantit "
         "la quasi-optimalité de u_h."),
    ]
    return out


# ------------------------------------------------------------------ quadrature
def gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points et poids de Gauss–Legendre sur [0, 1] (poids de somme 1)."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    return 0.5 * (xg + 1.0), 0.5 * wg


def domain_quadrature(prob: Problem, panels: int = 24, order: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature composite de Gauss sur Ω (indépendante du maillage)."""
    s, w = gauss_legendre_01(order)

    def comp(a: float, b: float, npan: int) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(a, b, npan + 1)
        hpan = np.diff(edges)
        pts = (edges[:-1, None] + hpan[:, None] * s[None, :]).ravel()
        wts = (hpan[:, None] * w[None, :]).ravel()
        return pts, wts

    if prob.dim == 1:
        xq, wq = comp(prob.xa, prob.xb, panels * 4)
        return xq, np.zeros_like(xq), wq
    if prob.shape == "lshape":
        boxes = [(-1, 0, -1, 0), (-1, 0, 0, 1), (0, 1, 0, 1)]
    else:
        boxes = [(prob.xa, prob.xb, prob.ya, prob.yb)]
    xs_all, ys_all, ws_all = [], [], []
    for (x0, x1, y0, y1) in boxes:
        px, wx = comp(x0, x1, panels)
        py, wy = comp(y0, y1, panels)
        XX, YY = np.meshgrid(px, py)
        WW = np.outer(wy, wx)
        xs_all.append(XX.ravel())
        ys_all.append(YY.ravel())
        ws_all.append(WW.ravel())
    return np.concatenate(xs_all), np.concatenate(ys_all), np.concatenate(ws_all)


def boundary_quadrature(prob: Problem, side: str, panels: int = 24, order: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prob.dim == 1:
        return np.array([prob.xa if side == "left" else prob.xb]), np.array([0.0]), np.array([1.0])
    s, w = gauss_legendre_01(order)
    edges = np.linspace(0, 1, panels + 1)
    sp_ = (edges[:-1, None] + np.diff(edges)[:, None] * s[None, :]).ravel()
    wp = (np.diff(edges)[:, None] * w[None, :]).ravel()
    if side == "boundary":  # bord polygonal du domaine en L
        pts = [(-1, -1), (0, -1), (0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1)]
        xs_, ys_, ws_ = [], [], []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            L = math.hypot(x1 - x0, y1 - y0)
            xs_.append(x0 + sp_ * (x1 - x0))
            ys_.append(y0 + sp_ * (y1 - y0))
            ws_.append(wp * L)
        return np.concatenate(xs_), np.concatenate(ys_), np.concatenate(ws_)
    if side in ("left", "right"):
        L = prob.yb - prob.ya
        xv = prob.xa if side == "left" else prob.xb
        return np.full_like(sp_, xv), prob.ya + L * sp_, wp * L
    L = prob.xb - prob.xa
    yv = prob.ya if side == "bottom" else prob.yb
    return prob.xa + L * sp_, np.full_like(sp_, yv), wp * L


def l2_norm_of(prob: Problem, fn: NumFunc, t: float = 0.0) -> float:
    xq, yq, wq = domain_quadrature(prob)
    return float(np.sqrt(np.sum(wq * fn(xq, yq, t) ** 2)))


def neumann_compatibility(prob: Problem, t: float = 0.0) -> dict[str, float]:
    """∫_Ω f + ∫_∂Ω g (doit être nul pour un problème de Neumann pur)."""
    xq, yq, wq = domain_quadrature(prob)
    int_f = float(np.sum(wq * prob.f(xq, yq, t)))
    int_g = 0.0
    for side, b in prob.bcf.items():
        bx, by, bw = boundary_quadrature(prob, side)
        int_g += float(np.sum(bw * b["g"](bx, by, t)))
    scale = abs(int_f) + abs(int_g) + l2_norm_of(prob, prob.f) + 1e-300
    return {"int_f": int_f, "int_g": int_g, "total": int_f + int_g, "relative": abs(int_f + int_g) / scale}


# ------------------------------------------------------------- Lax–Milgram
def _interval_lambda1(left: str, right: str, L: float) -> float:
    """Première valeur propre de −d²/dx² sur (0, L) ; Robin traité comme Neumann (minorant)."""
    lt = "D" if left == "dirichlet" else "N"
    rt = "D" if right == "dirichlet" else "N"
    if lt == "D" and rt == "D":
        return (math.pi / L) ** 2
    if lt == "N" and rt == "N":
        return 0.0
    return (math.pi / (2 * L)) ** 2


def _interval_robin_lambda1(types: tuple[str, str], L: float, n: int = 600) -> float:
    """Approximation (par excès, Galerkin P1) de la 1re valeur propre de −d²/dx² avec Robin de coefficient 1."""
    h = L / n
    main = np.full(n + 1, 2.0 / h)
    main[0] = main[-1] = 1.0 / h
    off = np.full(n, -1.0 / h)
    K = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    mm = np.full(n + 1, 4.0 * h / 6)
    mm[0] = mm[-1] = 2.0 * h / 6
    M = np.diag(mm) + np.diag(np.full(n, h / 6), 1) + np.diag(np.full(n, h / 6), -1)
    keep = np.ones(n + 1, bool)
    for end, typ in zip((0, n), types):
        if typ == "robin":
            K[end, end] += 1.0
        elif typ == "dirichlet":
            keep[end] = False
    K, M = K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]
    return float(sla.eigh(K, M, eigvals_only=True, subset_by_index=[0, 0])[0])


def poincare_constant(prob: Problem) -> dict[str, Any]:
    """Minorant rigoureux λ₁ de la constante de Poincaré–Friedrichs : ‖∇v‖² ≥ λ₁‖v‖² sur V₀."""
    types = {s: b["type"] for s, b in prob.bc.items()}
    if prob.dim == 1:
        L = prob.xb - prob.xa
        lam = _interval_lambda1(types["left"], types["right"], L)
        return {"lambda1": lam, "exact": True,
                "method": "valeur propre exacte de −d²/dx² (Robin minoré par Neumann)",
                "formula": rf"\lambda_1 = {ltx_num(lam)}"}
    if prob.shape == "lshape":
        lam = math.pi ** 2 / 2
        return {"lambda1": lam, "exact": True,
                "method": "minorant par monotonie des valeurs propres de Dirichlet : Ω ⊂ (−1,1)² ⇒ "
                          "λ₁(Ω) ≥ λ₁((−1,1)²) = π²/2 (valeur numérique connue : λ₁(L) ≈ 9.6397)",
                "formula": r"\lambda_1(\Omega)\ \ge\ \pi^2/2"}
    Lx, Ly = prob.xb - prob.xa, prob.yb - prob.ya
    lam = _interval_lambda1(types["left"], types["right"], Lx) + _interval_lambda1(types["bottom"], types["top"], Ly)
    return {"lambda1": lam, "exact": True,
            "method": "séparation des variables sur le rectangle (Robin minoré par Neumann) : λ₁ = μₓ + μ_y",
            "formula": rf"\lambda_1 = \mu_x + \mu_y = {ltx_num(lam)}"}


def friedrichs_robin_constant(prob: Problem) -> float:
    """λ_F ≈ min (‖∇v‖² + ∫_{Γ_R} v²)/‖v‖² (Robin de coefficient 1), par séparation des variables."""
    types = {s: b["type"] for s, b in prob.bc.items()}
    if prob.dim == 1:
        return _interval_robin_lambda1((types["left"], types["right"]), prob.xb - prob.xa)
    lx = _interval_robin_lambda1((types["left"], types["right"]), prob.xb - prob.xa)
    ly = _interval_robin_lambda1((types["bottom"], types["top"]), prob.yb - prob.ya)
    return lx + ly


def lax_milgram_analysis(prob: Problem) -> dict[str, Any]:
    """Analyse d'existence et d'unicité avec distinction stricte des statuts."""
    out: dict[str, Any] = {"items": [], "constants": {}}
    if prob.cls != "elliptic":
        out["applicable"] = False
        out["status"] = "not_treated"
        if prob.cls == "parabolic":
            out["framework"] = "parabolic"
        elif prob.cls == "hyperbolic":
            out["framework"] = "wave"
        else:
            out["framework"] = "transport"
        out.update(evolution_wellposedness(prob))
        return out
    out["framework"] = "elliptic"
    sp_ = function_spaces(prob)
    kmin, kmax, kex = prob.coefficient_range("k")
    cmin, cmax, cex = prob.coefficient_range("c")
    out["constants"].update(k_min=kmin, k_max=kmax, c_min=cmin, c_max=cmax)
    items = out["items"]
    # 1. Hilbert
    items.append({"title": "1. Espace de Hilbert", "latex": sp_["test"], "status": "ok_exact",
                  "detail": "Sous-espace fermé de H¹(Ω) (noyau d'une application linéaire continue : trace sur Γ_D "
                            "ou moyenne), donc complet pour la norme ‖v‖²_{H¹} = ‖v‖² + ‖∇v‖²."})
    wf = weak_formulation(prob)
    items.append({"title": "2. Forme bilinéaire", "latex": wf["a_latex"], "status": "ok_exact",
                  "detail": "Bilinéaire et symétrique (pas de terme d'advection) : a(u, v) = a(v, u)."})
    lift = sp_["case"] == "affine"
    items.append({"title": "3. Forme linéaire", "latex": wf["L_latex"] + (r"\ -\ a(Rg_D, v)" if lift else ""),
                  "status": "ok_exact",
                  "detail": "Linéaire en v" + (" ; le relèvement R g_D de la donnée de Dirichlet ajoute le terme "
                                                 "−a(R g_D, v)." if lift else ".")})
    # 4. continuité de a
    beta_max = 0.0
    beta_min = math.inf
    for side, b in prob.bcf.items():
        if b["type"] == "robin":
            bxs, bys = prob.boundary_samples(side)
            vals = b["beta"](bxs, bys)
            beta_max = max(beta_max, float(np.max(vals)))
            beta_min = min(beta_min, float(np.min(vals)))
    M = max(kmax, abs(cmin), abs(cmax))
    st_k = "ok_exact" if (kex and cex) else "ok_numeric"
    m_lx = rf"|a(u,v)|\le M\,\|u\|_{{H^1}}\|v\|_{{H^1}},\qquad M = \max(k_{{\max}}, \|c\|_\infty) = {ltx_num(M)}"
    if beta_max > 0:
        m_lx += rf" + \beta_{{\max}}C_{{tr}}^2,\ \beta_{{\max}} = {ltx_num(beta_max)}"
    items.append({"title": "4. Continuité de a", "latex": m_lx, "status": st_k,
                  "detail": ("Cauchy–Schwarz terme à terme. " + ("Coefficients constants : bornes exactes." if
                             st_k == "ok_exact" else "Bornes k_max, ‖c‖∞ obtenues par échantillonnage fin "
                             "(121² points) : vérification numérique, pas une preuve formelle.") +
                             (" Le terme de Robin est borné grâce au théorème de trace (C_tr constante de trace)."
                              if beta_max > 0 else ""))})
    out["constants"]["M"] = M
    # 5. coercivité
    coer: dict[str, Any] = {"title": "5. Coercivité"}
    alpha = None
    pc = poincare_constant(prob)
    out["constants"]["lambda1"] = pc["lambda1"]
    if prob.has_robin and beta_min < 0:
        coer.update(status="not_verified",
                    latex=r"\beta < 0\ \text{sur une partie de } \Gamma_R",
                    detail="Un coefficient de Robin négatif détruit la positivité du terme de bord : la coercivité "
                           "n'est pas établie par cette analyse.")
    elif prob.is_pure_neumann():
        lam_n = (math.pi / (prob.xb - prob.xa)) ** 2 if prob.dim == 1 else \
            min((math.pi / (prob.xb - prob.xa)) ** 2, (math.pi / (prob.yb - prob.ya)) ** 2)
        alpha = kmin * lam_n / (1 + lam_n)
        coer.update(status="ok_exact" if kex else "ok_numeric",
                    latex=(r"a(1,1) = 0\ \Rightarrow\ \text{non coercive sur } H^1;\quad\text{sur } V=\{\textstyle\int v=0\}:\ "
                           rf"\|v\|^2\le \tfrac1{{\mu_1}}\|\nabla v\|^2,\ \mu_1 = {ltx_num(lam_n)}\ \Rightarrow\ "
                           rf"\alpha = \frac{{k_{{\min}}\mu_1}}{{1+\mu_1}} = {ltx_num(alpha)}"),
                    detail="Inégalité de Poincaré–Wirtinger (μ₁ = première valeur propre non nulle du Laplacien de "
                           "Neumann, exacte sur l'intervalle/rectangle). La coercivité n'a lieu que sur l'espace "
                           "des fonctions de moyenne nulle.")
    elif prob.has_dirichlet:
        lam = pc["lambda1"]
        val = (kmin * lam + cmin) / (1 + lam)
        alpha_c = min(kmin, val)
        if alpha_c > 0:
            alpha = alpha_c
            coer.update(status="ok_exact" if (kex and cex) else "ok_numeric",
                        latex=(r"\begin{aligned}&a(v,v)\ \ge\ k_{\min}\|\nabla v\|^2 + c_{\min}\|v\|^2,\quad \|\nabla v\|^2\ge\lambda_1\|v\|^2\\"
                               rf"&\Rightarrow\ \alpha = \min\!\Big(k_{{\min}},\ \frac{{k_{{\min}}\lambda_1 + c_{{\min}}}}{{1+\lambda_1}}\Big) "
                               rf"= {ltx_num(alpha)}\end{{aligned}}"),
                        detail=f"Inégalité de Poincaré sur V₀ (Γ_D de mesure non nulle) : λ₁ = {fmt(lam)} — "
                               f"{pc['method']}. k_min = {fmt(kmin)}, c_min = {fmt(cmin)}. Les termes de Robin (β ≥ 0) "
                               "sont minorés par 0." + ("" if (kex and cex) else " Bornes des coefficients obtenues "
                                                        "par échantillonnage : vérification numérique."))
        else:
            coer.update(status="not_verified",
                        latex=rf"c_{{\min}} = {ltx_num(cmin)}\ \le\ -k_{{\min}}\lambda_1 = {ltx_num(-kmin * lam)}",
                        detail="La réaction est trop négative : la minoration par Poincaré ne prouve plus la "
                               "coercivité. Le problème peut rester bien posé (alternative de Fredholm, si 0 n'est "
                               "pas valeur propre), mais Lax–Milgram ne permet pas de conclure.")
    elif cmin > 0:
        alpha = min(kmin, cmin)
        coer.update(status="ok_exact" if (kex and cex) else "ok_numeric",
                    latex=rf"a(v,v)\ \ge\ \min(k_{{\min}}, c_{{\min}})\,\|v\|_{{H^1}}^2,\quad \alpha = {ltx_num(alpha)}",
                    detail="La réaction strictement positive contrôle directement la norme L² : aucune inégalité de "
                           "Poincaré n'est nécessaire.")
    elif prob.has_robin and beta_min > 0 and cmin >= 0:
        lamF = friedrichs_robin_constant(prob)
        m = min(kmin, beta_min)
        alpha = min(m, (m * lamF + cmin) / (1 + lamF))
        coer.update(status="ok_numeric",
                    latex=(r"\|v\|^2\le C_F\Big(\|\nabla v\|^2 + \int_{\Gamma_R}v^2\Big)\ \Rightarrow\ "
                           rf"\alpha \approx {ltx_num(alpha)}\quad(\lambda_F = 1/C_F \approx {ltx_num(lamF)})"),
                    detail="Coercivité démontrée par l'inégalité de Friedrichs généralisée (Robin de coefficient "
                           "β ≥ β₀ > 0 sur Γ_R de mesure non nulle) : le résultat qualitatif est rigoureux ; la "
                           "valeur de la constante est une approximation numérique (Galerkin 1D par séparation "
                           "des variables, approximation par excès de λ_F).")
        out["constants"]["lambdaF"] = lamF
    else:
        coer.update(status="not_verified", latex=r"\text{aucun terme ne contrôle } \|v\|_{L^2}",
                    detail="Sans Dirichlet, sans Robin positif et avec c ≥ 0 non strictement positif, la coercivité "
                           "sur H¹ n'est pas établie.")
    items.append(coer)
    out["constants"]["alpha"] = alpha
    # 6. continuité de L
    fnorm = l2_norm_of(prob, prob.f) if "f" in prob.meta["uses"] else 0.0
    lstat = "ok_numeric" if math.isfinite(fnorm) else "not_verified"
    gn = []
    for side in _neumann_sides(prob) + _robin_sides(prob):
        bx, by, bw = boundary_quadrature(prob, side)
        gn.append(float(np.sqrt(np.sum(bw * prob.bcf[side]["g"](bx, by) ** 2))))
    gnorm = float(np.sqrt(np.sum(np.square(gn)))) if gn else 0.0
    l_lx = rf"|L(v)|\le\Big(\|f\|_{{L^2}} + C_{{tr}}\|g\|_{{L^2(\Gamma_N\cup\Gamma_R)}}\Big)\|v\|_{{H^1}},\quad \|f\|_{{L^2}} = {ltx_num(fnorm)}"
    if gn:
        l_lx += rf",\ \|g\| = {ltx_num(gnorm)}"
    items.append({"title": "6. Continuité de L", "latex": l_lx, "status": lstat,
                  "detail": "Normes calculées par quadrature de Gauss composite (indépendante du maillage). " +
                            ("Le relèvement ajoute la contribution M‖R g_D‖_H¹, finie si g_D ∈ H¹ᐟ²(Γ_D) "
                             "(donnée régulière ici)." if lift else "")})
    out["constants"]["f_l2"] = fnorm
    # compatibilité Neumann
    if prob.is_pure_neumann():
        comp = neumann_compatibility(prob)
        out["compatibility"] = comp
        ok_c = comp["relative"] < 1e-8
        items.append({"title": "7. Condition de compatibilité",
                      "latex": rf"\int_\Omega f + \int_{{\partial\Omega}} g = {ltx_num(comp['total'])}",
                      "status": "ok_numeric" if ok_c else "not_verified",
                      "detail": ("Condition vérifiée (à la précision de la quadrature) : L s'annule sur les "
                                 "constantes, le problème quotient est bien posé." if ok_c else
                                 "Condition violée : le problème continu n'admet AUCUNE solution. Le solveur "
                                 "numérique calcule la solution du problème projeté (données moins leur moyenne).")})
    statuses = [it["status"] for it in items]
    all_ok = all(s in ("ok_exact", "ok_numeric") for s in statuses)
    out["applicable"] = all_ok
    out["status"] = ("applicable" if all_ok else "not_verified")
    if all_ok and alpha:
        out["conclusion"] = (r"\begin{aligned}&\exists!\,u\in V\ :\ a(u,v) = L(v)\quad\forall v\in V,\\ "
                             r"&\|u\|_{H^1}\le\frac{\|L\|_{V'}}{\alpha}\end{aligned}")
        out["cea"] = (r"\begin{aligned}&\|u-u_h\|_{H^1}\ \le\ \frac{M}{\alpha}\ \inf_{v_h\in V_h}\|u-v_h\|_{H^1},\\ "
                      rf"&\frac{{M}}{{\alpha}} = {ltx_num(M / alpha)}\end{{aligned}}")
        out["cea_sym"] = (r"\begin{aligned}&\text{a symétrique : } \|u-u_h\|_a = \min_{v_h}\|u-v_h\|_a,\\ "
                          rf"&\text{{constante }}\sqrt{{M/\alpha}} = {ltx_num(math.sqrt(M / alpha))}\end{{aligned}}")
        out["energy"] = (r"\begin{aligned}&u = \arg\min_{v\in V} J(v),\quad J(v) = \tfrac12 a(v,v) - L(v)\\ "
                         r"&\text{(principe de Dirichlet / méthode de Ritz)}\end{aligned}")
    return out


def evolution_wellposedness(prob: Problem) -> dict[str, Any]:
    """Outils de bien-posé adaptés aux problèmes d'évolution (pas de Lax–Milgram direct)."""
    res: dict[str, Any] = {}
    kmin, kmax, _ = prob.coefficient_range("k")
    if prob.cls == "parabolic":
        cmin = prob.coefficient_range("c")[0]
        bnorm = math.hypot(prob.bx, prob.by)
        lam_g = max(0.0, bnorm ** 2 / (2 * kmin) - cmin) if bnorm > 0 else max(0.0, -cmin)
        pc = poincare_constant(prob)
        dirichlet_all = set(_dirichlet_sides(prob)) == set(prob.sides)
        res["theorem"] = "Théorème de J.-L. Lions (méthode de Faedo–Galerkin) / théorie des semi-groupes"
        res["items"] = [
            {"title": "Triplet de Gelfand", "latex": r"V\subset H = L^2(\Omega)\subset V',\quad V = H^1_{\Gamma_D}(\Omega)",
             "status": "ok_exact", "detail": "Injections continues et denses."},
            {"title": "Inégalité de Gårding",
             "latex": rf"a(v,v) + \lambda\|v\|^2\ \ge\ \tfrac{{k_{{\min}}}}2\|v\|_{{H^1}}^2,\qquad \lambda = {ltx_num(lam_g + kmin / 2)}",
             "status": "ok_exact" if prob.k.is_constant else "ok_numeric",
             "detail": "Le terme d'advection est contrôlé par Young : |∫(b·∇v)v| ≤ (k_min/2)‖∇v‖² + |b|²/(2k_min)‖v‖². "
                       "Gårding suffit (la coercivité stricte n'est pas nécessaire pour un problème d'évolution)."},
            {"title": "Données", "latex": r"u_0\in L^2(\Omega),\quad f\in L^2(0,T;V')", "status": "ok_numeric",
             "detail": "Données bornées sur Ω̄ (vérifiées par échantillonnage)."},
        ]
        res["conclusion"] = (r"\exists!\,u\in L^2(0,T;V)\cap C([0,T];L^2(\Omega)),\quad \partial_t u\in L^2(0,T;V')")
        if dirichlet_all and all(prob.bc[s]["g"] == 0 for s in prob.sides) and cmin >= 0:
            rate = kmin * pc["lambda1"] + cmin
            res["energy"] = (r"\tfrac12\frac{d}{dt}\|u\|^2 + a(u,u) = (f,u)\ \Rightarrow\ "
                             rf"\|u(t)\|_{{L^2}}\le e^{{-{ltx_num(rate)}\,t}}\|u_0\|_{{L^2}}\quad (f=0)")
            res["decay_rate"] = rate
            res["energy_detail"] = ("Avec Dirichlet homogène, b constant (∫(b·∇v)v = 0 pour v ∈ H₀¹) et c ≥ 0, "
                                    f"la norme L² décroît au moins comme e^(−(k_min λ₁ + c_min) t), λ₁ = {fmt(pc['lambda1'])}.")
        else:
            res["energy"] = (r"\|u(t)\|^2 \le e^{2\lambda t}\Big(\|u_0\|^2 + \int_0^t \|f\|_{V'}^2\Big)\quad"
                             r"\text{(estimation d'énergie avec la constante de Gårding)}")
    elif prob.cls == "hyperbolic":
        res["theorem"] = "Méthode de Galerkin (Lions–Magenes) / semi-groupe unitaire sur V × L²"
        res["items"] = [
            {"title": "Données", "latex": r"u_0\in V,\quad v_0\in L^2(\Omega),\quad f\in L^2(0,T;L^2(\Omega))",
             "status": "ok_numeric", "detail": "Données régulières sur Ω̄ (vérification par échantillonnage)."},
            {"title": "Forme d'énergie", "latex": r"a(u,v) = c^2\int_\Omega\nabla u\cdot\nabla v",
             "status": "ok_exact", "detail": "Symétrique, positive ; coercive sur V₀ si Γ_D ≠ ∅ (Poincaré)."},
        ]
        res["conclusion"] = r"\exists!\,u\in C([0,T];V)\cap C^1([0,T];L^2(\Omega))"
        res["energy"] = (r"E(t) = \tfrac12\|\partial_t u\|^2 + \tfrac{c^2}2\|\nabla u\|^2,\qquad "
                         r"\frac{dE}{dt} = (f,\partial_t u)\ \Rightarrow\ E(t) = E(0)\ \text{si } f=0")
        res["energy_detail"] = ("Conservation de l'énergie (conditions homogènes de Dirichlet ou Neumann) : pas de "
                                "dissipation. Un schéma numérique pertinent doit conserver une énergie discrète.")
    else:
        res["theorem"] = "Méthode des caractéristiques + estimation d'énergie L²"
        res["items"] = [
            {"title": "Caractéristiques", "latex": r"\frac{dx}{dt} = a\ \Rightarrow\ u(x,t) = u_0(x - a t)",
             "status": "ok_exact", "detail": "Solution explicite : existence et unicité immédiates (u₀ ∈ C¹)."},
            {"title": "Stabilité L²", "latex": r"\frac{d}{dt}\|u\|_{L^2}^2 = -a\big[u^2\big]_{\partial\Omega} = 0\ \text{(périodique)}",
             "status": "ok_exact", "detail": "La norme L² est conservée : problème bien posé au sens de Hadamard."},
        ]
        res["conclusion"] = r"\exists!\,u(x,t) = u_0(x - a t)\ \text{(prolongée périodiquement ou par la donnée entrante)}"
        res["energy"] = r"\|u(t)\|_{L^2} = \|u_0\|_{L^2}"
    res["why_not_lm"] = ("Lax–Milgram est un théorème pour les problèmes stationnaires coercifs a(u,v) = L(v). "
                         "Pour un problème d'évolution, on utilise la théorie de Lions (paraboliques), "
                         "l'énergie (ondes) ou les caractéristiques (transport). En revanche, Lax–Milgram "
                         "s'applique À CHAQUE PAS DE TEMPS d'un schéma implicite : la forme (u,v)/Δt + ϑ a(u,v) est "
                         "coercive dès que Δt est assez petit (ou toujours si a est coercive).")
    return res


# =============================================================================
# 6. QUADRATURES, MAILLAGES, FONCTIONS DE FORME P1
# =============================================================================
def triangle_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Règles de quadrature sur le triangle de référence : coordonnées barycentriques (nq, 3) et
    poids normalisés (somme 1) ; ∫_K g ≈ |K| Σ w_q g(x_q)."""
    if degree <= 1:
        return np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])
    if degree == 2:
        b = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        return b, np.full(3, 1 / 3)
    if degree <= 4:  # Dunavant, 6 points, degré 4
        a1, w1 = 0.445948490915965, 0.223381589678011
        a2, w2 = 0.091576213509771, 0.109951743655322
        pts = []
        for a in (a1, a2):
            pts += [[1 - 2 * a, a, a], [a, 1 - 2 * a, a], [a, a, 1 - 2 * a]]
        return np.array(pts), np.array([w1] * 3 + [w2] * 3)
    # Radon, 7 points, degré 5
    s15 = math.sqrt(15.0)
    a = (6 - s15) / 21
    b = (6 + s15) / 21
    wa = (155 - s15) / 1200
    wb = (155 + s15) / 1200
    pts = [[1 / 3, 1 / 3, 1 / 3], [1 - 2 * a, a, a], [a, 1 - 2 * a, a], [a, a, 1 - 2 * a],
           [1 - 2 * b, b, b], [b, 1 - 2 * b, b], [b, b, 1 - 2 * b]]
    return np.array(pts), np.array([9 / 40, wa, wa, wa, wb, wb, wb])


def segment_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre sur [0,1] en coordonnées barycentriques (λ₁ = 1−s, λ₂ = s)."""
    s, w = gauss_legendre_01(n)
    return np.stack([1 - s, s], axis=1), w


class Mesh:
    """Maillage conforme simplicial (segments en 1D, triangles en 2D)."""

    def __init__(self, nodes: np.ndarray, elems: np.ndarray, bfaces: np.ndarray, bside: np.ndarray,
                 side_names: list[str], meta: dict | None = None):
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim == 1:
            self.nodes = self.nodes[:, None]
        self.elems = np.asarray(elems, dtype=np.int64)
        self.bfaces = np.asarray(bfaces, dtype=np.int64)
        self.bside = np.asarray(bside, dtype=np.int64)
        self.side_names = list(side_names)
        self.dim = self.nodes.shape[1]
        self.meta = dict(meta or {})

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elems(self) -> int:
        return self.elems.shape[0]

    def side_nodes(self, side: str) -> np.ndarray:
        if side not in self.side_names:
            return np.zeros(0, dtype=np.int64)
        sid = self.side_names.index(side)
        return np.unique(self.bfaces[self.bside == sid].ravel())

    def boundary_nodes(self) -> np.ndarray:
        return np.unique(self.bfaces.ravel())

    def edges(self) -> np.ndarray:
        if self.dim == 1:
            return self.elems.copy()
        e = np.vstack([self.elems[:, [0, 1]], self.elems[:, [1, 2]], self.elems[:, [2, 0]]])
        return np.unique(np.sort(e, axis=1), axis=0)

    def quality(self) -> dict[str, float]:
        if self.dim == 1:
            h = np.diff(np.sort(self.nodes[:, 0]))
            return {"h_max": float(h.max()), "h_min": float(h.min()), "ratio": float(h.max() / h.min())}
        P = self.nodes[self.elems]
        a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
        b = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
        c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
        with np.errstate(all="ignore"):
            A = np.degrees(np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)))
            B = np.degrees(np.arccos(np.clip((a ** 2 + c ** 2 - b ** 2) / (2 * a * c), -1, 1)))
        C = 180.0 - A - B
        ang = np.stack([A, B, C], axis=1)
        diam = np.maximum(np.maximum(a, b), c)
        area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                            (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        rho = 2 * area / (a + b + c)  # rayon inscrit
        return {"h_max": float(diam.max()), "h_min": float(np.minimum(np.minimum(a, b), c).min()),
                "angle_min": float(ang.min()), "angle_max": float(ang.max()),
                "shape_reg": float((diam / (2 * rho)).max()), "area_min": float(area.min()),
                "nonobtuse": bool(ang.max() <= 90.0 + 1e-9)}


def boundary_edges_from_elems(elems: np.ndarray) -> np.ndarray:
    e = np.vstack([elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]])
    es = np.sort(e, axis=1)
    _, idx, cnt = np.unique(es, axis=0, return_index=True, return_counts=True)
    return e[idx[cnt == 1]]


def mesh_1d(xa: float, xb: float, n: int, grading: str = "uniform", perturb: float = 0.25, seed: int = 7) -> Mesh:
    i = np.arange(n + 1)
    if grading == "chebyshev":
        x = xa + (xb - xa) * 0.5 * (1 - np.cos(np.pi * i / n))
    else:
        x = np.linspace(xa, xb, n + 1)
        if grading == "perturbed" and n > 1:
            rng = np.random.default_rng(seed)
            h = (xb - xa) / n
            x[1:-1] += perturb * h * (rng.random(n - 1) - 0.5)
    elems = np.stack([i[:-1], i[1:]], axis=1)
    return Mesh(x, elems, np.array([[0], [n]]), np.array([0, 1]), ["left", "right"],
                {"kind": "segments", "n": n, "grading": grading})


def mesh_rectangle(xa: float, xb: float, ya: float, yb: float, nx: int, ny: int, pattern: str = "right",
                   grading: str = "uniform", perturb: float = 0.25, seed: int = 7) -> Mesh:
    if grading == "chebyshev":
        xs = xa + (xb - xa) * 0.5 * (1 - np.cos(np.pi * np.arange(nx + 1) / nx))
        ys = ya + (yb - ya) * 0.5 * (1 - np.cos(np.pi * np.arange(ny + 1) / ny))
    else:
        xs, ys = np.linspace(xa, xb, nx + 1), np.linspace(ya, yb, ny + 1)
    XX, YY = np.meshgrid(xs, ys)
    nodes = np.stack([XX.ravel(), YY.ravel()], axis=1)
    I, J = np.meshgrid(np.arange(nx), np.arange(ny))
    I, J = I.ravel(), J.ravel()
    p00 = I + J * (nx + 1)
    p10, p01, p11 = p00 + 1, p00 + nx + 1, p00 + nx + 2
    if pattern == "crossed":
        centers = 0.25 * (nodes[p00] + nodes[p10] + nodes[p01] + nodes[p11])
        c = nodes.shape[0] + np.arange(len(p00))
        nodes = np.vstack([nodes, centers])
        elems = np.vstack([np.stack([p00, p10, c], 1), np.stack([p10, p11, c], 1),
                           np.stack([p11, p01, c], 1), np.stack([p01, p00, c], 1)])
    else:
        if pattern == "left":
            use_right = np.zeros(len(p00), bool)
        elif pattern == "alternate":
            use_right = (I + J) % 2 == 0
        else:
            use_right = np.ones(len(p00), bool)
        t1 = np.where(use_right[:, None], np.stack([p00, p10, p11], 1), np.stack([p00, p10, p01], 1))
        t2 = np.where(use_right[:, None], np.stack([p00, p11, p01], 1), np.stack([p10, p11, p01], 1))
        elems = np.vstack([t1, t2])
    # bords
    ii = np.arange(nx)
    jj = np.arange(ny)
    bottom = np.stack([ii, ii + 1], 1)
    top = np.stack([ii + ny * (nx + 1), ii + 1 + ny * (nx + 1)], 1)
    left = np.stack([jj * (nx + 1), (jj + 1) * (nx + 1)], 1)
    right = np.stack([nx + jj * (nx + 1), nx + (jj + 1) * (nx + 1)], 1)
    bfaces = np.vstack([left, right, bottom, top])
    bside = np.concatenate([np.full(ny, 0), np.full(ny, 1), np.full(nx, 2), np.full(nx, 3)])
    mesh = Mesh(nodes, elems, bfaces, bside, ["left", "right", "bottom", "top"],
                {"kind": "triangles", "nx": nx, "ny": ny, "pattern": pattern, "grading": grading})
    if grading == "perturbed":
        _perturb_interior(mesh, perturb, seed, (xb - xa) / nx, (yb - ya) / ny)
    return mesh


def _perturb_interior(mesh: Mesh, perturb: float, seed: int, hx: float, hy: float) -> None:
    rng = np.random.default_rng(seed)
    bnd = np.zeros(mesh.n_nodes, bool)
    bnd[mesh.boundary_nodes()] = True
    base = mesh.nodes.copy()
    disp = (rng.random(mesh.nodes.shape) - 0.5) * np.array([hx, hy])
    amp = perturb
    for _ in range(6):
        trial = base + amp * disp * (~bnd)[:, None]
        P = trial[mesh.elems]
        det = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        if det.min() > 1e-12:
            mesh.nodes = trial
            mesh.meta["perturb_effective"] = amp
            return
        amp *= 0.5
    mesh.meta["perturb_effective"] = 0.0


def mesh_lshape(n: int, pattern: str = "right") -> Mesh:
    """Domaine en L : (−1,1)² privé du quadrant x > 0, y < 0 ; n mailles par direction (pair)."""
    sq = mesh_rectangle(-1.0, 1.0, -1.0, 1.0, n, n, pattern)
    cen = sq.nodes[sq.elems].mean(axis=1)
    keep = ~((cen[:, 0] > 1e-12) & (cen[:, 1] < -1e-12))
    elems = sq.elems[keep]
    used = np.unique(elems)
    remap = -np.ones(sq.n_nodes, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes = sq.nodes[used]
    elems = remap[elems]
    bed = boundary_edges_from_elems(elems)
    return Mesh(nodes, elems, bed, np.zeros(len(bed), dtype=np.int64), ["boundary"],
                {"kind": "triangles", "nx": n, "ny": n, "pattern": pattern, "lshape": True})


def refine_red(mesh: Mesh) -> Mesh:
    """Raffinement uniforme : chaque segment en 2, chaque triangle en 4 (maillages emboîtés)."""
    if mesh.dim == 1:
        x = mesh.nodes[:, 0]
        order = np.argsort(x)
        xs = x[order]
        new = np.empty(2 * len(xs) - 1)
        new[0::2] = xs
        new[1::2] = 0.5 * (xs[:-1] + xs[1:])
        n = len(new) - 1
        i = np.arange(n + 1)
        return Mesh(new, np.stack([i[:-1], i[1:]], 1), np.array([[0], [n]]), np.array([0, 1]),
                    mesh.side_names, dict(mesh.meta, n=n))
    el = mesh.elems
    e_all = np.vstack([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]])
    es = np.sort(e_all, axis=1)
    uniq, inv = np.unique(es, axis=0, return_inverse=True)
    inv = inv.ravel()
    ne = el.shape[0]
    mid_index = mesh.n_nodes + np.arange(len(uniq))
    mids = 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]])
    nodes = np.vstack([mesh.nodes, mids])
    m01 = mid_index[inv[:ne]]
    m12 = mid_index[inv[ne:2 * ne]]
    m20 = mid_index[inv[2 * ne:]]
    v0, v1, v2 = el[:, 0], el[:, 1], el[:, 2]
    elems = np.vstack([np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1),
                       np.stack([m20, m12, v2], 1), np.stack([m01, m12, m20], 1)])
    # bords : chaque arête de bord est coupée en deux
    bs = np.sort(mesh.bfaces, axis=1)
    lookup = {tuple(r): k for k, r in enumerate(uniq)}
    bf_new, bside_new = [], []
    for (a, b), s in zip(mesh.bfaces, mesh.bside):
        mloc = mid_index[lookup[(min(a, b), max(a, b))]]
        bf_new += [[a, mloc], [mloc, b]]
        bside_new += [s, s]
    _ = bs
    meta = dict(mesh.meta)
    meta["refinements"] = meta.get("refinements", 0) + 1
    return Mesh(nodes, elems, np.array(bf_new), np.array(bside_new), mesh.side_names, meta)


def build_mesh(prob: Problem, cfg_mesh: dict, extra_refine: int = 0) -> Mesh:
    if prob.dim == 1:
        mesh = mesh_1d(prob.xa, prob.xb, int(cfg_mesh["nx"]), cfg_mesh.get("grading", "uniform"),
                       cfg_mesh.get("perturb", 0.25), cfg_mesh.get("seed", 7))
    elif prob.shape == "lshape":
        mesh = mesh_lshape(int(cfg_mesh["nx"]), cfg_mesh.get("pattern", "right"))
    else:
        mesh = mesh_rectangle(prob.xa, prob.xb, prob.ya, prob.yb, int(cfg_mesh["nx"]), int(cfg_mesh["ny"]),
                              cfg_mesh.get("pattern", "right"), cfg_mesh.get("grading", "uniform"),
                              cfg_mesh.get("perturb", 0.25), cfg_mesh.get("seed", 7))
    for _ in range(int(cfg_mesh.get("refine", 0)) + extra_refine):
        mesh = refine_red(mesh)
    if mesh.dim == 2 and mesh.n_elems > LIMITS["fem2d_elems_max"]:
        raise ConfigError(f"Maillage trop grand ({mesh.n_elems} triangles > {int(LIMITS['fem2d_elems_max'])}). "
                          "Réduisez le nombre de mailles ou de raffinements.")
    return mesh


def p1_geometry(mesh: Mesh) -> dict[str, np.ndarray]:
    """Mesures et gradients (constants) des fonctions de base barycentriques sur chaque élément."""
    P = mesh.nodes[mesh.elems]
    if mesh.dim == 1:
        h = P[:, 1, 0] - P[:, 0, 0]
        G = np.stack([-1.0 / h, 1.0 / h], axis=1)[:, :, None]
        return {"meas": np.abs(h), "grad": G, "P": P}
    x, y = P[:, :, 0], P[:, :, 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (y[:, 1] - y[:, 0]) * (x[:, 2] - x[:, 0])
    G = np.empty((len(det), 3, 2))
    G[:, 0, 0] = (y[:, 1] - y[:, 2]) / det
    G[:, 0, 1] = (x[:, 2] - x[:, 1]) / det
    G[:, 1, 0] = (y[:, 2] - y[:, 0]) / det
    G[:, 1, 1] = (x[:, 0] - x[:, 2]) / det
    G[:, 2, 0] = (y[:, 0] - y[:, 1]) / det
    G[:, 2, 1] = (x[:, 1] - x[:, 0]) / det
    return {"meas": 0.5 * np.abs(det), "grad": G, "P": P, "det": det}


def quad_points(mesh: Mesh, geo: dict, bary: np.ndarray) -> np.ndarray:
    """Coordonnées physiques des points de quadrature : (E, nq, d)."""
    return np.einsum("qi,eid->eqd", bary, geo["P"])


def _scatter(elems: np.ndarray, Ae: np.ndarray, n: int) -> sp.csr_matrix:
    m = elems.shape[1]
    rows = np.repeat(elems, m, axis=1).ravel()
    cols = np.tile(elems, (1, m)).ravel()
    return sp.coo_matrix((Ae.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def _eval_xy(fn: NumFunc, pts: np.ndarray, t: float = 0.0) -> np.ndarray:
    if pts.shape[-1] == 1:
        return fn(pts[..., 0], None, t)
    return fn(pts[..., 0], pts[..., 1], t)


def element_matrices(mesh: Mesh, prob: Problem, e: int) -> dict[str, Any]:
    """Matrices élémentaires d'un élément (affichage pédagogique)."""
    geo = p1_geometry(mesh)
    bary, w = (segment_rule(3) if mesh.dim == 1 else triangle_rule(4))
    Xq = quad_points(mesh, {"P": geo["P"][e:e + 1]}, bary)[0]
    kq = _eval_xy(prob.k, Xq)
    cq = _eval_xy(prob.c, Xq)
    fq = _eval_xy(prob.f, Xq, 0.0)
    meas = geo["meas"][e]
    G = geo["grad"][e]
    Ke = meas * float(np.dot(w, kq)) * (G @ G.T)
    Me = meas * np.einsum("q,qi,qj->ij", w, bary, bary)
    Mce = meas * np.einsum("q,q,qi,qj->ij", w, cq, bary, bary)
    bvec = np.array([prob.bx] if mesh.dim == 1 else [prob.bx, prob.by])
    Be = meas * np.einsum("qi,j->ij", bary * w[:, None], G @ bvec)
    Fe = meas * np.einsum("q,q,qi->i", w, fq, bary)
    return {"nodes": mesh.elems[e], "coords": geo["P"][e], "meas": meas, "grad": G, "K": Ke, "M": Me,
            "Mc": Mce, "B": Be, "F": Fe, "kbar": float(np.dot(w, kq))}


# =============================================================================
# 7. SYSTÈMES DISCRETS : ÉLÉMENTS FINIS P1 ET DIFFÉRENCES FINIES
#    Forme commune : M U' (ou U'') + K U = F(t) ;  K U = F en stationnaire.
#    FDM : M = D (volumes de contrôle, diagonale), la matrice « différences finies »
#    classique est A = D⁻¹K.  FEM : M = matrice de masse (cohérente ou condensée).
# =============================================================================
class DiscreteSystem:
    def __init__(self, **kw: Any):
        self.method: str = kw["method"]
        self.dim: int = kw["dim"]
        self.nodes: np.ndarray = kw["nodes"]
        self.K: sp.csr_matrix = kw["K"].tocsr()
        self.M: sp.csr_matrix = kw["M"].tocsr()
        self.M_cons: sp.csr_matrix = kw.get("M_cons", kw["M"]).tocsr()
        self._load: Callable[[float], np.ndarray] = kw["load"]
        self.dir_idx: np.ndarray = np.asarray(kw["dir_idx"], dtype=np.int64)
        self._dir_val: Callable[[float], np.ndarray] = kw["dir_val"]
        self.W: np.ndarray = kw["W"]
        self.components: dict[str, sp.csr_matrix] = kw.get("components", {})
        self.info: dict[str, Any] = kw.get("info", {})
        self.mesh: Mesh | None = kw.get("mesh")
        self.grid: dict | None = kw.get("grid")
        self.load_time_dependent: bool = kw.get("load_time_dependent", False)
        self.dir_time_dependent: bool = kw.get("dir_time_dependent", False)
        self.n = self.K.shape[0]
        mask = np.ones(self.n, bool)
        mask[self.dir_idx] = False
        self.free = np.nonzero(mask)[0]
        self.m_is_diag = bool(kw.get("m_is_diag", False))
        self.kernel = bool(kw.get("kernel", False))
        self.mvec = np.asarray(self.M_cons @ np.ones(self.n))
        self._F_cache: dict[float, np.ndarray] = {}

    def load(self, t: float) -> np.ndarray:
        if not self.load_time_dependent:
            if "static" not in self._F_cache:
                self._F_cache["static"] = self._load(0.0)
            return self._F_cache["static"]
        return self._load(t)

    def dir_val(self, t: float) -> np.ndarray:
        return self._dir_val(t)

    def fd_matrix(self) -> sp.csr_matrix:
        """Matrice des différences finies classique A = D⁻¹ K (FDM)."""
        d = np.asarray(self.M.diagonal())
        return sp.diags(1.0 / d) @ self.K

    def memory_bytes(self) -> int:
        tot = 0
        for mat in (self.K, self.M):
            tot += mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
        return int(tot + self.nodes.nbytes + 4 * self.n * 8)


def _dirichlet_assembler(prob: Problem, side_node_lists: dict[str, np.ndarray], nodes: np.ndarray):
    """Indices des nœuds de Dirichlet et fonction g_D(t) (moyenne aux coins partagés)."""
    dsides = [s for s in prob.sides if prob.bc[s]["type"] == "dirichlet"]
    if not dsides:
        return np.zeros(0, dtype=np.int64), (lambda t: np.zeros(0)), False
    allnodes = np.unique(np.concatenate([side_node_lists[s] for s in dsides]))
    pos = {int(p): i for i, p in enumerate(allnodes)}
    lists = [(np.array([pos[int(p)] for p in side_node_lists[s]], dtype=np.int64), side_node_lists[s], s) for s in dsides]
    count = np.zeros(len(allnodes))
    for loc, _, _ in lists:
        np.add.at(count, loc, 1.0)
    tdep = any(prob.bcf[s]["g"].depends_t for s in dsides)

    def dir_val(t: float) -> np.ndarray:
        acc = np.zeros(len(allnodes))
        for loc, glob, s in lists:
            pts = nodes[glob]
            vals = prob.bcf[s]["g"](pts[:, 0], pts[:, 1] if nodes.shape[1] > 1 else None, t)
            np.add.at(acc, loc, vals)
        return acc / count

    return allnodes, dir_val, tdep


def assemble_fem(prob: Problem, mesh: Mesh, lumped: bool = False) -> DiscreteSystem:
    t0 = time.perf_counter()
    n = mesh.n_nodes
    geo = p1_geometry(mesh)
    meas, G = geo["meas"], geo["grad"]
    d = mesh.dim
    bary, w = (segment_rule(3) if d == 1 else triangle_rule(4))
    Xq = quad_points(mesh, geo, bary)
    # rigidité ∫ k ∇φj·∇φi
    kq = _eval_xy(prob.k, Xq)
    kbar = kq @ w
    Ke = (kbar * meas)[:, None, None] * np.einsum("eid,ejd->eij", G, G)
    K_k = _scatter(mesh.elems, Ke, n)
    # masse ∫ φj φi (exacte)
    Mref = np.einsum("q,qi,qj->ij", w, bary, bary)
    Me = meas[:, None, None] * Mref[None]
    M = _scatter(mesh.elems, Me, n)
    comps: dict[str, sp.csr_matrix] = {"K_diffusion": K_k, "M_masse": M}
    K = K_k.copy()
    # réaction ∫ c φj φi
    if not (prob.c_expr.is_number and prob.c_expr == 0):
        cq = _eval_xy(prob.c, Xq)
        Mce = meas[:, None, None] * np.einsum("q,eq,qi,qj->eij", w, cq, bary, bary)
        Mc = _scatter(mesh.elems, Mce, n)
        comps["M_reaction"] = Mc
        K = K + Mc
    # advection ∫ (b·∇φj) φi
    if prob.bx != 0.0 or prob.by != 0.0:
        bvec = np.array([prob.bx] if d == 1 else [prob.bx, prob.by])
        bg = G @ bvec  # (E, nloc)
        Be = (meas / (d + 1))[:, None, None] * np.broadcast_to(bg[:, None, :], (len(meas), d + 1, d + 1))
        B = _scatter(mesh.elems, np.ascontiguousarray(Be), n)
        comps["B_advection"] = B
        K = K + B
    # conditions de Robin : ∫_Γ β φj φi
    sb, sw = segment_rule(3)
    robin_sides = [s for s in prob.sides if prob.bc[s]["type"] == "robin"]
    natural_sides = [s for s in prob.sides if prob.bc[s]["type"] in ("neumann", "robin")]
    if robin_sides:
        rows, cols, vals = [], [], []
        for s in robin_sides:
            sid = mesh.side_names.index(s)
            faces = mesh.bfaces[mesh.bside == sid]
            if d == 1:
                p = faces[:, 0]
                bv = prob.bcf[s]["beta"](mesh.nodes[p, 0])
                rows.append(p)
                cols.append(p)
                vals.append(bv)
            else:
                P0, P1 = mesh.nodes[faces[:, 0]], mesh.nodes[faces[:, 1]]
                L = np.linalg.norm(P1 - P0, axis=1)
                Xs = P0[:, None, :] * sb[None, :, 0:1] + P1[:, None, :] * sb[None, :, 1:2]
                bq = prob.bcf[s]["beta"](Xs[..., 0], Xs[..., 1])
                Re = L[:, None, None] * np.einsum("q,eq,qi,qj->eij", sw, bq, sb, sb)
                m = 2
                rows.append(np.repeat(faces, m, axis=1).ravel())
                cols.append(np.tile(faces, (1, m)).ravel())
                vals.append(Re.ravel())
        R = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        comps["R_robin"] = R
        K = K + R
    # second membre
    f_tdep = prob.f.depends_t
    Fq_static = None
    if not f_tdep:
        fq = _eval_xy(prob.f, Xq, 0.0)
        Fe = meas[:, None] * np.einsum("eq,q,qi->ei", fq, w, bary)
        Fq_static = np.bincount(mesh.elems.ravel(), Fe.ravel(), minlength=n)
    nat_data = []
    for s in natural_sides:
        sid = mesh.side_names.index(s)
        faces = mesh.bfaces[mesh.bside == sid]
        if d == 1:
            nat_data.append((s, faces[:, 0], None, None))
        else:
            P0, P1 = mesh.nodes[faces[:, 0]], mesh.nodes[faces[:, 1]]
            L = np.linalg.norm(P1 - P0, axis=1)
            Xs = P0[:, None, :] * sb[None, :, 0:1] + P1[:, None, :] * sb[None, :, 1:2]
            nat_data.append((s, faces, L, Xs))
    g_tdep = any(prob.bcf[s]["g"].depends_t for s in natural_sides)
    M_cons = M

    def load(t: float) -> np.ndarray:
        if Fq_static is not None:
            F = Fq_static.copy()
        else:  # second membre interpolé F = M Π_h f(t) (ordre O(h²) en norme L²)
            fn = _eval_xy(prob.f, mesh.nodes, t)
            F = np.asarray(M_cons @ fn)
        for s, faces, L, Xs in nat_data:
            if d == 1:
                p = faces
                F[p] += prob.bcf[s]["g"](mesh.nodes[p, 0], None, t)
            else:
                gq = prob.bcf[s]["g"](Xs[..., 0], Xs[..., 1], t)
                Ge = L[:, None] * np.einsum("eq,q,qi->ei", gq, sw, sb)
                F += np.bincount(faces.ravel(), Ge.ravel(), minlength=n)
        return F

    side_lists = {s: mesh.side_nodes(s) for s in prob.sides}
    dir_idx, dir_val, d_tdep = _dirichlet_assembler(prob, side_lists, mesh.nodes)
    # zéros structurels exacts (ex. arête d'angles opposés droits : cot α + cot β = 0) : non stockés
    K = K.tocsr()
    K.eliminate_zeros()
    for mat in comps.values():
        mat.eliminate_zeros()
    W = np.asarray(M.sum(axis=1)).ravel()
    Muse = sp.diags(W).tocsr() if lumped else M
    t1 = time.perf_counter()
    return DiscreteSystem(method="fem", dim=d, nodes=mesh.nodes, K=K, M=Muse, M_cons=M_cons, load=load,
                          dir_idx=dir_idx, dir_val=dir_val, W=W, components=comps, mesh=mesh,
                          load_time_dependent=(f_tdep or g_tdep), dir_time_dependent=d_tdep,
                          m_is_diag=lumped, kernel=prob.is_pure_neumann(),
                          info={"assembly_time": t1 - t0, "quadrature": "Gauss 3 pts (1D)" if d == 1 else
                                "Dunavant 6 pts, degré 4 (2D)", "lumped": lumped})


def fdm_grid(prob: Problem, nx: int, ny: int) -> dict[str, Any]:
    xs = np.linspace(prob.xa, prob.xb, nx + 1)
    hx = (prob.xb - prob.xa) / nx
    if prob.dim == 1:
        ys, hy, nyy = np.array([0.0]), 1.0, 0
    else:
        ys = np.linspace(prob.ya, prob.yb, ny + 1)
        hy = (prob.yb - prob.ya) / ny
        nyy = ny
    XX, YY = np.meshgrid(xs, ys)
    nodes = np.stack([XX.ravel(), YY.ravel()], axis=1) if prob.dim == 2 else xs[:, None]
    return {"xs": xs, "ys": ys, "hx": hx, "hy": hy, "nx": nx, "ny": nyy, "nodes": nodes}


def assemble_fdm(prob: Problem, nx: int, ny: int, adv_scheme: str = "centered") -> DiscreteSystem:
    """Différences finies (forme volumes de contrôle sur grille uniforme, nœuds aux sommets).
    À l'intérieur : schéma à 3 points (1D) / 5 points (2D) d'ordre 2 avec k aux demi-points.
    Neumann/Robin : demi-mailles, équivalent au point fantôme centré (ordre 2 global)."""
    t0 = time.perf_counter()
    g = fdm_grid(prob, nx, ny)
    xs, ys, hx, hy = g["xs"], g["ys"], g["hx"], g["hy"]
    nyy = g["ny"]
    nxp, nyp = nx + 1, nyy + 1
    n = nxp * nyp
    nodes = g["nodes"]
    d = prob.dim
    I, J = np.meshgrid(np.arange(nxp), np.arange(nyp))
    I, J = I.ravel(), J.ravel()
    wx = np.full(nxp, hx)
    wx[[0, -1]] = hx / 2
    if d == 2:
        wy = np.full(nyp, hy)
        wy[[0, -1]] = hy / 2
    else:
        wy = np.array([1.0])
    vol = wx[I] * wy[J]
    X = nodes[:, 0]
    Y = nodes[:, 1] if d == 2 else np.zeros(n)
    idx = lambda i, j: i + j * nxp  # noqa: E731
    rows, cols, vals = [], [], []
    # arêtes en x
    ii, jj = np.meshgrid(np.arange(nx), np.arange(nyp))
    ii, jj = ii.ravel(), jj.ravel()
    p, q = idx(ii, jj), idx(ii + 1, jj)
    kmid = prob.k(0.5 * (xs[ii] + xs[ii + 1]), ys[jj] if d == 2 else None)
    T = kmid * wy[jj] / hx
    rows += [p, q, p, q]
    cols += [p, q, q, p]
    vals += [T, T, -T, -T]
    if d == 2:
        ii, jj = np.meshgrid(np.arange(nxp), np.arange(nyy))
        ii, jj = ii.ravel(), jj.ravel()
        p, q = idx(ii, jj), idx(ii, jj + 1)
        kmid = prob.k(xs[ii], 0.5 * (ys[jj] + ys[jj + 1]))
        T = kmid * wx[ii] / hy
        rows += [p, q, p, q]
        cols += [p, q, q, p]
        vals += [T, T, -T, -T]
    S = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    comps: dict[str, sp.csr_matrix] = {"S_diffusion": S}
    K = S.copy()
    # côtés
    side_mask = {"left": I == 0, "right": I == nx}
    side_meas = {"left": wy[J], "right": wy[J]}
    if d == 2:
        side_mask.update(bottom=J == 0, top=J == nyy)
        side_meas.update(bottom=wx[I], top=wx[I])
    side_lists = {s: np.nonzero(side_mask[s])[0] for s in prob.sides}
    dir_idx, dir_val, d_tdep = _dirichlet_assembler(prob, side_lists, nodes if d == 2 else np.stack([X, Y], 1))
    is_dir = np.zeros(n, bool)
    is_dir[dir_idx] = True
    # Robin
    rdiag = np.zeros(n)
    for s in prob.sides:
        if prob.bc[s]["type"] == "robin":
            m = side_mask[s]
            rdiag[m] += prob.bcf[s]["beta"](X[m], Y[m] if d == 2 else None) * side_meas[s][m]
    if np.any(rdiag):
        R = sp.diags(rdiag).tocsr()
        comps["R_robin"] = R
        K = K + R
    # réaction
    if not (prob.c_expr.is_number and prob.c_expr == 0):
        C = sp.diags(prob.c(X, Y if d == 2 else None) * vol).tocsr()
        comps["C_reaction"] = C
        K = K + C
    # advection (forme non conservative, nœud par nœud)
    adv_bc_terms: list[tuple[str, np.ndarray, float, float]] = []
    if prob.bx != 0.0 or prob.by != 0.0:
        brow, bcol, bval = [], [], []
        adiag = np.zeros(n)
        for comp, bcomp, h, coord_idx, nmax, sides in (("x", prob.bx, hx, I, nx, ("left", "right")),
                                                       ("y", prob.by, hy, J, nyy, ("bottom", "top"))):
            if bcomp == 0.0 or (comp == "y" and d == 1):
                continue
            step = 1 if comp == "x" else nxp
            interior = (coord_idx > 0) & (coord_idx < nmax) & ~is_dir
            pi = np.nonzero(interior)[0]
            if adv_scheme == "upwind":
                if bcomp > 0:
                    brow += [pi, pi]
                    bcol += [pi, pi - step]
                    bval += [vol[pi] * bcomp / h, -vol[pi] * bcomp / h]
                else:
                    brow += [pi, pi]
                    bcol += [pi + step, pi]
                    bval += [vol[pi] * bcomp / h, -vol[pi] * bcomp / h]
            else:
                brow += [pi, pi]
                bcol += [pi + step, pi - step]
                bval += [vol[pi] * bcomp / (2 * h), -vol[pi] * bcomp / (2 * h)]
            for s, sgn in zip(sides, (-1.0, 1.0)):
                m = side_mask[s] & ~is_dir
                if not np.any(m):
                    continue
                pm = np.nonzero(m)[0]
                kk = prob.k(X[pm], Y[pm] if d == 2 else None)
                beta = (prob.bcf[s]["beta"](X[pm], Y[pm] if d == 2 else None)
                        if prob.bc[s]["type"] == "robin" else np.zeros(len(pm)))
                # ∂u/∂(axe) = sgn (g − β u)/k  (dérivée normale imposée)
                adiag[pm] += -sgn * vol[pm] * bcomp * beta / kk
                adv_bc_terms.append((s, pm, sgn * bcomp, 0.0))
        Badv = sp.coo_matrix((np.concatenate(bval) if bval else np.zeros(0),
                              (np.concatenate(brow) if brow else np.zeros(0, int),
                               np.concatenate(bcol) if bcol else np.zeros(0, int))), shape=(n, n)).tocsr()
        Badv = Badv + sp.diags(adiag)
        comps["B_advection"] = Badv.tocsr()
        K = K + Badv
    K = K.tocsr()
    K.eliminate_zeros()
    for mat in comps.values():
        mat.eliminate_zeros()
    Dm = sp.diags(vol).tocsr()
    natural = [s for s in prob.sides if prob.bc[s]["type"] in ("neumann", "robin")]
    f_tdep = prob.f.depends_t
    g_tdep = any(prob.bcf[s]["g"].depends_t for s in natural)

    def load(t: float) -> np.ndarray:
        F = prob.f(X, Y if d == 2 else None, t) * vol
        for s in natural:
            m = side_mask[s]
            F[m] += prob.bcf[s]["g"](X[m], Y[m] if d == 2 else None, t) * side_meas[s][m]
        for s, pm, coef, _ in adv_bc_terms:
            kk = prob.k(X[pm], Y[pm] if d == 2 else None)
            gv = prob.bcf[s]["g"](X[pm], Y[pm] if d == 2 else None, t)
            F[pm] -= vol[pm] * coef * gv / kk
        return F

    t1 = time.perf_counter()
    grid = {"xs": xs, "ys": ys, "hx": hx, "hy": hy, "nx": nx, "ny": nyy, "I": I, "J": J, "vol": vol,
            "side_mask": side_mask}
    return DiscreteSystem(method="fdm", dim=d, nodes=nodes, K=K, M=Dm, M_cons=Dm, load=load, dir_idx=dir_idx,
                          dir_val=dir_val, W=vol, components=comps, grid=grid,
                          load_time_dependent=(f_tdep or g_tdep), dir_time_dependent=d_tdep, m_is_diag=True,
                          kernel=prob.is_pure_neumann(),
                          info={"assembly_time": t1 - t0, "adv_scheme": adv_scheme})


def build_system(prob: Problem, cfg: dict, n_override: tuple[int, int] | None = None,
                 extra_refine: int = 0) -> DiscreteSystem:
    """Construit le système discret selon la méthode choisie (avec contrôle des tailles)."""
    mcfg = dict(cfg["mesh"])
    if n_override is not None:
        mcfg["nx"], mcfg["ny"] = n_override
    if prob.method == "fem":
        mesh = build_mesh(prob, mcfg, extra_refine)
        return assemble_fem(prob, mesh, lumped=bool(cfg["time"].get("lumped", False)) and prob.time_dependent)
    nx = int(mcfg["nx"]) * (2 ** extra_refine)
    ny = int(mcfg.get("ny", nx)) * (2 ** extra_refine)
    if prob.dim == 1 and nx > LIMITS["n1d_max"] * 4:
        raise ConfigError("Grille 1D trop fine.")
    if prob.dim == 2 and (nx + 1) * (ny + 1) > (LIMITS["n2d_max"] * 2 + 1) ** 2:
        raise ConfigError("Grille 2D trop fine.")
    return assemble_fdm(prob, nx, ny, cfg.get("adv_scheme", "centered"))


def mesh_size(sysm: DiscreteSystem) -> float:
    """Pas caractéristique h (plus grand diamètre d'élément / pas de grille)."""
    if sysm.method == "fdm":
        return float(max(sysm.grid["hx"], sysm.grid["hy"] if sysm.dim == 2 else 0.0))
    return float(sysm.mesh.quality()["h_max"])


def mesh_hmin(sysm: DiscreteSystem) -> float:
    if sysm.method == "fdm":
        return float(min(sysm.grid["hx"], sysm.grid["hy"] if sysm.dim == 2 else math.inf))
    return float(sysm.mesh.quality()["h_min"])


# =============================================================================
# 8. SOLVEURS
# =============================================================================
def is_symmetric(A: sp.spmatrix, rtol: float = 1e-12) -> bool:
    A = A.tocsr()
    if A.shape[0] != A.shape[1]:
        return False
    diff = abs(A - A.T)
    scale = abs(A).max() if A.nnz else 1.0
    return bool((diff.max() if diff.nnz else 0.0) <= rtol * max(scale, 1e-300))


def time_step_plan(prob: Problem, cfg: dict, h_min: float, n_unknowns: int) -> dict[str, Any]:
    """Pas de temps à partir du nombre de stabilité choisi (r, σ ou λ), de Δt ou de N_t."""
    tm = cfg["time"]
    T = prob.T
    mode = tm["dt_mode"]
    if prob.cls == "parabolic":
        kmax = prob.k_max()
        name, scale = "r", kmax / h_min ** 2
    elif prob.cls == "hyperbolic":
        name, scale = "σ", (prob.cw or 1.0) / h_min
    else:
        name, scale = "λ", abs(prob.a or 1.0) / h_min
    if mode == "number":
        dt = tm["number"] / scale
    elif mode == "dt":
        dt = tm["dt"]
    else:
        dt = T / tm["nt"]
    nt = max(1, int(math.ceil(T / dt - 1e-9)))
    if nt > LIMITS["nt_max"]:
        raise ConfigError(f"Nombre de pas de temps trop élevé ({nt} > {int(LIMITS['nt_max'])}) : augmentez Δt "
                          "ou réduisez T.")
    if nt * max(n_unknowns, 1) > LIMITS["work_max"]:
        raise ConfigError(f"Calcul trop coûteux : {nt} pas × {n_unknowns} inconnues > {LIMITS['work_max']:.1e}. "
                          "Réduisez le maillage ou le nombre de pas.")
    dt = T / nt
    numbers = {name: dt * scale, "dt": dt, "nt": nt, "number_name": name}
    if prob.cls == "parabolic" and (prob.bx or prob.by):
        kmin = prob.coefficient_range("k")[0]
        bn = math.hypot(prob.bx, prob.by)
        numbers["Pe_h"] = bn * h_min / (2 * kmin)
        numbers["CFL_adv"] = bn * dt / h_min
    return numbers


def _jacobi_operator(A: sp.csr_matrix) -> spla.LinearOperator:
    d = A.diagonal()
    d = np.where(np.abs(d) > 0, d, 1.0)
    return spla.LinearOperator(A.shape, matvec=lambda v: v / d)


def solve_linear(A: sp.csr_matrix, b: np.ndarray, solver: dict) -> tuple[np.ndarray, dict[str, Any]]:
    """Résolution de A x = b : direct (SuperLU) ou itératif (CG / GMRES / BiCGSTAB) avec historique."""
    kind = solver.get("kind", "direct")
    info: dict[str, Any] = {"kind": kind, "requested": kind}
    sym = is_symmetric(A)
    info["symmetric"] = sym
    t0 = time.perf_counter()
    nb = float(np.linalg.norm(b)) or 1.0
    if kind == "cg" and not sym:
        info["note"] = "Matrice non symétrique : CG n'est pas applicable, GMRES utilisé à la place."
        kind = "gmres"
        info["kind"] = kind
    if kind == "direct":
        lu = spla.splu(A.tocsc(), permc_spec="COLAMD")
        t1 = time.perf_counter()
        x = lu.solve(b)
        info.update(factor_time=t1 - t0, solve_time=time.perf_counter() - t1,
                    fill=int(lu.L.nnz + lu.U.nnz), nnz=int(A.nnz), iterations=0, history=[])
    else:
        prec = solver.get("precond", "none")
        Mop = None
        if prec == "jacobi":
            Mop = _jacobi_operator(A)
        elif prec == "ilu":
            if kind == "cg":
                info["note"] = "ILU non symétrique incompatible avec CG : préconditionneur de Jacobi utilisé."
                Mop = _jacobi_operator(A)
                prec = "jacobi"
            else:
                try:
                    ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
                    Mop = spla.LinearOperator(A.shape, matvec=ilu.solve)
                except RuntimeError:
                    info["note"] = "Factorisation ILU impossible (pivot nul) : préconditionneur de Jacobi utilisé."
                    Mop = _jacobi_operator(A)
                    prec = "jacobi"
        info["precond"] = prec
        hist: list[float] = []
        tol = float(solver.get("tol", 1e-10))
        maxiter = int(solver.get("maxiter", 5000))
        if kind == "cg":
            x, code = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=Mop,
                              callback=lambda xk: hist.append(float(np.linalg.norm(b - A @ xk)) / nb))
        elif kind == "bicgstab":
            x, code = spla.bicgstab(A, b, rtol=tol, atol=0.0, maxiter=maxiter, M=Mop,
                                    callback=lambda xk: hist.append(float(np.linalg.norm(b - A @ xk)) / nb))
        else:
            x, code = spla.gmres(A, b, rtol=tol, atol=0.0, restart=50, maxiter=max(1, maxiter // 50), M=Mop,
                                 callback=lambda r: hist.append(float(r)), callback_type="pr_norm")
        info.update(code=int(code), converged=(code == 0), iterations=len(hist), history=hist,
                    factor_time=0.0, solve_time=time.perf_counter() - t0, nnz=int(A.nnz))
    res = float(np.linalg.norm(b - A @ x)) / nb
    info["residual"] = res
    info["time"] = time.perf_counter() - t0
    return x, info


def solve_stationary(sysm: DiscreteSystem, solver: dict) -> tuple[np.ndarray, dict[str, Any]]:
    n = sysm.n
    U = np.zeros(n)
    F = sysm.load(0.0)
    if sysm.kernel:
        m = sysm.mvec
        Aug = sp.bmat([[sysm.K, sp.csr_matrix(m[:, None])], [sp.csr_matrix(m[None, :]), None]]).tocsc()
        rhs = np.concatenate([F, [0.0]])
        x, info = solve_linear(Aug.tocsr(), rhs, {"kind": "direct"})
        U = x[:n]
        info["multiplier"] = float(x[n])
        info["kind"] = "direct (système augmenté, multiplicateur de Lagrange pour ∫u = 0)"
        info["discrete_compat"] = float(F.sum())
        info["dofs"] = n
        return U, info
    D, Fr = sysm.dir_idx, sysm.free
    gD = sysm.dir_val(0.0)
    U[D] = gD
    KFF = sysm.K[Fr][:, Fr]
    b = F[Fr] - (sysm.K[Fr][:, D] @ gD if len(D) else 0.0)
    x, info = solve_linear(KFF.tocsr(), np.asarray(b, dtype=float), solver)
    U[Fr] = x
    info["dofs"] = len(Fr)
    return U, info


def _frame_steps(nt: int, nframes: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, nt, max(2, min(nframes, nt + 1)))).astype(int))


def _frames_budget(n: int, requested: int) -> int:
    return int(max(2, min(requested, LIMITS["frame_values_max"] // max(n, 1))))


def _blowup_scale(*arrays: np.ndarray) -> float:
    s = 1.0
    for a in arrays:
        if a is not None and len(a):
            s = max(s, float(np.nanmax(np.abs(a))))
    return s


def data_scale(prob: Problem, sysm: DiscreteSystem) -> float:
    """Échelle des données (u₀, g_D, T·f, solution exacte), majorée par la croissance physique
    admissible e^{max(0,−c_min)T} : sert de référence pour détecter une explosion numérique."""
    nodes = sysm.nodes
    T = prob.T
    arrs = [_eval_xy(prob.u0, nodes), T * _eval_xy(prob.f, nodes, 0.0), T * _eval_xy(prob.f, nodes, T)]
    if prob.time_order == 2:
        arrs.append(T * _eval_xy(prob.v0, nodes))
    if len(sysm.dir_idx):
        arrs += [sysm.dir_val(0.0), sysm.dir_val(T)]
    if prob.exact is not None:
        arrs += [_eval_xy(prob.exact, nodes, 0.0), _eval_xy(prob.exact, nodes, T)]
    s = _blowup_scale(*arrs)
    cmin = prob.coefficient_range("c")[0]
    return s * math.exp(min(max(0.0, -cmin) * T, 50.0))


def _initial_state(prob: Problem, sysm: DiscreteSystem) -> tuple[np.ndarray, list[str]]:
    nodes = sysm.nodes
    U0 = _eval_xy(prob.u0, nodes)
    warn = []
    if len(sysm.dir_idx):
        g0 = sysm.dir_val(0.0)
        mism = float(np.max(np.abs(U0[sysm.dir_idx] - g0))) if len(g0) else 0.0
        if mism > 1e-8 * (1 + float(np.max(np.abs(U0)))):
            warn.append(f"Incompatibilité entre u₀ et la donnée de Dirichlet à t = 0 (écart max {mism:.2e}) : "
                        "la valeur de Dirichlet est imposée aux nœuds du bord.")
        U0[sysm.dir_idx] = g0
    return U0, warn


def solve_theta(prob: Problem, sysm: DiscreteSystem, dt: float, nt: int, theta: float,
                nframes: int, exact: bool = True) -> dict[str, Any]:
    """θ-schéma : (M + ϑΔt K)Uⁿ⁺¹ = (M − (1−ϑ)Δt K)Uⁿ + Δt(ϑFⁿ⁺¹ + (1−ϑ)Fⁿ)."""
    t0 = time.perf_counter()
    M, K = sysm.M, sysm.K
    Fr, D = sysm.free, sysm.dir_idx
    Lm = (M + theta * dt * K).tocsr()
    Rm = (M - (1 - theta) * dt * K).tocsr()
    LFF = Lm[Fr][:, Fr].tocsc()
    LFD = Lm[Fr][:, D].tocsr() if len(D) else None
    RF = Rm[Fr].tocsr()
    explicit_diag = theta == 0.0 and sysm.m_is_diag
    if explicit_diag:
        dinv = 1.0 / LFF.diagonal()
        solve = lambda r: r * dinv  # noqa: E731
        fact_info = "aucune (masse diagonale, schéma explicite)"
    else:
        lu = spla.splu(LFF, permc_spec="COLAMD")
        solve = lu.solve
        fact_info = f"SuperLU, nnz(L+U) = {lu.L.nnz + lu.U.nnz}"
    t_fact = time.perf_counter() - t0
    U, warn = _initial_state(prob, sysm)
    nframes = _frames_budget(sysm.n, nframes)
    fsteps = set(_frame_steps(nt, nframes).tolist())
    frames, ftimes = [U.astype(np.float32).copy()], [0.0]
    Fn = sysm.load(0.0)
    scale = data_scale(prob, sysm) * LIMITS["blowup_factor"]
    hist_every = max(1, nt // 4000)
    h_t, h_norm = [0.0], [float(np.sqrt(U @ (sysm.M_cons @ U)))]
    blowup = None
    tl = time.perf_counter()
    for n in range(nt):
        t1 = (n + 1) * dt
        Fn1 = sysm.load(t1)
        rhs = RF @ U + dt * (theta * Fn1[Fr] + (1 - theta) * Fn[Fr])
        if len(D):
            g1 = sysm.dir_val(t1)
            if LFD is not None:
                rhs = rhs - LFD @ g1
        U = U.copy()
        U[Fr] = solve(rhs)
        if len(D):
            U[D] = g1
        Fn = Fn1
        if (n + 1) % hist_every == 0 or n + 1 == nt:
            nrm = float(np.sqrt(abs(U @ (sysm.M_cons @ U))))
            h_t.append(t1)
            h_norm.append(nrm)
        umax = float(np.max(np.abs(U)))
        if not math.isfinite(umax) or umax > scale:
            blowup = {"step": n + 1, "time": t1, "value": umax}
            frames.append(np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            ftimes.append(t1)
            break
        if (n + 1) in fsteps:
            frames.append(U.astype(np.float32).copy())
            ftimes.append(t1)
    t_loop = time.perf_counter() - tl
    return {"U": U, "frames": np.array(frames), "times": np.array(ftimes), "hist_t": np.array(h_t),
            "hist_norm": np.array(h_norm), "blowup": blowup, "warnings": warn,
            "timings": {"factorization": t_fact, "time_loop": t_loop}, "factor_info": fact_info,
            "steps_done": (blowup["step"] if blowup else nt)}


def solve_wave(prob: Problem, sysm: DiscreteSystem, dt: float, nt: int, beta: float, nframes: int) -> dict[str, Any]:
    """Schéma à trois niveaux (Newmark γ = 1/2) :
    (M + βΔt²K)Uⁿ⁺¹ = (2M − (1−2β)Δt²K)Uⁿ − (M + βΔt²K)Uⁿ⁻¹ + Δt²(βFⁿ⁺¹ + (1−2β)Fⁿ + βFⁿ⁻¹).
    β = 0 : saute-mouton explicite ; β = 1/4 : accélération moyenne (inconditionnellement stable)."""
    t0 = time.perf_counter()
    M, K = sysm.M, sysm.K
    Fr, D = sysm.free, sysm.dir_idx
    Lm = (M + beta * dt * dt * K).tocsr()
    LFF = Lm[Fr][:, Fr].tocsc()
    LFD = Lm[Fr][:, D].tocsr() if len(D) else None
    R1 = (2 * M - (1 - 2 * beta) * dt * dt * K).tocsr()
    explicit_diag = beta == 0.0 and sysm.m_is_diag
    if explicit_diag:
        dinv = 1.0 / LFF.diagonal()
        solve = lambda r: r * dinv  # noqa: E731
        fact_info = "aucune (masse diagonale, schéma explicite)"
    else:
        lu = spla.splu(LFF, permc_spec="COLAMD")
        solve = lu.solve
        fact_info = f"SuperLU, nnz(L+U) = {lu.L.nnz + lu.U.nnz}"
    MFF = M[Fr][:, Fr].tocsc()
    msolve = (lambda r: r / MFF.diagonal()) if sysm.m_is_diag else spla.splu(MFF).solve
    t_fact = time.perf_counter() - t0
    U0, warn = _initial_state(prob, sysm)
    V0 = _eval_xy(prob.v0, sysm.nodes)
    F0 = sysm.load(0.0)
    # premier pas (Taylor d'ordre 2) : U¹ = U⁰ + Δt V⁰ + Δt²/2 · M⁻¹(F⁰ − K U⁰)
    acc = np.zeros(sysm.n)
    r0 = (F0 - K @ U0)[Fr]
    if len(D) and not sysm.m_is_diag:
        aD = (sysm.dir_val(dt) - 2 * sysm.dir_val(0.0) + sysm.dir_val(-dt)) / dt ** 2
        r0 = r0 - M[Fr][:, D] @ aD
    acc[Fr] = msolve(r0)
    U1 = U0 + dt * V0 + 0.5 * dt * dt * acc
    if len(D):
        U1[D] = sysm.dir_val(dt)
    nframes = _frames_budget(sysm.n, nframes)
    fsteps = set(_frame_steps(nt, nframes).tolist())
    frames, ftimes = [U0.astype(np.float32)], [0.0]
    if 1 in fsteps:
        frames.append(U1.astype(np.float32))
        ftimes.append(dt)
    Mcorr = (M + (beta - 0.25) * dt * dt * K).tocsr()

    def energy(Un1: np.ndarray, Un: np.ndarray) -> float:
        V = (Un1 - Un) / dt
        Ub = 0.5 * (Un1 + Un)
        return float(0.5 * V @ (Mcorr @ V) + 0.5 * Ub @ (K @ Ub))

    e_t, e_val = [0.5 * dt], [energy(U1, U0)]
    scale = data_scale(prob, sysm) * LIMITS["blowup_factor"]
    Fm, Fn = F0, sysm.load(dt)
    Um, Un = U0, U1
    blowup = None
    hist_every = max(1, nt // 4000)
    tl = time.perf_counter()
    for n in range(1, nt):
        t1 = (n + 1) * dt
        Fp = sysm.load(t1)
        rhs_full = R1 @ Un - Lm @ Um + dt * dt * (beta * Fp + (1 - 2 * beta) * Fn + beta * Fm)
        rhs = rhs_full[Fr]
        if len(D):
            g1 = sysm.dir_val(t1)
            if LFD is not None:
                rhs = rhs - LFD @ g1
        Up = np.empty_like(Un)
        Up[Fr] = solve(rhs)
        if len(D):
            Up[D] = g1
        if (n + 1) % hist_every == 0 or n + 1 == nt:
            e_t.append(t1 - 0.5 * dt)
            e_val.append(energy(Up, Un))
        umax = float(np.max(np.abs(Up)))
        Um, Un = Un, Up
        Fm, Fn = Fn, Fp
        if not math.isfinite(umax) or umax > scale:
            blowup = {"step": n + 1, "time": t1, "value": umax}
            frames.append(np.nan_to_num(Up, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            ftimes.append(t1)
            break
        if (n + 1) in fsteps:
            frames.append(Up.astype(np.float32))
            ftimes.append(t1)
    t_loop = time.perf_counter() - tl
    U = Un if nt > 1 else U1
    return {"U": U, "frames": np.array(frames), "times": np.array(ftimes), "hist_t": np.array(e_t),
            "hist_energy": np.array(e_val), "blowup": blowup, "warnings": warn,
            "timings": {"factorization": t_fact, "time_loop": t_loop}, "factor_info": fact_info,
            "steps_done": (blowup["step"] if blowup else nt)}


def transport_exact(prob: Problem, x: np.ndarray, t: float) -> np.ndarray:
    """Solution exacte du transport (formule fournie, ou caractéristiques u₀(x − a t))."""
    if prob.exact is not None:
        return prob.exact(x, None, t)
    a = prob.a or 1.0
    xi = x - a * t
    if prob.transport_bc == "periodic":
        L = prob.xb - prob.xa
        xi = prob.xa + np.mod(xi - prob.xa, L)
    return prob.u0(xi)


def transport_symbol(scheme: str, lam: float, th: np.ndarray) -> np.ndarray:
    """Facteur d'amplification G(θ) des schémas pour u_t + a u_x = 0 (λ = aΔt/h signé)."""
    if scheme == "upwind":
        lp, lm = max(lam, 0.0), min(lam, 0.0)
        return 1 - lp * (1 - np.exp(-1j * th)) - lm * (np.exp(1j * th) - 1)
    if scheme == "ftcs":
        return 1 - 1j * lam * np.sin(th)
    if scheme == "lax_friedrichs":
        return np.cos(th) - 1j * lam * np.sin(th)
    if scheme == "lax_wendroff":
        return 1 - 1j * lam * np.sin(th) - 2 * lam ** 2 * np.sin(th / 2) ** 2
    if scheme == "btcs":
        return 1.0 / (1 + 1j * lam * np.sin(th))
    raise ValueError(scheme)


def solve_transport(prob: Problem, cfg: dict) -> dict[str, Any]:
    t0 = time.perf_counter()
    nx = int(cfg["mesh"]["nx"])
    a = float(prob.a or 1.0)
    scheme = cfg["time"]["transport_scheme"]
    periodic = prob.transport_bc == "periodic"
    L = prob.xb - prob.xa
    h = L / nx
    x = prob.xa + h * np.arange(nx) if periodic else np.linspace(prob.xa, prob.xb, nx + 1)
    plan = time_step_plan(prob, cfg, h, len(x))
    dt, nt = plan["dt"], plan["nt"]
    lam = a * dt / h
    U = prob.u0(x)
    inflow_left = a > 0
    if not periodic:
        xin = prob.xa if inflow_left else prob.xb
        gin = (lambda t: float(prob.exact(np.array([xin]), None, t)[0])) if prob.exact is not None else \
            (lambda t: float(prob.u0(np.array([xin - a * t]))[0]))
        U[0 if inflow_left else -1] = gin(0.0)
    lu = None
    if scheme == "btcs":
        nn = len(x)
        main = np.ones(nn)
        up = np.full(nn - 1, lam / 2)
        lo = np.full(nn - 1, -lam / 2)
        A = sp.diags([lo, main, up], [-1, 0, 1], format="lil")
        if periodic:
            A[0, nn - 1] = -lam / 2
            A[nn - 1, 0] = lam / 2
        else:
            if inflow_left:
                A[0, :] = 0
                A[0, 0] = 1
                A[nn - 1, nn - 2] = -lam
                A[nn - 1, nn - 1] = 1 + lam
            else:
                A[nn - 1, :] = 0
                A[nn - 1, nn - 1] = 1
                A[0, 0] = 1 - lam
                A[0, 1] = lam
        lu = spla.splu(A.tocsc())
    nframes = _frames_budget(len(x), cfg["time"]["frames"])
    fsteps = set(_frame_steps(nt, nframes).tolist())
    frames, ftimes = [U.astype(np.float32).copy()], [0.0]
    U0 = U.copy()
    scale = _blowup_scale(U) * LIMITS["blowup_factor"]
    blowup = None
    h_t, h_norm = [0.0], [float(np.sqrt(h * np.sum(U ** 2)))]
    hist_every = max(1, nt // 4000)
    for n in range(nt):
        t1 = (n + 1) * dt
        if scheme == "btcs":
            rhs = U.copy()
            if not periodic:
                rhs[0 if inflow_left else -1] = gin(t1)
            Unew = lu.solve(rhs)
        else:
            if periodic:
                Up, Um = np.roll(U, -1), np.roll(U, 1)
            else:
                Up = np.concatenate([U[1:], U[-1:]])
                Um = np.concatenate([U[:1], U[:-1]])
            if scheme == "upwind":
                Unew = U - max(lam, 0.0) * (U - Um) - min(lam, 0.0) * (Up - U)
            elif scheme == "ftcs":
                Unew = U - 0.5 * lam * (Up - Um)
            elif scheme == "lax_friedrichs":
                Unew = 0.5 * (Up + Um) - 0.5 * lam * (Up - Um)
            else:
                Unew = U - 0.5 * lam * (Up - Um) + 0.5 * lam ** 2 * (Up - 2 * U + Um)
            if not periodic:
                if inflow_left:
                    Unew[0] = gin(t1)
                    Unew[-1] = U[-1] - lam * (U[-1] - U[-2])  # bord de sortie : décentré amont
                else:
                    Unew[-1] = gin(t1)
                    Unew[0] = U[0] - lam * (U[1] - U[0])
        U = Unew
        if (n + 1) % hist_every == 0 or n + 1 == nt:
            h_t.append(t1)
            h_norm.append(float(np.sqrt(h * np.sum(U ** 2))))
        umax = float(np.max(np.abs(U)))
        if not math.isfinite(umax) or umax > scale:
            blowup = {"step": n + 1, "time": t1, "value": umax}
            frames.append(np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            ftimes.append(t1)
            break
        if (n + 1) in fsteps:
            frames.append(U.astype(np.float32).copy())
            ftimes.append(t1)
    t_end = (blowup["time"] if blowup else nt * dt)
    Uex = transport_exact(prob, x, t_end)
    w = np.full(len(x), h)
    if not periodic:
        w[[0, -1]] = h / 2
    e = U - Uex
    errs = {"l2": float(np.sqrt(np.sum(w * e ** 2))), "max": float(np.max(np.abs(e))),
            "l1": float(np.sum(w * np.abs(e)))}
    ne = float(np.sqrt(np.sum(w * Uex ** 2))) or 1.0
    errs.update(l2_rel=errs["l2"] / ne, max_rel=errs["max"] / (float(np.max(np.abs(Uex))) or 1.0),
                l1_rel=errs["l1"] / (float(np.sum(w * np.abs(Uex))) or 1.0))
    # vérification de von Neumann sur grille périodique (modes de Fourier = vecteurs propres)
    vn = None
    if periodic and not blowup and len(x) >= 8:
        F0, FN = np.fft.fft(U0), np.fft.fft(U)
        kk = int(np.argmax(np.abs(F0[1:len(x) // 2])) + 1)
        if abs(F0[kk]) > 1e-8 * (np.abs(F0).max() + 1e-300):
            th = 2 * np.pi * kk / len(x)
            G = transport_symbol(scheme, lam, np.array([th]))[0]
            meas = FN[kk] / F0[kk]
            vn = {"k": kk, "theta": th, "G_abs": float(abs(G)), "pred_amp": float(abs(G) ** nt),
                  "meas_amp": float(abs(meas)), "pred_phase": float(np.angle(G ** nt)),
                  "meas_phase": float(np.angle(meas)),
                  "rel_diff": float(abs(abs(meas) - abs(G) ** nt) / max(abs(G) ** nt, 1e-300)),
                  "exact_phase": float(-lam * th * nt)}
    hist_err = []
    for tt, fr in zip(ftimes, frames):
        ex = transport_exact(prob, x, tt)
        hist_err.append((tt, float(np.sqrt(np.sum(w * (fr - ex) ** 2))), float(np.max(np.abs(fr - ex)))))
    exact_frames = np.array([transport_exact(prob, x, tt) for tt in ftimes], dtype=np.float32)
    return {
        "ok": True, "family": prob.family, "cls": prob.cls, "method": "fdm", "dim": 1, "time_dependent": True,
        "time_order": 1, "nodes": x[:, None], "elems": None, "grid": {"xs": x, "ys": np.array([0.0])},
        "U": U, "Uex": Uex, "times": np.array(ftimes), "frames": np.array(frames),
        "exact_frames": exact_frames, "hist": {"t": np.array(h_t), "norm": np.array(h_norm),
                                               "err": np.array(hist_err)},
        "errors": errs, "dofs": len(x), "n_nodes": len(x), "n_elems": 0, "h": h, "hmin": h,
        "dt": dt, "nt": nt, "numbers": dict(plan, λ=lam), "scheme": scheme, "periodic": periodic,
        "blowup": blowup, "vonneumann_check": vn, "warnings": list(prob.warnings),
        "timings": {"total": time.perf_counter() - t0, "assembly": 0.0, "time_loop": time.perf_counter() - t0},
        "memory": int(U.nbytes * (len(frames) + 4)), "label": prob.describe(),
    }


# =============================================================================
# 9a. NORMES D'ERREUR
# =============================================================================
def compute_errors(prob: Problem, sysm: DiscreteSystem, U: np.ndarray, t: float) -> dict[str, float] | None:
    """Erreurs e = u − u_h : normes continues (FEM, quadrature d'ordre élevé) ou discrètes (FDM)."""
    if prob.exact is None:
        return None
    nodes = sysm.nodes
    ue = _eval_xy(prob.exact, nodes, t)
    e = U - ue
    out: dict[str, float] = {}
    out["max_nodal"] = float(np.max(np.abs(e)))
    out["max_nodal_rel"] = out["max_nodal"] / (float(np.max(np.abs(ue))) or 1.0)
    W = sysm.W
    out["l2_discrete"] = float(np.sqrt(np.sum(W * e ** 2)))
    nue = float(np.sqrt(np.sum(W * ue ** 2))) or 1.0
    out["l2_discrete_rel"] = out["l2_discrete"] / nue
    if sysm.method == "fem":
        mesh = sysm.mesh
        geo = p1_geometry(mesh)
        bary, w = (segment_rule(5) if mesh.dim == 1 else triangle_rule(5))
        Xq = quad_points(mesh, geo, bary)
        uq = _eval_xy(prob.exact, Xq, t)
        Ue = U[mesh.elems]
        uhq = np.einsum("qi,ei->eq", bary, Ue)
        meas = geo["meas"]
        l2 = float(np.sqrt(np.sum(meas[:, None] * w[None, :] * (uq - uhq) ** 2)))
        nl2 = float(np.sqrt(np.sum(meas[:, None] * w[None, :] * uq ** 2))) or 1.0
        guh = np.einsum("eid,ei->ed", geo["grad"], Ue)
        gx = _eval_xy(prob.exact_dx, Xq, t)
        diff2 = (gx - guh[:, None, 0]) ** 2
        g2 = gx ** 2
        if mesh.dim == 2:
            gy = _eval_xy(prob.exact_dy, Xq, t)
            diff2 = diff2 + (gy - guh[:, None, 1]) ** 2
            g2 = g2 + gy ** 2
        h1 = float(np.sqrt(np.sum(meas[:, None] * w[None, :] * diff2)))
        nh1 = float(np.sqrt(np.sum(meas[:, None] * w[None, :] * g2))) or 1.0
        out.update(l2=l2, l2_rel=l2 / nl2, h1=h1, h1_rel=h1 / nh1,
                   max_quad=float(np.max(np.abs(uq - uhq))), norm_l2_exact=nl2, norm_h1_exact=nh1)
        out["max"] = max(out["max_nodal"], out["max_quad"])
    else:
        g = sysm.grid
        I, J, nx, ny = g["I"], g["J"], g["nx"], g["ny"]
        nxp = nx + 1
        wx = np.full(nxp, g["hx"])
        wx[[0, -1]] = g["hx"] / 2
        if sysm.dim == 2:
            wy = np.full(ny + 1, g["hy"])
            wy[[0, -1]] = g["hy"] / 2
        else:
            wy = np.array([1.0])
        E2 = e.reshape(len(wy), nxp)
        UE2 = ue.reshape(len(wy), nxp)
        sx = np.sum(((E2[:, 1:] - E2[:, :-1]) / g["hx"]) ** 2 * g["hx"] * wy[:, None])
        nx2 = np.sum(((UE2[:, 1:] - UE2[:, :-1]) / g["hx"]) ** 2 * g["hx"] * wy[:, None])
        if sysm.dim == 2:
            sx += np.sum(((E2[1:, :] - E2[:-1, :]) / g["hy"]) ** 2 * g["hy"] * wx[None, :])
            nx2 += np.sum(((UE2[1:, :] - UE2[:-1, :]) / g["hy"]) ** 2 * g["hy"] * wx[None, :])
        out.update(l2=out["l2_discrete"], l2_rel=out["l2_discrete_rel"], h1=float(np.sqrt(sx)),
                   h1_rel=float(np.sqrt(sx)) / (float(np.sqrt(nx2)) or 1.0), max=out["max_nodal"])
        _ = (I, J)
    return out


# =============================================================================
# 8b. PILOTE DE RÉSOLUTION
# =============================================================================
def run_solve(cfg_in: dict) -> dict[str, Any]:
    """Résolution complète d'une configuration : maillage, assemblage, solveur, erreurs."""
    t_start = time.perf_counter()
    prob = Problem(cfg_in)
    if not prob.ok:
        return {"ok": False, "errors": prob.errors, "warnings": prob.warnings}
    cfg = prob.cfg
    try:
        if prob.cls == "transport":
            res = solve_transport(prob, cfg)
            res["cfg"] = cfg
            res["cfg_hash"] = config_hash(cfg)
            res["exact_available"] = True
            return res
        sysm = build_system(prob, cfg)
        h, hmin = mesh_size(sysm), mesh_hmin(sysm)
        res: dict[str, Any] = {
            "ok": True, "family": prob.family, "cls": prob.cls, "method": prob.method, "dim": prob.dim,
            "time_dependent": prob.time_dependent, "time_order": prob.time_order, "nodes": sysm.nodes,
            "elems": sysm.mesh.elems if sysm.mesh is not None else None,
            "grid": ({"xs": sysm.grid["xs"], "ys": sysm.grid["ys"]} if sysm.grid is not None else None),
            "dofs": int(len(sysm.free)), "n_nodes": int(sysm.n),
            "n_elems": int(sysm.mesh.n_elems) if sysm.mesh is not None else 0,
            "h": h, "hmin": hmin, "warnings": list(prob.warnings), "label": prob.describe(),
            "matrix": {"n": int(len(sysm.free)), "nnz": int(sysm.K[sysm.free][:, sysm.free].nnz)},
            "timings": {"assembly": sysm.info["assembly_time"]}, "exact_available": prob.exact is not None,
            "kernel": sysm.kernel,
        }
        if prob.exact is not None:
            chk = prob.exact_residuals() or {}
            bad = {k: v for k, v in chk.items() if not (v <= 1e-6)}
            if bad:
                res["warnings"].append("La solution exacte fournie ne vérifie pas toutes les équations ("
                                       + ", ".join(f"{k} : résidu {v:.1e}" for k, v in bad.items())
                                       + ") : l'« erreur » affichée ne mesure pas l'erreur de discrétisation.")
        if prob.cls == "elliptic" and not sysm.kernel:
            lm = lax_milgram_analysis(prob)
            if not lm.get("applicable"):
                res["warnings"].append("Hypothèses de Lax–Milgram non établies (voir « Existence et unicité ») : le "
                                       "système discret peut être résolu s'il est inversible, mais ni l'existence pour "
                                       "le problème continu ni la stabilité uniforme en h ne sont garanties.")
        if not prob.time_dependent:
            U, info = solve_stationary(sysm, cfg["solver"])
            res.update(U=U, solver=info, times=None, frames=None)
            res["timings"].update(solve=info.get("time", 0.0), factorization=info.get("factor_time", 0.0))
            res["matrix"]["symmetric"] = info.get("symmetric")
            res["errors"] = compute_errors(prob, sysm, U, 0.0)
            res["Uex"] = _eval_xy(prob.exact, sysm.nodes, 0.0) if prob.exact is not None else None
            if sysm.kernel:
                comp = neumann_compatibility(prob)
                res["compatibility"] = comp
                if comp["relative"] > 1e-8:
                    res["warnings"].append("Condition de compatibilité violée : la solution affichée est celle du "
                                           "problème projeté (le problème continu n'a pas de solution).")
        else:
            plan = time_step_plan(prob, cfg, hmin, len(sysm.free))
            dt, nt = plan["dt"], plan["nt"]
            res.update(dt=dt, nt=nt, numbers=plan)
            if prob.time_order == 1:
                out = solve_theta(prob, sysm, dt, nt, cfg["time"]["theta"], cfg["time"]["frames"])
                res["scheme"] = f"θ-schéma (ϑ = {cfg['time']['theta']:g})"
                res["theta"] = cfg["time"]["theta"]
                res["hist"] = {"t": out["hist_t"], "norm": out["hist_norm"]}
            else:
                beta = 0.0 if cfg["time"]["wave_scheme"] == "leapfrog" else 0.25
                out = solve_wave(prob, sysm, dt, nt, beta, cfg["time"]["frames"])
                res["scheme"] = WAVE_SCHEMES[cfg["time"]["wave_scheme"]]
                res["beta"] = beta
                res["hist"] = {"t": out["hist_t"], "energy": out["hist_energy"]}
            res.update(U=out["U"], frames=out["frames"], times=out["times"], blowup=out["blowup"],
                       factor_info=out["factor_info"], steps_done=out["steps_done"])
            res["warnings"] += out["warnings"]
            res["timings"].update(out["timings"])
            t_end = out["times"][-1]
            res["t_final"] = float(t_end)
            if prob.exact is not None:
                ex_frames = np.array([_eval_xy(prob.exact, sysm.nodes, tt) for tt in out["times"]], dtype=np.float32)
                res["exact_frames"] = ex_frames
                errs_t = []
                for tt, fr, ex in zip(out["times"], out["frames"], ex_frames):
                    ee = fr.astype(float) - ex
                    errs_t.append((tt, float(np.sqrt(np.sum(sysm.W * ee ** 2))), float(np.max(np.abs(ee)))))
                res["hist"]["err"] = np.array(errs_t)
                res["errors"] = compute_errors(prob, sysm, out["U"], t_end)
                res["Uex"] = _eval_xy(prob.exact, sysm.nodes, t_end)
            else:
                res["errors"] = None
                res["Uex"] = None
        res["memory"] = sysm.memory_bytes() + (res["frames"].nbytes if res.get("frames") is not None else 0)
        res["timings"]["total"] = time.perf_counter() - t_start
        res["cfg"] = cfg
        res["cfg_hash"] = config_hash(cfg)
        return res
    except ConfigError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": prob.warnings}
    except MemoryError:
        return {"ok": False, "errors": ["Mémoire insuffisante : réduisez la taille du maillage."],
                "warnings": prob.warnings}
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return {"ok": False, "errors": [f"Échec numérique : {exc}"], "warnings": prob.warnings}


# =============================================================================
# 9b. STABILITÉ : VON NEUMANN, ANALYSE SPECTRALE, PERTURBATIONS
# =============================================================================
def theta_R(z: Any, vt: float) -> Any:
    """Fonction d'amplification du θ-schéma pour U' = −μU, z = Δt μ : R(z) = (1 − (1−ϑ)z)/(1 + ϑz)."""
    return (1 - (1 - vt) * z) / (1 + vt * z)


def _adv_symbol(lam: float, th: np.ndarray, adv: str) -> Any:
    if lam == 0:
        return 0.0
    if adv == "upwind":
        return abs(lam) * (1 - np.cos(th)) + 1j * lam * np.sin(th)
    return 1j * lam * np.sin(th)


def parabolic_symbol(th_x: np.ndarray, rx: float, lamx: float = 0.0, cdt: float = 0.0, vt: float = 0.0,
                     adv: str = "centered", th_y: np.ndarray | None = None, ry: float = 0.0,
                     lamy: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Symbole spatial z(θ) (multiplié par Δt) et facteur d'amplification G(θ) du θ-schéma DF."""
    z = 4 * rx * np.sin(th_x / 2) ** 2 + _adv_symbol(lamx, th_x, adv) + cdt
    if th_y is not None:
        z = z + 4 * ry * np.sin(th_y / 2) ** 2 + _adv_symbol(lamy, th_y, adv)
    z = np.asarray(z, dtype=complex)
    return theta_R(z, vt), z


def fem1d_symbol(th: np.ndarray, r: float, vt: float, lumped: bool) -> tuple[np.ndarray, np.ndarray]:
    """Analyse de Fourier de P1 sur maillage 1D uniforme : λ(θ)Δt = 12 r sin²(θ/2)/(2+cos θ) (masse cohérente)."""
    s2 = np.sin(th / 2) ** 2
    z = 4 * r * s2 if lumped else 12 * r * s2 / (2 + np.cos(th))
    z = np.asarray(z, dtype=complex)
    return theta_R(z, vt), z


def wave_roots(w: Any, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Racines de (1+βw)G² − (2 − (1−2β)w)G + (1+βw) = 0, w = Δt²·(symbole de −c²Δ)."""
    w = np.asarray(w, dtype=complex)
    a = 1 + beta * w
    b = -(2 - (1 - 2 * beta) * w)
    disc = np.sqrt(b * b - 4 * a * a)
    return (-b + disc) / (2 * a), (-b - disc) / (2 * a)


def _grid_steps(prob: Problem, res: dict) -> tuple[float, float]:
    if prob.method == "fdm" or prob.cls == "transport":
        g = res.get("grid") or {}
        xs = np.asarray(g.get("xs", [0, 1]))
        ys = np.asarray(g.get("ys", [0]))
        hx = float(xs[1] - xs[0]) if len(xs) > 1 else res.get("h", 1.0)
        hy = float(ys[1] - ys[0]) if len(ys) > 1 else hx
        return hx, hy
    return res.get("hmin", res.get("h", 1.0)), res.get("hmin", res.get("h", 1.0))


def stability_report(prob: Problem, res: dict) -> dict[str, Any]:
    """Critères de stabilité (von Neumann, coefficients constants) évalués avec les paramètres réels."""
    cfg = prob.cfg
    out: dict[str, Any] = {"criteria": [], "framework": "", "vn_valid": [], "notes": []}
    if not prob.time_dependent:
        out["framework"] = "stationnaire"
        out["notes"].append("Problème stationnaire : pas de stabilité temporelle. La « stabilité » est ici "
                            "l'inversibilité uniforme de la matrice (‖A_h⁻¹‖ borné), garantie par la coercivité "
                            "(EF) ou par le principe du maximum discret (DF).")
        return out
    dt = float(res["dt"])
    hx, hy = _grid_steps(prob, res)
    th = np.linspace(-np.pi, np.pi, 2001)
    kmin, kmax, kex = prob.coefficient_range("k") if prob.cls != "transport" else (0, 0, True)
    frozen = not kex
    out["frozen"] = frozen
    out["vn_valid"] = [
        ("Linéarité", True), ("Coefficients constants", kex and (prob.cls == "transport" or prob.c.is_constant)),
        ("Grille uniforme", prob.method == "fdm" or prob.cls == "transport"),
        ("Conditions périodiques (ou frontière compatible)", prob.cls == "transport" and prob.transport_bc == "periodic"),
    ]
    if prob.cls == "parabolic":
        vt = float(cfg["time"]["theta"])
        rx = kmax * dt / hx ** 2
        ry = kmax * dt / hy ** 2 if prob.dim == 2 else 0.0
        lamx = prob.bx * dt / hx
        lamy = prob.by * dt / hy if prob.dim == 2 else 0.0
        cmin, cmax, _ = prob.coefficient_range("c")
        cdt = max(cmin, 0.0) * dt
        adv = cfg.get("adv_scheme", "centered")
        out.update(r=rx, ry=ry, lam=lamx, lamy=lamy, theta=vt, cdt=cmax * dt)
        if prob.method == "fdm":
            if prob.dim == 1:
                G, z = parabolic_symbol(th, rx, lamx, cdt, vt, adv)
            else:
                tx, ty = np.meshgrid(np.linspace(-np.pi, np.pi, 241), np.linspace(-np.pi, np.pi, 241))
                G, z = parabolic_symbol(tx, rx, lamx, cdt, vt, adv, ty, ry, lamy)
            out["max_G"] = float(np.max(np.abs(G)))
            out["max_G_principal"] = float(np.max(np.abs(
                parabolic_symbol(th, rx, lamx, 0.0, vt, adv)[0] if prob.dim == 1 else
                parabolic_symbol(tx, rx, lamx, 0.0, vt, adv, ty, ry, lamy)[0])))
        else:
            G, z = fem1d_symbol(th, rx, vt, bool(cfg["time"]["lumped"]))
            out["max_G_fem1d"] = float(np.max(np.abs(G)))
        out["framework"] = "θ-schéma pour U' + A U = F"
        rsum = rx + ry
        if vt >= 0.5:
            out["criteria"].append({"name": "θ-schéma avec ϑ ≥ 1/2", "latex": r"\vartheta\ge\tfrac12\ \Rightarrow\ |G(\theta)|\le 1\ \ \forall r",
                                    "value": f"ϑ = {vt:g}", "ok": True, "kind": "inconditionnellement stable (A-stable)"})
        else:
            lim = 1.0 / (2 * (1 - 2 * vt))
            if prob.method == "fdm":
                if prob.dim == 1:
                    name = "Condition de von Neumann (DF 1D)"
                    lx = rf"r(1-2\vartheta)\le\tfrac12\ \Leftrightarrow\ r = \frac{{k\Delta t}}{{h^2}}\le {ltx_num(lim)}"
                    if vt == 0:
                        lx = r"r = \frac{k\,\Delta t}{h^2}\ \le\ \frac12"
                    out["criteria"].append({"name": name, "latex": lx, "value": f"r = {fmt(rx)}", "ok": rx <= lim + 1e-12,
                                            "kind": "nécessaire et suffisante (coefficients constants)" + (" — coefficients gelés à k_max" if frozen else "")})
                else:
                    lx = rf"(r_x + r_y)(1-2\vartheta)\le\tfrac12\ \Leftrightarrow\ r_x + r_y\le {ltx_num(lim)}"
                    out["criteria"].append({"name": "Condition de von Neumann (DF 2D)", "latex": lx,
                                            "value": f"rₓ + r_y = {fmt(rsum)}", "ok": rsum <= lim + 1e-12,
                                            "kind": "nécessaire et suffisante (coefficients constants)"})
            else:
                lim_fem = lim if cfg["time"]["lumped"] else lim / 3.0
                out["criteria"].append({"name": "Fourier P1 1D uniforme (indicatif)",
                                        "latex": (r"\lambda_{\max}(M^{-1}K) = \frac{12k}{h^2}\ \Rightarrow\ r\le\frac{1}{6(1-2\vartheta)}"
                                                  if not cfg["time"]["lumped"] else
                                                  r"\lambda_{\max}(M_L^{-1}K) = \frac{4k}{h^2}\ \Rightarrow\ r\le\frac{1}{2(1-2\vartheta)}"),
                                        "value": f"r = {fmt(rx)}", "ok": rx <= lim_fem + 1e-12,
                                        "kind": "indicatif (maillage uniforme) — le verdict exact est donné par l'analyse spectrale"})
            if (prob.bx or prob.by) and prob.method == "fdm" and vt == 0 and prob.dim == 1:
                if adv == "centered":
                    out["criteria"].append({"name": "Advection centrée explicite", "latex": r"r\le\tfrac12\ \ \text{et}\ \ \lambda^2\le 2r",
                                            "value": f"λ² = {fmt(lamx ** 2)}, 2r = {fmt(2 * rx)}",
                                            "ok": rx <= 0.5 + 1e-12 and lamx ** 2 <= 2 * rx + 1e-12, "kind": "nécessaire et suffisante"})
                else:
                    out["criteria"].append({"name": "Advection décentrée explicite", "latex": r"2r + |\lambda|\le 1",
                                            "value": f"2r + |λ| = {fmt(2 * rx + abs(lamx))}",
                                            "ok": 2 * rx + abs(lamx) <= 1 + 1e-12, "kind": "nécessaire et suffisante"})
        if prob.bx or prob.by:
            pe = math.hypot(prob.bx, prob.by) * min(hx, hy) / (2 * kmin)
            out["criteria"].append({"name": "Nombre de Péclet de maille", "latex": r"\mathrm{Pe}_h = \frac{|\mathbf b|\,h}{2k}\le 1",
                                    "value": f"Pe_h = {fmt(pe)}", "ok": pe <= 1 + 1e-12,
                                    "kind": "précision / monotonie (pas une condition de stabilité) : au-delà, le schéma "
                                            "centré (ou Galerkin) peut produire des oscillations parasites"})
            out["Pe_h"] = pe
        if cmax > 0 and vt < 0.5:
            out["notes"].append(f"Réaction c ≤ {fmt(cmax)} : terme borné qui ne modifie pas la stabilité au sens de "
                                "Lax–Richtmyer (Δt → 0), mais qui s'ajoute au symbole pour la stabilité absolue au Δt choisi.")
    elif prob.cls == "hyperbolic":
        beta = 0.0 if cfg["time"]["wave_scheme"] == "leapfrog" else 0.25
        c = prob.cw or 1.0
        sx, sy = c * dt / hx, (c * dt / hy if prob.dim == 2 else 0.0)
        out.update(sigma=sx, sigma_y=sy, beta=beta)
        w = 4 * sx ** 2 * np.sin(th / 2) ** 2
        if prob.dim == 2:
            w = 4 * sx ** 2 * np.sin(th / 2) ** 2 + 4 * sy ** 2  # pire cas θ_y = π
        if prob.method == "fem" and prob.dim == 1:
            s2 = np.sin(th / 2) ** 2
            w = (4 if cfg["time"]["lumped"] else 12 / (2 + np.cos(th))) * sx ** 2 * s2
        g1, g2 = wave_roots(w, beta)
        out["max_G"] = float(max(np.max(np.abs(g1)), np.max(np.abs(g2))))
        out["framework"] = "schéma à trois niveaux (Newmark γ = 1/2)"
        if beta >= 0.25:
            out["criteria"].append({"name": "Newmark β = 1/4", "latex": r"\beta\ge\tfrac14\ \Rightarrow\ |G_\pm| = 1\ \ \forall\sigma",
                                    "value": f"σ = {fmt(sx)}", "ok": True, "kind": "inconditionnellement stable, conservatif"})
        elif prob.method == "fdm":
            if prob.dim == 1:
                out["criteria"].append({"name": "Condition CFL (DF 1D)", "latex": r"\sigma = \frac{c\,\Delta t}{h}\le 1",
                                        "value": f"σ = {fmt(sx)}", "ok": sx <= 1 + 1e-12,
                                        "kind": "nécessaire et suffisante (σ = 1 : schéma exact en 1D)"})
            else:
                out["criteria"].append({"name": "Condition CFL (DF 2D)", "latex": r"c\,\Delta t\sqrt{\frac1{h_x^2}+\frac1{h_y^2}}\le 1",
                                        "value": f"σ_eff = {fmt(math.hypot(sx, sy))}", "ok": math.hypot(sx, sy) <= 1 + 1e-12,
                                        "kind": "nécessaire et suffisante"})
        else:
            lim = 1 / math.sqrt(3) if not cfg["time"]["lumped"] else 1.0
            out["criteria"].append({"name": "Fourier P1 1D uniforme (indicatif)",
                                    "latex": r"\sigma\le\frac1{\sqrt3}\ \text{(masse cohérente)},\quad \sigma\le 1\ \text{(masse condensée)}",
                                    "value": f"σ = {fmt(sx)}", "ok": sx <= lim + 1e-12,
                                    "kind": "indicatif — le verdict exact est donné par l'analyse spectrale"})
        out["criteria"].append({"name": "Domaine de dépendance (CFL)",
                                "latex": r"\text{le domaine de dépendance numérique doit contenir le domaine physique}",
                                "value": f"c Δt/h = {fmt(sx)}", "ok": sx <= 1 + 1e-12 or beta >= 0.25,
                                "kind": "condition nécessaire de Courant–Friedrichs–Lewy pour un schéma explicite"})
    else:
        lam = float(res["numbers"]["λ"]) if "numbers" in res else 0.0
        scheme = cfg["time"]["transport_scheme"]
        G = transport_symbol(scheme, lam, th)
        out["max_G"] = float(np.max(np.abs(G)))
        out["lam"] = lam
        out["framework"] = TRANSPORT_SCHEMES[scheme]
        crit = {
            "upwind": (r"0\le\lambda\le 1\ (a>0),\ \ |\lambda|\le 1", abs(lam) <= 1 + 1e-12, "nécessaire et suffisante ; schéma monotone d'ordre 1"),
            "lax_friedrichs": (r"|\lambda|\le 1", abs(lam) <= 1 + 1e-12, "nécessaire et suffisante ; ordre 1, très dissipatif"),
            "lax_wendroff": (r"|\lambda|\le 1", abs(lam) <= 1 + 1e-12, "nécessaire et suffisante ; ordre 2, dispersif"),
            "ftcs": (r"|G|^2 = 1+\lambda^2\sin^2\theta > 1\ \ \forall\lambda\ne 0", False,
                     "instable au sens de von Neumann pour tout λ ≠ 0 (stable seulement si Δt = O(h²))"),
            "btcs": (r"|G|^2 = \frac{1}{1+\lambda^2\sin^2\theta}\le 1", True, "inconditionnellement stable, dissipatif"),
        }[scheme]
        out["criteria"].append({"name": f"Von Neumann — {TRANSPORT_SCHEMES[scheme]}", "latex": crit[0],
                                "value": f"λ = a Δt / h = {fmt(lam)}", "ok": crit[1], "kind": crit[2]})
        out["criteria"].append({"name": "Condition CFL", "latex": r"|\lambda| = \frac{|a|\,\Delta t}{h}\le 1",
                                "value": f"|λ| = {fmt(abs(lam))}", "ok": abs(lam) <= 1 + 1e-12 or scheme == "btcs",
                                "kind": "nécessaire pour tout schéma explicite à 3 points (domaine de dépendance)"})
    return out


def spectral_analysis(prob: Problem, cfg: dict, dt: float | None = None, lumped_compare: bool = True) -> dict[str, Any]:
    """Valeurs propres généralisées K z = λ M z du problème semi-discret (nœuds libres)."""
    sysm = build_system(prob, cfg)
    Fr = sysm.free
    K = sysm.K[Fr][:, Fr].tocsc()
    M = sysm.M[Fr][:, Fr].tocsc()
    n = len(Fr)
    out: dict[str, Any] = {"n": n, "method": sysm.method}
    sym = is_symmetric(K) and is_symmetric(M)
    out["symmetric"] = sym
    t0 = time.perf_counter()
    v0 = np.ones(n) / math.sqrt(max(n, 1))
    if n == 0:
        return {"n": 0, "error": "aucune inconnue libre"}
    try:
        if sym and n <= LIMITS["dense_eig_max"]:
            ev = sla.eigh(K.toarray(), M.toarray(), eigvals_only=True)
            out["eigs"] = np.sort(ev)
            out["complete"] = True
        elif (not sym) and n <= LIMITS["dense_eig_nonsym_max"]:
            ev = sla.eig(K.toarray(), M.toarray(), right=False)
            out["eigs"] = ev[np.argsort(ev.real)]
            out["complete"] = True
        else:
            out["complete"] = False
            if sym:
                lmax = spla.eigsh(K, k=1, M=M, which="LA", v0=v0, return_eigenvectors=False)
                lmin = spla.eigsh(K, k=4, M=M, sigma=0.0, which="LM", v0=v0, return_eigenvectors=False)
                out["eigs"] = np.sort(np.concatenate([lmin, lmax]))
            else:
                big = spla.eigs(K, k=4, M=M, which="LM", v0=v0, return_eigenvectors=False)
                small = spla.eigs(K, k=4, M=M, sigma=0.0, which="LM", v0=v0, return_eigenvectors=False)
                ev = np.concatenate([small, big])
                out["eigs"] = ev[np.argsort(ev.real)]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"calcul du spectre impossible : {exc}"
        return out
    ev = out["eigs"]
    out["time"] = time.perf_counter() - t0
    out["lam_max"] = float(np.max(np.abs(ev)))
    out["lam_min"] = float(np.min(ev.real))
    if sym:
        out["cond_generalized"] = out["lam_max"] / out["lam_min"] if out["lam_min"] > 0 else math.inf
    # conditionnement de la matrice du système stationnaire
    try:
        if sym:
            if n <= LIMITS["dense_eig_max"]:
                kev = sla.eigh(K.toarray(), eigvals_only=True)
                out["cond_K"] = float(kev.max() / kev.min()) if kev.min() > 0 else math.inf
                out["K_min_eig"] = float(kev.min())
            else:
                kmax_ = spla.eigsh(K, k=1, which="LA", v0=v0, return_eigenvectors=False)[0]
                kmin_ = spla.eigsh(K, k=1, sigma=0.0, which="LM", v0=v0, return_eigenvectors=False)[0]
                out["cond_K"] = float(kmax_ / kmin_) if kmin_ > 0 else math.inf
                out["K_min_eig"] = float(kmin_)
    except Exception:  # noqa: BLE001
        pass
    if lumped_compare and sysm.method == "fem" and sym:
        try:
            Mc = sysm.M_cons[Fr][:, Fr].tocsc()
            ML = sp.diags(np.asarray(sysm.M_cons.sum(axis=1)).ravel()[Fr]).tocsc()
            out["lam_max_consistent"] = float(spla.eigsh(K, k=1, M=Mc, which="LA", v0=v0, return_eigenvectors=False)[0])
            out["lam_max_lumped"] = float(spla.eigsh(K, k=1, M=ML, which="LA", v0=v0, return_eigenvectors=False)[0])
        except Exception:  # noqa: BLE001
            pass
    if prob.time_dependent and dt:
        if prob.time_order == 1:
            vt = float(cfg["time"]["theta"])
            Rz = theta_R(dt * ev, vt)
            out["amplification"] = np.abs(Rz)
            out["rho"] = float(np.max(np.abs(Rz)))
            if sym and vt < 0.5:
                out["dt_max"] = 2.0 / ((1 - 2 * vt) * out["lam_max"])
            out["scheme"] = f"θ-schéma, ϑ = {vt:g}"
        else:
            beta = 0.0 if cfg["time"]["wave_scheme"] == "leapfrog" else 0.25
            g1, g2 = wave_roots(dt * dt * ev, beta)
            amp = np.maximum(np.abs(g1), np.abs(g2))
            out["amplification"] = amp
            out["rho"] = float(np.max(amp))
            if beta < 0.25:
                out["dt_max"] = 2.0 / math.sqrt((1 - 4 * beta) * out["lam_max"])
            out["scheme"] = WAVE_SCHEMES[cfg["time"]["wave_scheme"]]
        out["dt"] = dt
    # spectre continu de référence (Dirichlet homogène partout, coefficient constant)
    if all(prob.bc[s]["type"] == "dirichlet" for s in prob.sides) and prob.k.is_constant and prob.shape != "lshape" \
            and prob.c_expr == 0 and not (prob.bx or prob.by):
        kc = prob.k.const_value
        m_ = np.arange(1, 80)
        if prob.dim == 1:
            exact = kc * (np.pi * m_ / (prob.xb - prob.xa)) ** 2
        else:
            MM, NN = np.meshgrid(m_[:40], m_[:40])
            exact = np.sort((kc * np.pi ** 2 * ((MM / (prob.xb - prob.xa)) ** 2 + (NN / (prob.yb - prob.ya)) ** 2)).ravel())
        out["exact_spectrum"] = exact[: min(len(exact), n)]
    return out


def perturbation_experiment(kind: str, params: dict, n: int = 64, nsteps: int = 80, seed: int = 0) -> dict[str, Any]:
    """Évolution d'une perturbation aléatoire sur grille périodique avec le schéma réel (espace physique)."""
    rng = np.random.default_rng(seed)
    e0 = 1e-3 * (2 * rng.random(n) - 1)
    e = e0.copy()
    norms = [1.0]
    profiles = {0: e0.copy()}
    keep = {nsteps // 3, 2 * nsteps // 3, nsteps}
    I = sp.identity(n, format="csc")
    th = np.linspace(-np.pi, np.pi, 2001)

    def circ(coefs: dict[int, float]) -> sp.csc_matrix:
        A = sp.lil_matrix((n, n))
        for off, val in coefs.items():
            for j in range(n):
                A[j, (j + off) % n] += val
        return A.tocsc()

    if kind == "parabolic":
        r, lam, vt, adv = params.get("r", 0.4), params.get("lam", 0.0), params.get("theta", 0.0), params.get("adv", "centered")
        coefs = {-1: -r, 0: 2 * r, 1: -r}
        if lam:
            if adv == "upwind":
                if lam > 0:
                    coefs[0] += lam
                    coefs[-1] -= lam
                else:
                    coefs[1] += lam
                    coefs[0] -= lam
            else:
                coefs[1] += lam / 2
                coefs[-1] -= lam / 2
        A = circ(coefs)
        L, R = (I + vt * A).tocsc(), (I - (1 - vt) * A).tocsc()
        lu = spla.splu(L) if vt > 0 else None
        for k in range(1, nsteps + 1):
            e = R @ e
            if lu is not None:
                e = lu.solve(e)
            norms.append(float(np.linalg.norm(e) / np.linalg.norm(e0)))
            if k in keep:
                profiles[k] = e.copy()
        G, _ = parabolic_symbol(th, r, lam, 0.0, vt, adv)
        gmax = float(np.max(np.abs(G)))
    elif kind == "wave":
        s, beta = params.get("sigma", 0.9), params.get("beta", 0.0)
        Wm = circ({-1: -s * s, 0: 2 * s * s, 1: -s * s})
        L = (I + beta * Wm).tocsc()
        R1 = (2 * I - (1 - 2 * beta) * Wm).tocsc()
        lu = spla.splu(L)
        em, e = e0.copy(), e0.copy()
        for k in range(1, nsteps + 1):
            ep = lu.solve(R1 @ e - L @ em)
            em, e = e, ep
            norms.append(float(np.linalg.norm(e) / np.linalg.norm(e0)))
            if k in keep:
                profiles[k] = e.copy()
        g1, g2 = wave_roots(4 * s * s * np.sin(th / 2) ** 2, beta)
        gmax = float(max(np.max(np.abs(g1)), np.max(np.abs(g2))))
    else:
        scheme, lam = params.get("scheme", "upwind"), params.get("lam", 0.8)
        lu = None
        if scheme == "btcs":
            lu = spla.splu((I + circ({1: lam / 2, -1: -lam / 2})).tocsc())
        for k in range(1, nsteps + 1):
            Up, Um = np.roll(e, -1), np.roll(e, 1)
            if scheme == "upwind":
                e = e - max(lam, 0) * (e - Um) - min(lam, 0) * (Up - e)
            elif scheme == "ftcs":
                e = e - 0.5 * lam * (Up - Um)
            elif scheme == "lax_friedrichs":
                e = 0.5 * (Up + Um) - 0.5 * lam * (Up - Um)
            elif scheme == "lax_wendroff":
                e = e - 0.5 * lam * (Up - Um) + 0.5 * lam ** 2 * (Up - 2 * e + Um)
            else:
                e = lu.solve(e)
            norms.append(float(np.linalg.norm(e) / np.linalg.norm(e0)))
            if k in keep:
                profiles[k] = e.copy()
        gmax = float(np.max(np.abs(transport_symbol(scheme, lam, th))))
    steps = np.arange(nsteps + 1)
    with np.errstate(over="ignore"):
        pred = np.power(gmax, steps.astype(float))
    return {"steps": steps, "norms": np.array(norms), "pred": pred, "gmax": gmax, "profiles": profiles, "n": n}


# =============================================================================
# 9c. CONVERGENCE
# =============================================================================
def theoretical_orders(prob: Problem) -> dict[str, Any]:
    """Ordres théoriques attendus ET conditions sous lesquelles ils sont garantis."""
    smooth = prob.exact_expr is not None and is_smooth_expr(prob.exact_expr) and is_smooth_expr(prob.k_expr)
    cfg = prob.cfg
    out: dict[str, Any] = {"smooth": smooth}
    if prob.cls == "transport":
        sch = cfg["time"]["transport_scheme"]
        p = 2 if sch == "lax_wendroff" else 1
        out["orders"] = {"l2": p, "max": p}
        out["text"] = (f"Schéma consistant d'ordre {p} et stable sous CFL : convergence d'ordre {p} (Lax–Richtmyer) pour "
                       "une solution régulière. Pour une donnée discontinue, l'ordre chute (≈ 1/2 en norme L¹ pour un "
                       "schéma d'ordre 1, ≈ 2/3 pour Lax–Wendroff).")
        out["conditions"] = "u₀ ∈ C^{p+1} et |λ| ≤ 1 (FTCS : instable, pas de convergence)."
        return out
    if prob.method == "fdm":
        out["orders"] = {"l2": 2, "max": 2, "h1": 2}
        out["text"] = ("Schéma 3/5 points consistant d'ordre 2 (erreur de troncature h²/12 u⁽⁴⁾) et stable "
                       "(principe du maximum discret ou norme spectrale) : convergence d'ordre 2 en normes discrètes.")
        out["conditions"] = "u ∈ C⁴(Ω̄) (et k ∈ C³) ; traitement des bords de Neumann/Robin par demi-maille (ordre 2 global)."
    else:
        if prob.shape == "lshape":
            out["orders"] = {"l2": 4 / 3, "h1": 2 / 3, "max": 2 / 3}
            out["text"] = ("Coin rentrant d'angle ω = 3π/2 : u ~ r^(π/ω) = r^(2/3) n'appartient qu'à H^(1+2/3−ε)(Ω). "
                           "Le lemme de Céa donne seulement ‖u − u_h‖_H¹ = O(h^(2/3)) et Aubin–Nitsche O(h^(4/3)) en L².")
            out["conditions"] = "Ω non convexe : l'estimation O(h) en H¹ N'EST PAS garantie (u ∉ H²)."
        else:
            out["orders"] = {"l2": 2, "h1": 1, "max": 2}
            out["text"] = ("Lemme de Céa + interpolation P1 : ‖u − u_h‖_{H¹} ≤ (M/α) C h |u|_{H²} ; dualité "
                           "d'Aubin–Nitsche (Ω convexe ⇒ régularité H²) : ‖u − u_h‖_{L²} ≤ C h² |u|_{H²}.")
            out["conditions"] = "u ∈ H²(Ω), Ω convexe (rectangle ✓), famille de maillages régulière (angles minorés)."
    if prob.time_dependent:
        if prob.time_order == 1:
            vt = float(cfg["time"]["theta"])
            p_t = 2 if abs(vt - 0.5) < 1e-12 else 1
            out["time_order"] = p_t
            out["text"] += f" En temps : θ-schéma d'ordre {p_t} ⇒ erreur O(h² + Δt^{p_t})."
        else:
            out["time_order"] = 2
            out["text"] += " En temps : schéma centré / Newmark d'ordre 2 ⇒ erreur O(h² + Δt²)."
    if not smooth:
        out["conditions"] += (" ATTENTION : la solution exacte (ou k) contient des fonctions non régulières : "
                              "l'ordre théorique n'est pas garanti.")
    return out


def _level_config(cfg: dict, prob: Problem, level: int, n0: int, dt_strategy: str, dt0: float | None,
                  num0: float | None) -> tuple[dict, dict]:
    c = copy.deepcopy(cfg)
    info: dict[str, Any] = {}
    if prob.method == "fem" and prob.cls != "transport":
        c["mesh"]["nx"] = n0
        ratio = cfg["mesh"].get("ny", n0) / max(cfg["mesh"]["nx"], 1)
        c["mesh"]["ny"] = max(1, int(round(n0 * ratio)))
        c["mesh"]["refine"] = level
        if prob.shape == "lshape" and n0 % 2:
            c["mesh"]["nx"] = c["mesh"]["ny"] = n0 + 1
        info["n"] = n0 * 2 ** level
    else:
        n = n0 * 2 ** level
        c["mesh"]["nx"] = n
        ratio = cfg["mesh"].get("ny", n0) / max(cfg["mesh"]["nx"], 1)
        c["mesh"]["ny"] = max(1, int(round(n * ratio)))
        info["n"] = n
    if prob.time_dependent and dt0 is not None:
        factor = 0.5 ** level
        if dt_strategy == "dt_h":
            c["time"]["dt_mode"], c["time"]["dt"] = "dt", dt0 * factor
        elif dt_strategy == "dt_h2":
            c["time"]["dt_mode"], c["time"]["dt"] = "dt", dt0 * factor ** 2
        elif dt_strategy == "fixed_dt":
            c["time"]["dt_mode"], c["time"]["dt"] = "dt", dt0
        elif num0 is not None:  # "number" : r, σ ou λ constant d'un niveau à l'autre
            c["time"]["dt_mode"], c["time"]["number"] = "number", num0
    return c, info


def default_dt_strategy(prob: Problem) -> str:
    if not prob.time_dependent:
        return "none"
    if prob.cls in ("hyperbolic", "transport"):
        return "number"
    vt = float(prob.cfg["time"]["theta"])
    return "dt_h" if abs(vt - 0.5) < 1e-12 else "number"


DT_STRATEGY_LABELS = {
    "number": "Nombre de stabilité constant (r : Δt ∝ h² ; σ, λ : Δt ∝ h)",
    "dt_h": "Δt ∝ h",
    "dt_h2": "Δt ∝ h²",
    "fixed_dt": "Δt fixe",
    "none": "—",
}


def estimate_cost(prob: Problem, cfg: dict) -> tuple[int, int]:
    """(inconnues, pas de temps) estimés sans assembler."""
    mesh = cfg["mesh"]
    if prob.dim == 1:
        nn = mesh["nx"] * (2 ** mesh.get("refine", 0)) + 1
        h = (prob.xb - prob.xa) / (nn - 1)
    else:
        f = 2 ** mesh.get("refine", 0) if prob.method == "fem" else 1
        nx, ny = mesh["nx"] * f, mesh["ny"] * f
        nn = (nx + 1) * (ny + 1) + (nx * ny if (prob.method == "fem" and mesh.get("pattern") == "crossed") else 0)
        h = min((prob.xb - prob.xa) / nx, (prob.yb - prob.ya) / ny)
        if prob.method == "fem" and mesh.get("pattern") == "crossed":
            h /= 2
    nt = 0
    if prob.time_dependent:
        tm = cfg["time"]
        if tm["dt_mode"] == "dt":
            nt = int(math.ceil(prob.T / tm["dt"]))
        elif tm["dt_mode"] == "nt":
            nt = tm["nt"]
        else:
            if prob.cls == "parabolic":
                dt = tm["number"] * h ** 2 / prob.k_max()
            elif prob.cls == "hyperbolic":
                dt = tm["number"] * h / (prob.cw or 1)
            else:
                dt = tm["number"] * h / abs(prob.a or 1)
            nt = int(math.ceil(prob.T / dt))
    return int(nn), int(nt)


def convergence_study(cfg_in: dict, levels: int = 4, n0: int | None = None, dt_strategy: str | None = None) -> dict[str, Any]:
    """Étude de convergence par raffinement successif (h → h/2)."""
    prob = Problem(cfg_in)
    if not prob.ok:
        return {"ok": False, "errors": prob.errors}
    if prob.exact is None and prob.cls != "transport":
        return {"ok": False, "errors": ["Solution exacte inconnue : l'erreur ne peut pas être calculée. Fournissez une "
                                        "solution exacte (ou activez la solution manufacturée)."]}
    cfg = prob.cfg
    n0 = int(n0 or max(2, cfg["mesh"]["nx"] // 2 ** (levels - 1)))
    if prob.shape == "lshape" and n0 % 2:
        n0 += 1
    dt_strategy = dt_strategy or default_dt_strategy(prob)
    dt0 = None
    if prob.time_dependent:
        c0, _ = _level_config(cfg, prob, 0, n0, "number", None, None)
        p0 = Problem(c0)
        if prob.cls == "transport":
            h0 = (prob.xb - prob.xa) / n0
        else:
            s0 = build_system(p0, c0)
            h0 = mesh_hmin(s0)
        plan0 = time_step_plan(p0, c0, h0, 1)
        dt0 = plan0["dt"]
        num0 = plan0[plan0["number_name"]]
    else:
        num0 = None
    rows: list[dict[str, Any]] = []
    skipped = None
    for lev in range(levels):
        c, info = _level_config(cfg, prob, lev, n0, dt_strategy, dt0, num0)
        pl = Problem(c)
        nn, nt = estimate_cost(pl, c)
        if (prob.dim == 2 and nn > 70_000) or (prob.dim == 1 and nn > 20_000) or nt * nn > LIMITS["work_max"] / 2:
            skipped = f"niveau {lev} non calculé (coût estimé : {nn} nœuds × {max(nt, 1)} pas) — limite de sécurité"
            break
        t0 = time.perf_counter()
        r = run_solve(c)
        el = time.perf_counter() - t0
        if not r["ok"]:
            return {"ok": False, "errors": r["errors"], "rows": rows}
        e = r["errors"] or {}
        rows.append({"level": lev, "n": info["n"], "h": r["h"], "dofs": r["dofs"], "dt": r.get("dt"),
                     "nt": r.get("nt"), "time": el, "l2": e.get("l2"), "h1": e.get("h1"), "max": e.get("max"),
                     "l2_rel": e.get("l2_rel"), "blowup": r.get("blowup") is not None})
    if len(rows) < 2:
        return {"ok": False, "errors": ["Moins de deux niveaux calculés : étude impossible. " + (skipped or "")],
                "rows": rows}
    keys = [k for k in ("l2", "h1", "max") if rows[0].get(k) is not None]
    orders: dict[str, list[float | None]] = {k: [None] for k in keys}
    for k in keys:
        for i in range(1, len(rows)):
            e0, e1 = rows[i - 1][k], rows[i][k]
            h0_, h1_ = rows[i - 1]["h"], rows[i]["h"]
            orders[k].append(math.log(e0 / e1) / math.log(h0_ / h1_) if (e0 and e1 and e0 > 0 and e1 > 0) else None)
    slopes = {}
    for k in keys:
        hs = np.array([r_["h"] for r_ in rows])
        es = np.array([r_[k] for r_ in rows], dtype=float)
        mask = es > 0
        if mask.sum() >= 2:
            sel = np.nonzero(mask)[0][-3:]
            slopes[k] = float(np.polyfit(np.log(hs[sel]), np.log(es[sel]), 1)[0])
    return {"ok": True, "rows": rows, "orders": orders, "slopes": slopes, "keys": keys, "n0": n0,
            "dt_strategy": dt_strategy, "theory": theoretical_orders(prob), "skipped": skipped,
            "method": prob.method, "family": prob.family, "cfg_hash": config_hash(cfg)}


def temporal_convergence(cfg_in: dict, levels: int = 5, dt0: float | None = None) -> dict[str, Any]:
    """Convergence en temps à maillage fixe : référence = même discrétisation spatiale, Δt_ref = Δt_min/8."""
    prob = Problem(cfg_in)
    if not prob.ok or not prob.time_dependent or prob.cls == "transport":
        return {"ok": False, "errors": ["Étude temporelle disponible pour les problèmes paraboliques et les ondes."]}
    cfg = prob.cfg
    sysm = build_system(prob, cfg)
    if len(sysm.free) > 6000:
        return {"ok": False, "errors": ["Maillage trop fin pour l'étude temporelle (> 6000 inconnues)."]}
    conditional = (prob.time_order == 1 and float(cfg["time"]["theta"]) < 0.5) or \
        (prob.time_order == 2 and cfg["time"]["wave_scheme"] == "leapfrog")
    if dt0 is None:
        dt0 = time_step_plan(prob, cfg, mesh_hmin(sysm), len(sysm.free))["dt"] if conditional else prob.T / 8
    nt0 = max(1, int(math.ceil(prob.T / dt0 - 1e-9)))
    dts = [prob.T / (nt0 * 2 ** k) for k in range(levels)]
    dt_ref = dts[-1] / 8
    nt_ref = int(round(prob.T / dt_ref))
    if nt_ref * len(sysm.free) > LIMITS["work_max"]:
        return {"ok": False, "errors": ["Référence trop coûteuse : réduisez le maillage ou le nombre de niveaux."]}

    def integrate(dt: float, reference: bool = False) -> np.ndarray:
        nt = int(round(prob.T / dt))
        if prob.time_order == 1:
            vt = 0.5 if reference else float(cfg["time"]["theta"])
            return solve_theta(prob, sysm, prob.T / nt, nt, vt, 2)["U"]
        beta = 0.25 if reference else (0.0 if cfg["time"]["wave_scheme"] == "leapfrog" else 0.25)
        return solve_wave(prob, sysm, prob.T / nt, nt, beta, 2)["U"]

    Uref = integrate(dt_ref, reference=True)
    rows = []
    for dt in dts:
        t0 = time.perf_counter()
        try:
            U = integrate(dt)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "errors": [str(exc)]}
        err = float(np.sqrt(np.sum(sysm.W * (U - Uref) ** 2)))
        rows.append({"dt": prob.T / int(round(prob.T / dt)), "err": err, "time": time.perf_counter() - t0})
    orders = [None] + [math.log(rows[i - 1]["err"] / rows[i]["err"]) / math.log(rows[i - 1]["dt"] / rows[i]["dt"])
                       if rows[i]["err"] > 0 and rows[i - 1]["err"] > 0 else None for i in range(1, len(rows))]
    return {"ok": True, "rows": rows, "orders": orders, "dt_ref": dt_ref, "dofs": len(sysm.free),
            "reference": ("Crank–Nicolson" if prob.time_order == 1 else "Newmark β = 1/4") + f", Δt_ref = {dt_ref:.3e}"}


# =============================================================================
# 10. PROPRIÉTÉS MATRICIELLES, INTERPRÉTATION, RAPPORT
# =============================================================================
def matrix_properties(A: sp.spmatrix, spd_check: bool = True) -> dict[str, Any]:
    A = A.tocsr()
    n = A.shape[0]
    out: dict[str, Any] = {"n": n, "nnz": int(A.nnz), "density": A.nnz / max(n * n, 1)}
    out["symmetric"] = is_symmetric(A)
    diag = A.diagonal()
    out["diag_positive"] = bool(np.all(diag > 0))
    off = A - sp.diags(diag)
    off = off.tocsr()
    off.eliminate_zeros()
    out["offdiag_nonpositive"] = bool(off.nnz == 0 or off.data.max() <= 1e-14 * max(abs(diag).max(), 1e-300))
    rowsum_off = np.asarray(abs(off).sum(axis=1)).ravel()
    out["diag_dominant"] = bool(np.all(diag >= rowsum_off - 1e-12 * np.abs(diag)))
    out["strict_rows"] = int(np.sum(diag > rowsum_off + 1e-12 * np.abs(diag)))
    ncomp, _ = connected_components(abs(A) + abs(A.T), directed=False)
    out["irreducible"] = ncomp == 1
    out["m_matrix"] = bool(out["diag_positive"] and out["offdiag_nonpositive"] and out["diag_dominant"]
                           and out["irreducible"] and out["strict_rows"] > 0)
    rows_nnz = np.diff(A.indptr)
    out["max_row_nnz"] = int(rows_nnz.max()) if n else 0
    try:
        nz_rows, nz_cols = A.nonzero()
        out["bandwidth"] = int(np.max(np.abs(nz_rows - nz_cols))) if A.nnz else 0
    except Exception:  # noqa: BLE001
        out["bandwidth"] = None
    if spd_check and out["symmetric"] and n:
        try:
            if n <= LIMITS["dense_eig_max"]:
                ev = sla.eigh(A.toarray(), eigvals_only=True)
                out["lam_min"], out["lam_max"] = float(ev.min()), float(ev.max())
            else:
                v0 = np.ones(n) / math.sqrt(n)
                out["lam_max"] = float(spla.eigsh(A, k=1, which="LA", v0=v0, return_eigenvectors=False)[0])
                out["lam_min"] = float(spla.eigsh(A, k=1, sigma=0.0, which="LM", v0=v0, return_eigenvectors=False)[0])
            out["spd"] = out["lam_min"] > 0
            out["cond2"] = out["lam_max"] / out["lam_min"] if out["lam_min"] > 0 else math.inf
        except Exception:  # noqa: BLE001
            out["spd"] = None
    return out


def _decay_rate(hist_t: np.ndarray, hist_norm: np.ndarray) -> float | None:
    if hist_t is None or len(hist_t) < 6:
        return None
    t = np.asarray(hist_t)
    nrm = np.asarray(hist_norm)
    sel = (t >= t[-1] * 0.5) & (nrm > 1e-300)
    if sel.sum() < 3:
        return None
    return float(np.polyfit(t[sel], np.log(nrm[sel]), 1)[0])


def measured_growth_factor(res: dict) -> float | None:
    """Facteur d'amplification par pas mesuré sur la fin d'un calcul instable (historique ou images)."""
    dt = res.get("dt")
    if not dt:
        return None
    hist = res.get("hist") or {}
    if hist.get("norm") is not None and len(hist["norm"]) >= 6:
        t = np.asarray(hist["t"], dtype=float)
        nrm = np.asarray(hist["norm"], dtype=float)
        k = min(10, len(nrm) - 1)
        if nrm[-1] > 0 and nrm[-1 - k] > 0 and t[-1] > t[-1 - k]:
            return float((nrm[-1] / nrm[-1 - k]) ** (dt / (t[-1] - t[-1 - k])))
    fr, tm = res.get("frames"), res.get("times")
    if fr is not None and len(fr) >= 2:
        a, b = float(np.max(np.abs(fr[-2]))), float(np.max(np.abs(fr[-1])))
        steps = (float(tm[-1]) - float(tm[-2])) / dt
        if a > 0 and b > 0 and steps >= 1:
            return float((b / a) ** (1.0 / steps))
    return None


def build_report(prob: Problem, res: dict, stab: dict | None = None, conv: dict | None = None,
                 mprops: dict | None = None, spectral: dict | None = None) -> dict[str, Any]:
    """Synthèse structurée et interprétation fondée exclusivement sur les résultats calculés."""
    rep: dict[str, Any] = {"sections": [], "interpretation": []}
    S = rep["sections"]
    S.append(("1. Problème résolu", [("EDP", prob.meta["label"]), ("Type", CLASS_LABELS[prob.cls]),
                                      ("Domaine", prob.describe().split(" — ")[-1]),
                                      ("Formulation", pde_latex(prob, False))]))
    meth = METHOD_LABELS[prob.method] if prob.cls != "transport" else "Différences finies (schémas de transport)"
    scheme = res.get("scheme") or ("stationnaire" if not prob.time_dependent else "")
    S.append(("2. Méthode", [("Discrétisation spatiale", meth), ("Schéma temporel", scheme or "—")]))
    params = [("Configuration", f"#{res.get('cfg_hash', '')}")]
    if prob.time_dependent:
        params += [("T", fmt(prob.T)), ("Δt", fmt(res.get("dt"))), ("N_t", str(res.get("nt")))]
        for k, v in (res.get("numbers") or {}).items():
            if k in ("r", "σ", "λ", "Pe_h", "CFL_adv"):
                params.append((k, fmt(v)))
    S.append(("3. Paramètres utilisés", params))
    disc = [("Pas h (max)", fmt(res.get("h"))), ("Nœuds", str(res.get("n_nodes")))]
    if res.get("n_elems"):
        disc.append(("Éléments", str(res.get("n_elems"))))
    S.append(("4. Discrétisation", disc))
    S.append(("5. Nombre d'inconnues", [("Degrés de liberté", str(res.get("dofs")))]))
    sysl = [("Taille", f"{res.get('dofs')} × {res.get('dofs')}"), ("Non-zéros", str((res.get("matrix") or {}).get("nnz")))]
    if res.get("solver"):
        sv = res["solver"]
        sysl += [("Solveur", str(sv.get("kind"))), ("Résidu relatif", fmt(sv.get("residual"))),
                 ("Itérations", str(sv.get("iterations")))]
        if sv.get("symmetric") is not None:
            sysl.append(("Symétrique", "oui" if sv.get("symmetric") else "non"))
    elif res.get("factor_info"):
        sysl.append(("Factorisation", res.get("factor_info")))
    S.append(("6. Système numérique", sysl))
    stl = []
    if stab:
        for c in stab.get("criteria", []):
            stl.append((c["name"], f"{c['value']} — {'satisfaite' if c['ok'] else 'NON satisfaite'} ({c['kind']})"))
        if "max_G" in stab:
            stl.append(("max |G(θ)|", fmt(stab["max_G"], 6)))
    else:
        stl.append(("—", "sans objet (stationnaire)" if not prob.time_dependent else "non analysée"))
    S.append(("7. Conditions de stabilité", stl))
    U = np.asarray(res["U"])
    resl = [("min u_h", fmt(U.min())), ("max u_h", fmt(U.max()))]
    if res.get("blowup"):
        b = res["blowup"]
        resl.append(("Explosion numérique", f"détectée au pas {b['step']} (t = {fmt(b['time'])}, |u| = {fmt(b['value'])})"))
    S.append(("8. Résultats", resl))
    errs = res.get("errors")
    if errs:
        el = []
        for k, lab in (("l2", "‖e‖_L²"), ("h1", "|e|_H¹"), ("max", "‖e‖_∞"), ("l2_rel", "erreur L² relative"),
                       ("l1", "‖e‖_L¹")):
            if errs.get(k) is not None:
                el.append((lab, fmt(errs[k])))
        S.append(("9. Erreurs calculées", el))
    else:
        S.append(("9. Erreurs calculées", [("—", "solution exacte non fournie")]))
    concl = []
    I = rep["interpretation"]
    if res.get("blowup"):
        concl.append("Calcul instable : résultats non exploitables.")
    elif errs:
        rel = errs.get("l2_rel", errs.get("max_rel", 0))
        concl.append(f"Calcul stable ; erreur relative L² = {fmt(rel)}.")
    else:
        concl.append("Calcul stable ; pas de solution exacte pour quantifier l'erreur.")
    S.append(("10. Conclusion numérique", [("Bilan", " ".join(concl))]))
    # -------- interprétation mathématique (faits calculés uniquement) --------
    mp = mprops or {}
    if not prob.time_dependent:
        if res.get("solver", {}).get("symmetric"):
            I.append("La matrice du système est symétrique car la forme bilinéaire a(u, v) est symétrique (pas de terme "
                     "d'advection) et la méthode la préserve : en éléments finis K_ij = a(φ_j, φ_i) ; en différences "
                     "finies, le flux entre deux nœuds voisins est compté avec le même coefficient dans les deux lignes.")
        if mp.get("spd"):
            I.append(f"Elle est définie positive (plus petite valeur propre {fmt(mp['lam_min'])} > 0) : c'est la "
                     "traduction discrète de la coercivité — Lax–Milgram en dimension finie garantit l'unicité de u_h. "
                     f"Conditionnement spectral κ₂ = {fmt(mp.get('cond2'))}, qui croît comme h⁻².")
        if mp.get("m_matrix"):
            I.append("C'est une M-matrice (diagonale positive, termes hors diagonaux ≤ 0, diagonale dominante, "
                     "irréductible) : A⁻¹ ≥ 0, d'où le principe du maximum discret et la stabilité en norme ∞.")
        if res.get("kernel"):
            comp = res.get("compatibility", {})
            I.append(f"Problème de Neumann pur : la matrice est singulière (noyau = constantes). Le système augmenté "
                     f"impose ∫u_h = 0. Défaut de compatibilité mesuré ∫f + ∫g = {fmt(comp.get('total', float('nan')))}.")
    if prob.cls == "parabolic" and res.get("hist") is not None and not res.get("blowup"):
        rate = _decay_rate(res["hist"]["t"], res["hist"]["norm"])
        f_zero = prob.f_expr == 0 and all(b["g"] == 0 for b in prob.bc.values())
        if rate is not None and f_zero:
            pc = poincare_constant(prob)
            kmin = prob.coefficient_range("k")[0]
            cmin = prob.coefficient_range("c")[0]
            pred = -(kmin * pc["lambda1"] + max(cmin, 0.0)) if not (prob.bx or prob.by) else None
            txt = (f"Amortissement : la norme L² décroît exponentiellement, taux mesuré d/dt log‖u_h‖ ≈ {fmt(rate)}. ")
            if pred is not None and prob.k.is_constant and prob.c.is_constant and all(
                    prob.bc[s]["type"] in ("dirichlet", "neumann") for s in prob.sides):
                txt += (f"La théorie prévoit −(k λ₁ + c) = {fmt(pred)} (premier mode propre) : chaque mode de Fourier "
                        "décroît comme e^{−kλ_m t}, les hautes fréquences disparaissent les premières — effet régularisant "
                        "de l'équation de la chaleur.")
            I.append(txt)
    if prob.time_dependent and stab:
        bad = [c for c in stab.get("criteria", []) if not c["ok"] and "Péclet" not in c["name"]]
        if res.get("blowup"):
            gm = stab.get("max_G")
            b = res["blowup"]
            msg = (f"Instabilité observée au pas {b['step']} (t = {fmt(b['time'])}, max|u_h| = {fmt(b['value'])}). ")
            meas = measured_growth_factor(res)
            facts = []
            if meas is not None:
                facts.append(f"facteur d'amplification mesuré par pas ≈ {meas:.4f}")
            if spectral and spectral.get("rho") is not None:
                facts.append(f"rayon spectral exact de la matrice d'itération (conditions aux limites réelles) "
                             f"ρ = {spectral['rho']:.4f}")
            if gm is not None:
                facts.append(f"borne de von Neumann (grille périodique) max|G(θ)| = {gm:.4f}")
            if facts:
                msg += "Comparaison : " + " ; ".join(facts) + ". "
            msg += ("Les modes les plus oscillants (θ ≈ π), présents initialement au niveau des erreurs d'arrondi, sont "
                    "multipliés à chaque pas par un facteur de module > 1 et finissent par dominer la solution "
                    "(oscillations en dents de scie). Le critère de stabilité n'est pas respecté : réduire Δt ou "
                    "utiliser un schéma implicite.")
            I.append(msg)
        elif bad:
            gm = stab.get("max_G")
            n_steps = res.get("steps_done", res.get("nt", 0))
            if gm is not None and gm > 1 and n_steps:
                I.append(f"Le schéma est instable pour ces paramètres (max|G| = {fmt(gm, 6)} > 1) même si la solution "
                         f"calculée paraît acceptable : le mode le plus amplifié est multiplié par au plus "
                         f"max|G|ⁿ = {fmt(gm ** n_steps)} en n = {n_steps} pas. Tant que son amplitude initiale "
                         "(niveau des erreurs d'arrondi) reste petite devant ce facteur, l'instabilité est invisible ; "
                         "elle apparaît d'autant plus tôt que le maillage est fin ou que T est grand.")
        elif stab.get("criteria"):
            I.append("Toutes les conditions de stabilité sont satisfaites : les perturbations (erreurs d'arrondi, erreurs "
                     "de troncature) ne sont pas amplifiées, ce qui, combiné à la consistance, garantit la convergence "
                     "(théorème de Lax–Richtmyer).")
    if prob.cls == "hyperbolic" and res.get("hist") is not None and "energy" in res["hist"]:
        E = np.asarray(res["hist"]["energy"])
        if len(E) > 2 and abs(E[0]) > 0 and not res.get("blowup"):
            drift = float((E.max() - E.min()) / abs(E[0]))
            forcing = not (prob.f_expr == 0 and all(b["g"] == 0 for b in prob.bc.values()))
            I.append(f"Énergie discrète : variation relative {fmt(drift)} sur [0, T]. " +
                     ("Le schéma est non dissipatif : il conserve exactement une énergie discrète (au sens de la forme "
                      "quadratique ½VᵀMV + ½ŪᵀKŪ), image de la conservation E(t) = E(0) du problème continu."
                      if not forcing else "Les termes sources ou conditions non homogènes injectent de l'énergie : "
                                          "la variation est physique."))
    if prob.cls == "transport":
        vn = res.get("vonneumann_check")
        if vn:
            phase_err = (vn["meas_phase"] - ((vn["exact_phase"] + np.pi) % (2 * np.pi) - np.pi))
            I.append(f"Vérification de von Neumann : le mode de Fourier dominant (k = {vn['k']}) a été amorti d'un "
                     f"facteur {fmt(vn['meas_amp'], 8)} ; la prédiction |G(θ_k)|^N = {fmt(vn['pred_amp'], 8)} "
                     f"(écart relatif {fmt(vn['rel_diff'])}). Sur une grille périodique, les modes de Fourier sont "
                     "vecteurs propres de l'opérateur discret : l'analyse de von Neumann est exacte.")
            if vn["meas_amp"] < 0.999:
                I.append("Dissipation numérique : l'amplitude diminue alors que la solution exacte est transportée sans "
                         "déformation. Pour le schéma décentré, l'équation équivalente u_t + a u_x = (a h/2)(1 − λ) u_xx "
                         "révèle une viscosité numérique d'ordre h.")
            if abs(phase_err) > 1e-3:
                I.append(f"Dispersion numérique : déphasage de {fmt(phase_err)} rad par rapport à la translation exacte "
                         "(la vitesse de phase discrète dépend du nombre d'onde).")
    if (prob.bx or prob.by) and stab and stab.get("Pe_h", 0) > 1 and res.get("Uex") is not None:
        Uex = np.asarray(res["Uex"])
        over = float(max(U.max() - Uex.max(), Uex.min() - U.min(), 0.0))
        I.append(f"Péclet de maille Pe_h = {fmt(stab['Pe_h'])} > 1 : régime dominé par l'advection. "
                 + (f"Dépassement mesuré par rapport aux bornes de la solution exacte : {fmt(over)} (oscillations "
                    "parasites)." if over > 1e-3 * (abs(Uex).max() + 1e-300) else
                    "Aucun dépassement significatif des bornes de la solution exacte n'est observé."))
    if errs and res.get("Uex") is not None:
        e = np.abs(np.asarray(res["U"]) - np.asarray(res["Uex"]))
        j = int(np.argmax(e))
        pt = res["nodes"][j]
        where = f"x = {fmt(pt[0])}" + (f", y = {fmt(pt[1])}" if len(pt) > 1 else "")
        I.append(f"L'erreur nodale maximale ({fmt(e[j])}) est atteinte en {where}. "
                 + ("Elle se situe au voisinage du coin rentrant (0, 0) où la solution est singulière (∇u ~ r^(−1/3))."
                    if prob.shape == "lshape" and float(np.hypot(*pt[:2])) < 0.3 else ""))
        if prob.method == "fem" and prob.dim == 1 and errs.get("max_nodal", 1) < 1e-10 and prob.cls == "elliptic" \
                and prob.k.is_constant:
            I.append("Superconvergence nodale : en 1D, avec k constant, la solution P1 coïncide avec la solution exacte "
                     "aux nœuds (la fonction de Green appartient à V_h) ; l'erreur ne provient que de l'intérieur des éléments.")
    if conv and conv.get("ok") and any(r_.get("blowup") for r_ in conv["rows"]):
        I.append("Étude de convergence : les calculs explosent sur les maillages raffinés (nombre de stabilité "
                 "maintenu au-delà de la limite). L'erreur augmente quand h diminue : un schéma consistant mais "
                 "instable ne converge pas — illustration directe du théorème de Lax–Richtmyer. Les ordres « observés » "
                 "n'ont alors aucune signification.")
    elif conv and conv.get("ok"):
        th = conv["theory"]
        for k in conv["keys"]:
            p_obs = conv["slopes"].get(k)
            p_th = th["orders"].get(k)
            if p_obs is None or p_th is None:
                continue
            agree = abs(p_obs - p_th) < 0.25
            if prob.method == "fem" and prob.cls != "transport":
                why = ("en accord avec l'analyse : u_h est quasi optimale dans V_h (lemme de Céa) et l'interpolation P1 "
                       "approche u à l'ordre h (H¹) ou h² (L², dualité d'Aubin–Nitsche) ; raffiner le maillage enrichit "
                       "V_h et réduit donc l'erreur.")
            else:
                why = ("en accord avec l'analyse : l'erreur de troncature (consistance) est en hᵖ et la stabilité "
                       "empêche son amplification — Lax–Richtmyer ; raffiner réduit donc l'erreur comme hᵖ.")
            I.append(f"Convergence ({k.upper()}) : ordre observé {p_obs:.2f}, ordre théorique {fmt(p_th)} — "
                     + (why if agree else
                        "ÉCART avec la théorie : régime pré-asymptotique, pas de temps mal couplé, régularité "
                        "insuffisante ou plancher d'arrondi — voir les conditions ci-dessous."))
        I.append("Conditions de validité de l'ordre théorique : " + th["conditions"])
    if not I:
        I.append("Aucune propriété remarquable supplémentaire n'a été détectée dans les résultats calculés.")
    return rep


def report_markdown(rep: dict, cfg: dict) -> str:
    lines = [f"# {APP_TITLE} — rapport de résolution", "",
             f"*Généré le {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} — version {APP_VERSION}*", ""]
    for title, items in rep["sections"]:
        lines.append(f"## {title}")
        for k, v in items:
            v = str(v)
            if k == "Formulation":
                v = f"$${v}$$"
            lines.append(f"- **{k}** : {v}")
        lines.append("")
    lines.append("## INTERPRÉTATION MATHÉMATIQUE")
    for p in rep["interpretation"]:
        lines.append(f"- {p}")
    lines += ["", "## Configuration (reproductibilité)", "```json", json.dumps(cfg, indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines)


# =============================================================================
# 10b. VALIDATION AUTOMATIQUE (problèmes de référence)
# =============================================================================
def _cfg(**upd: Any) -> dict:
    return deep_merge(default_config(), upd)


def _orders_of(cfg: dict, ns: list[int], key: str) -> tuple[list[float], list[dict]]:
    res = []
    for n in ns:
        c = copy.deepcopy(cfg)
        c["mesh"]["nx"] = c["mesh"]["ny"] = n
        r = run_solve(c)
        if not r["ok"]:
            raise RuntimeError("; ".join(r["errors"]))
        res.append(r)
    ords = [math.log(res[i - 1]["errors"][key] / res[i]["errors"][key]) / math.log(res[i - 1]["h"] / res[i]["h"])
            for i in range(1, len(res))]
    return ords, res


REFERENCE_CASES: dict[str, dict[str, Any]] = {
    "T1": {"title": "Test 1 — Poisson 1D", "cfg": dict(family="poisson", dim=1, domain={"shape": "interval", "xa": 0, "xb": 1},
                                                       f="pi^2*sin(pi*x)", exact="sin(pi*x)", manufactured=False)},
    "T2": {"title": "Test 2 — Poisson 2D", "cfg": dict(family="poisson", dim=2, f="2*pi^2*sin(pi*x)*sin(pi*y)",
                                                       exact="sin(pi*x)*sin(pi*y)", manufactured=False)},
    "T3": {"title": "Test 3 — Chaleur 1D", "cfg": dict(family="heat", dim=1, domain={"shape": "interval"},
                                                       coef={"kappa": "1"}, f="0", u0="sin(pi*x)",
                                                       exact="exp(-pi^2*t)*sin(pi*x)", time={"T": 0.1})},
    "T4": {"title": "Test 4 — Transport 1D", "cfg": dict(family="transport", dim=1, domain={"shape": "interval"},
                                                         coef={"a": "1"}, u0="sin(2*pi*x)", exact="sin(2*pi*(x-t))",
                                                         transport_bc="periodic", time={"T": 1.0})},
    "T5": {"title": "Test 5 — Ondes 1D", "cfg": dict(family="wave", dim=1, domain={"shape": "interval"},
                                                     coef={"cw": "1"}, f="0", u0="sin(pi*x)", v0="0",
                                                     exact="sin(pi*x)*cos(pi*t)", time={"T": 0.6})},
}


def run_validation_suite(progress: Callable[[float, str], None] | None = None) -> list[dict[str, Any]]:
    """Batterie de tests automatiques : sécurité, matrices, solveurs, ordres, stabilité, conditions aux limites."""
    tests: list[tuple[str, str, Callable[[], tuple[bool, str, str]]]] = []

    def add(cat: str, name: str):
        def deco(fn):
            tests.append((cat, name, fn))
            return fn
        return deco

    @add("Sécurité", "Analyseur d'expressions : rejet du code arbitraire")
    def _t_parser():
        bad = ["__import__('os').system('ls')", "x.__class__", "lambda: 1", "open('f')", "x; y", "[x]", "1e999",
               "9^9^9", "eval(x)", "exec(x)", "x if x else y", "globals()"]
        rejected = 0
        for s in bad:
            try:
                parse_expression(s, ("x", "y"))
            except ExpressionError:
                rejected += 1
        good = {"2x": 2 * X_SYM, "x^2": X_SYM ** 2, "sin(pi x)": sympy.sin(sympy.pi * X_SYM), "-x^2": -X_SYM ** 2}
        okg = all(sympy.simplify(parse_expression(k) - v) == 0 for k, v in good.items())
        return rejected == len(bad) and okg, f"{rejected}/{len(bad)} rejetées, {len(good)} expressions valides reconnues", "toutes"

    @add("Éléments finis", "Fonctions de forme P1 : λᵢ(Aⱼ) = δᵢⱼ, Σλᵢ = 1, Σ∇λᵢ = 0")
    def _t_shape():
        P = np.array([[0.1, 0.2], [1.3, 0.4], [0.5, 1.1]])
        mesh = Mesh(P, np.array([[0, 1, 2]]), np.zeros((0, 2), int), np.zeros(0, int), ["boundary"])
        geo = p1_geometry(mesh)
        G = geo["grad"][0]
        # λ₁(x) = 1 + ∇λ₁·(x − A₁) et λᵢ(x) = ∇λᵢ·(x − A₁) (i = 2, 3) : gradients testés indépendamment
        lam_at = lambda pt: np.array([1 + G[0] @ (pt - P[0]), G[1] @ (pt - P[0]), G[2] @ (pt - P[0])])  # noqa: E731
        Mv = np.array([lam_at(P[j]) for j in range(3)])
        e1 = float(np.max(np.abs(Mv - np.eye(3))))
        rng = np.random.default_rng(1)
        pts = P[0] + rng.random((20, 1)) * (P[1] - P[0]) * 0.5 + rng.random((20, 1)) * (P[2] - P[0]) * 0.5
        e2 = max(abs(lam_at(p).sum() - 1) for p in pts)
        e3 = float(np.max(np.abs(G.sum(axis=0))))
        err = max(e1, e2, e3)
        return err < 1e-13, f"écart max {err:.1e}", "< 1e-13"

    @add("Éléments finis", "Quadratures sur le triangle : exactitude polynomiale")
    def _t_quad():
        worst = 0.0
        for deg in (2, 4, 5):
            b, w = triangle_rule(deg)
            for a_ in range(deg + 1):
                for c_ in range(deg + 1 - a_):
                    approx = 0.5 * np.sum(w * b[:, 1] ** a_ * b[:, 2] ** c_)
                    exact = math.factorial(a_) * math.factorial(c_) / math.factorial(a_ + c_ + 2)
                    worst = max(worst, abs(approx - exact))
        return worst < 1e-14, f"erreur max {worst:.1e}", "< 1e-14"

    @add("Matrices", "Matrices EF 1D : K = (1/h)tridiag(−1,2,−1), ΣMᵢⱼ = |Ω|, K·1 = 0")
    def _t_fem1d():
        prob = Problem(_cfg(family="poisson", dim=1, domain={"shape": "interval"}, method="fem",
                            bc={"left": {"type": "neumann"}, "right": {"type": "neumann"}}))
        mesh = mesh_1d(0, 1, 10)
        s = assemble_fem(prob, mesh)
        h = 0.1
        K = s.K.toarray()
        ref = (np.diag(np.r_[1, np.full(9, 2), 1]) - np.eye(11, k=1) - np.eye(11, k=-1)) / h
        e1 = float(np.max(np.abs(K - ref)))
        e2 = abs(s.M_cons.sum() - 1.0)
        e3 = float(np.max(np.abs(s.K @ np.ones(11))))
        sym = is_symmetric(s.K) and is_symmetric(s.M_cons)
        err = max(e1, e2, e3)
        return err < 1e-12 and sym, f"écart {err:.1e}, symétrie {'oui' if sym else 'non'}", "< 1e-12, symétrique"

    @add("Matrices", "Matrices EF 2D : symétrie, K·1 = 0, ΣMᵢⱼ = |Ω|, K semi-définie positive")
    def _t_fem2d():
        prob = Problem(_cfg(family="poisson", dim=2, method="fem",
                            bc={s: {"type": "neumann"} for s in ("left", "right", "bottom", "top")}))
        mesh = mesh_rectangle(0, 2, 0, 1, 6, 4, "crossed", "perturbed", 0.25, 3)
        s = assemble_fem(prob, mesh)
        e1 = float(np.max(np.abs(s.K @ np.ones(s.n))))
        e2 = abs(s.M_cons.sum() - 2.0)
        lmin = float(sla.eigvalsh(s.K.toarray()).min())
        ok = e1 < 1e-12 and e2 < 1e-12 and lmin > -1e-12 and is_symmetric(s.K)
        return ok, f"|K·1| = {e1:.1e}, |ΣM − 2| = {e2:.1e}, λ_min(K) = {lmin:.1e}", "≈ 0, λ_min ≥ 0"

    @add("Matrices", "Matrice DF 2D : stencil à 5 points et spectre 1D exact")
    def _t_fdm():
        prob = Problem(_cfg(family="poisson", dim=2, method="fdm"))
        s = assemble_fdm(prob, 8, 8)
        A = s.fd_matrix().tocsr()
        p = 4 + 4 * 9
        row = A[p].toarray().ravel()
        h = 1 / 8
        e1 = abs(row[p] - 4 / h ** 2) + abs(row[p - 1] + 1 / h ** 2) + abs(row[p + 1] + 1 / h ** 2) + \
            abs(row[p - 9] + 1 / h ** 2) + abs(row[p + 9] + 1 / h ** 2)
        prob1 = Problem(_cfg(family="poisson", dim=1, domain={"shape": "interval"}, method="fdm"))
        s1 = assemble_fdm(prob1, 16, 1)
        A1 = s1.fd_matrix()[s1.free][:, s1.free].toarray()
        ev = np.sort(np.linalg.eigvals(A1).real)
        h1 = 1 / 16
        ex = np.sort(4 / h1 ** 2 * np.sin(np.arange(1, 16) * np.pi * h1 / 2) ** 2)
        e2 = float(np.max(np.abs(ev - ex)) / ex.max())
        return e1 < 1e-9 and e2 < 1e-12, f"stencil {e1:.1e}, spectre {e2:.1e}", "≈ 0"

    @add("Problèmes de référence", "Test 1 — Poisson 1D (DF) : ordre 2, CL exactes")
    def _t1_fdm():
        c = _cfg(**REFERENCE_CASES["T1"]["cfg"], method="fdm")
        ords, res = _orders_of(c, [16, 32, 64, 128], "max")
        bc = max(abs(res[-1]["U"][0]), abs(res[-1]["U"][-1]))
        return abs(ords[-1] - 2) < 0.1 and bc == 0.0, f"ordres {['%.3f' % o for o in ords]}, |u_h(0)|+|u_h(1)| = {bc:.0e}", "p ≈ 2, CL exactes"

    @add("Problèmes de référence", "Test 1 — Poisson 1D (EF P1) : L² ordre 2, H¹ ordre 1")
    def _t1_fem():
        c = _cfg(**REFERENCE_CASES["T1"]["cfg"], method="fem")
        o2, res = _orders_of(c, [16, 32, 64, 128], "l2")
        o1, _ = _orders_of(c, [16, 32, 64, 128], "h1")
        nod = res[-1]["errors"]["max_nodal"]
        ok = abs(o2[-1] - 2) < 0.1 and abs(o1[-1] - 1) < 0.1 and nod < 1e-10
        return ok, f"L² {o2[-1]:.3f}, H¹ {o1[-1]:.3f}, erreur nodale {nod:.1e} (superconvergence)", "2 / 1"

    @add("Problèmes de référence", "Test 2 — Poisson 2D (DF) : ordre 2")
    def _t2_fdm():
        c = _cfg(**REFERENCE_CASES["T2"]["cfg"], method="fdm")
        ords, _ = _orders_of(c, [8, 16, 32, 64], "max")
        return abs(ords[-1] - 2) < 0.1, f"ordres {['%.3f' % o for o in ords]}", "p ≈ 2"

    @add("Problèmes de référence", "Test 2 — Poisson 2D (EF P1) : L² ordre 2, H¹ ordre 1")
    def _t2_fem():
        c = _cfg(**REFERENCE_CASES["T2"]["cfg"], method="fem")
        o2, _ = _orders_of(c, [8, 16, 32, 64], "l2")
        o1, _ = _orders_of(c, [8, 16, 32, 64], "h1")
        return abs(o2[-1] - 2) < 0.1 and abs(o1[-1] - 1) < 0.1, f"L² {o2[-1]:.3f}, H¹ {o1[-1]:.3f}", "2 / 1"

    @add("Éléments finis", "Patch test : solution affine reproduite exactement (maillage perturbé)")
    def _t_patch():
        g = "1+2*x+3*y"
        c = _cfg(family="laplace", dim=2, method="fem", exact=g, mesh={"grading": "perturbed", "pattern": "alternate"},
                 bc={s: {"type": "dirichlet", "g": g} for s in ("left", "right", "bottom", "top")})
        r = run_solve(c)
        e = r["errors"]["max_nodal"]
        return e < 1e-12, f"erreur max {e:.1e}", "< 1e-12"

    @add("Conditions aux limites", "Neumann + Robin, coefficients variables (DF et EF) : ordre 2")
    def _t_mixed():
        upd = dict(family="diffusion_reaction", dim=2, coef={"k": "1+x^2*y", "c": "1+x"}, exact="exp(x)*cos(2*y)+x*y",
                   manufactured=True, bc={"left": {"type": "neumann"}, "right": {"type": "robin", "beta": "2+y"},
                                          "bottom": {"type": "dirichlet"}, "top": {"type": "robin", "beta": "1"}})
        of, _ = _orders_of(_cfg(**upd, method="fdm"), [8, 16, 32], "l2")
        oe, _ = _orders_of(_cfg(**upd, method="fem"), [8, 16, 32], "l2")
        return abs(of[-1] - 2) < 0.15 and abs(oe[-1] - 2) < 0.15, f"DF {of[-1]:.3f}, EF {oe[-1]:.3f}", "2 / 2"

    @add("Conditions aux limites", "Neumann pur : compatibilité, moyenne nulle, ordre 2")
    def _t_neumann():
        upd = dict(family="poisson", dim=2, method="fem", f="2*pi^2*cos(pi*x)*cos(pi*y)", exact="cos(pi*x)*cos(pi*y)",
                   bc={s: {"type": "neumann", "g": "0"} for s in ("left", "right", "bottom", "top")})
        o, res = _orders_of(_cfg(**upd), [8, 16, 32], "l2")
        comp = res[-1]["compatibility"]["relative"]
        return abs(o[-1] - 2) < 0.15 and comp < 1e-10, f"ordre {o[-1]:.3f}, compatibilité {comp:.1e}", "2, ≈ 0"

    @add("Problèmes de référence", "Test 3 — Chaleur 1D, Crank–Nicolson (Δt ∝ h) : ordre 2 et solution propre discrète")
    def _t3_cn():
        errs, hs, ident = [], [], 0.0
        for n in (16, 32, 64, 128):
            c = _cfg(**REFERENCE_CASES["T3"]["cfg"], method="fdm")
            c["mesh"]["nx"] = n
            c["time"].update(theta=0.5, dt_mode="dt", dt=(1 / n) / 4)
            r = run_solve(c)
            h = 1 / n
            mu = 4 / h ** 2 * math.sin(math.pi * h / 2) ** 2
            z = r["dt"] * mu
            pred = ((1 - z / 2) / (1 + z / 2)) ** r["nt"] * np.sin(np.pi * r["nodes"][:, 0])
            ident = max(ident, float(np.max(np.abs(r["U"] - pred))))
            errs.append(r["errors"]["max"])
            hs.append(h)
        o = math.log(errs[-2] / errs[-1]) / math.log(2)
        return abs(o - 2) < 0.1 and ident < 1e-12, f"ordre {o:.3f}, |U − Rⁿ sin(πx)| = {ident:.1e}", "p ≈ 2, identité exacte"

    @add("Stabilité", "Euler explicite : r = 0,45 stable, r = 0,55 instable (critère r ≤ 1/2)")
    def _t_explicit():
        out = []
        for r_ in (0.45, 0.55):
            c = _cfg(**REFERENCE_CASES["T3"]["cfg"], method="fdm")
            c["mesh"]["nx"] = 50
            c["time"].update(theta=0.0, dt_mode="number", number=r_, T=0.5)
            out.append(run_solve(c))
        ok = out[0]["blowup"] is None and out[0]["errors"]["max"] < 1e-3 and out[1]["blowup"] is not None
        return ok, (f"r=0.45 : erreur {out[0]['errors']['max']:.1e} ; r=0.55 : explosion au pas "
                    f"{out[1]['blowup']['step'] if out[1]['blowup'] else '—'}"), "stable / instable"

    @add("Stabilité", "EF : λ_max(M⁻¹K)·h² → 12 (masse cohérente), 4 (masse condensée)")
    def _t_fem_spec():
        c = _cfg(family="heat", dim=1, domain={"shape": "interval"}, method="fem", time={"T": 0.1, "theta": 0.0})
        c["mesh"]["nx"] = 60
        prob = Problem(c)
        sa = spectral_analysis(prob, prob.cfg, dt=None)
        h2 = (1 / 60) ** 2
        r1 = sa["lam_max_consistent"] * h2
        r2 = sa["lam_max_lumped"] * h2
        return abs(r1 - 12) / 12 < 0.01 and abs(r2 - 4) / 4 < 0.01, f"{r1:.3f} et {r2:.3f}", "12 et 4 (à 1 %)"

    @add("Problèmes de référence", "Test 4 — Transport : décentré ordre 1, Lax–Wendroff ordre 2, |G|ᴺ vérifié")
    def _t4():
        res = {}
        for sc in ("upwind", "lax_wendroff"):
            c = _cfg(**REFERENCE_CASES["T4"]["cfg"])
            c["time"].update(transport_scheme=sc, dt_mode="number", number=0.8)
            o, rr = _orders_of(c, [32, 64, 128, 256], "l2")
            res[sc] = (o[-1], rr[-1]["vonneumann_check"]["rel_diff"])
        ok = abs(res["upwind"][0] - 1) < 0.1 and abs(res["lax_wendroff"][0] - 2) < 0.1 and \
            max(res["upwind"][1], res["lax_wendroff"][1]) < 1e-10
        return ok, (f"décentré {res['upwind'][0]:.3f}, LW {res['lax_wendroff'][0]:.3f}, écart von Neumann "
                    f"{max(res['upwind'][1], res['lax_wendroff'][1]):.1e}"), "1 / 2 / < 1e-10"

    @add("Problèmes de référence", "Test 5 — Ondes 1D : ordre 2, σ = 1 exact, σ = 1,1 instable, énergie conservée")
    def _t5():
        c = _cfg(**REFERENCE_CASES["T5"]["cfg"], method="fdm")
        c["time"].update(dt_mode="number", number=0.5)
        o, _ = _orders_of(c, [16, 32, 64, 128], "max")
        c1 = _cfg(**REFERENCE_CASES["T5"]["cfg"], method="fdm")
        c1["mesh"]["nx"] = 40
        c1["time"].update(dt_mode="number", number=1.0, T=1.0)
        r1 = run_solve(c1)
        c2 = copy.deepcopy(c1)
        c2["time"].update(number=1.1, T=4.0)
        r2 = run_solve(c2)
        c3 = copy.deepcopy(c1)
        c3["time"].update(number=0.9, T=2.0)
        r3 = run_solve(c3)
        E = r3["hist"]["energy"]
        drift = float((E.max() - E.min()) / E[0])
        ok = abs(o[-1] - 2) < 0.1 and r1["errors"]["max"] < 1e-12 and r2["blowup"] is not None and drift < 1e-12
        return ok, (f"ordre {o[-1]:.3f}, erreur σ=1 : {r1['errors']['max']:.1e}, σ=1,1 : "
                    f"{'explosion' if r2['blowup'] else 'pas d’explosion'}, dérive d'énergie {drift:.1e}"), "2 / ≈0 / instable / ≈0"

    @add("Problèmes d'évolution", "Ondes 2D (DF et EF condensée) : ordre 2")
    def _t_wave2d():
        upd = dict(family="wave", dim=2, coef={"cw": "1"}, exact="sin(pi*x)*sin(pi*y)*cos(sqrt(2)*pi*t)",
                   manufactured=True, time={"T": 0.5, "dt_mode": "number", "number": 0.5, "lumped": True})
        of, _ = _orders_of(_cfg(**upd, method="fdm"), [8, 16, 32], "max")
        oe, _ = _orders_of(_cfg(**upd, method="fem"), [8, 16, 32], "l2")
        return abs(of[-1] - 2) < 0.15 and abs(oe[-1] - 2) < 0.15, f"DF {of[-1]:.3f}, EF {oe[-1]:.3f}", "2 / 2"

    @add("Problèmes d'évolution", "Advection–diffusion 1D, Dirichlet dépendant du temps, CN : ordre 2")
    def _t_advdiff():
        upd = dict(family="advection_diffusion", dim=1, domain={"shape": "interval"}, coef={"k": "0.1", "bx": "1"},
                   exact="exp(-0.1*pi^2*t)*sin(pi*(x-t))", manufactured=True,
                   time={"T": 0.5, "theta": 0.5, "dt_mode": "dt", "dt": 0.0})
        orders = {}
        for meth in ("fdm", "fem"):
            errs = []
            for n in (32, 64, 128):
                c = _cfg(**upd, method=meth)
                c["mesh"]["nx"] = n
                c["time"]["dt"] = 0.5 / n
                errs.append(run_solve(c)["errors"]["l2"])
            orders[meth] = math.log(errs[-2] / errs[-1]) / math.log(2)
        ok = all(abs(v - 2) < 0.15 for v in orders.values())
        return ok, f"DF {orders['fdm']:.3f}, EF {orders['fem']:.3f}", "2 / 2"

    @add("Régularité", "Domaine en L : ordre H¹ réduit ≈ 2/3 (singularité r^(2/3))")
    def _t_lshape():
        g = "(x^2+y^2)^(1/3)*sin(2/3*(atan2(-y,-x)+pi))"
        c = _cfg(family="laplace", dim=2, domain={"shape": "lshape"}, method="fem", exact=g,
                 bc={"boundary": {"type": "dirichlet", "g": g}})
        o, _ = _orders_of(c, [8, 16, 32], "h1")
        return 0.55 < o[-1] < 0.8, f"ordre H¹ observé {o[-1]:.3f} (théorie 2/3)", "∈ [0,55 ; 0,8]"

    @add("Analyse", "Lax–Milgram : constantes de Poincaré et statuts")
    def _t_lm():
        lm = lax_milgram_analysis(Problem(_cfg(family="poisson", dim=2)))
        a_ok = abs(lm["constants"]["lambda1"] - 2 * math.pi ** 2) < 1e-12 and lm["applicable"]
        a_ok = a_ok and abs(lm["constants"]["alpha"] - 2 * math.pi ** 2 / (1 + 2 * math.pi ** 2)) < 1e-12
        lm2 = lax_milgram_analysis(Problem(_cfg(family="diffusion_reaction", dim=1, domain={"shape": "interval"},
                                                coef={"k": "1", "c": "-20"})))
        lm3 = lax_milgram_analysis(Problem(_cfg(family="poisson", dim=1, domain={"shape": "interval"}, f="cos(pi*x)",
                                                bc={"left": {"type": "neumann"}, "right": {"type": "neumann"}})))
        ok = a_ok and not lm2["applicable"] and lm3["applicable"] and "compatibility" in lm3
        return ok, (f"λ₁ = {lm['constants']['lambda1']:.6f} (2π²), c = −20 : "
                    f"{'non établi' if not lm2['applicable'] else 'établi'}, Neumann pur : compatibilité testée"), "conformes"

    @add("Analyse", "Classification : elliptique / parabolique / hyperbolique / 1er ordre")
    def _t_class():
        got = [classify_problem(Problem(_cfg(family=f, dim=d, domain={"shape": "interval" if d == 1 else "rectangle"})))["type"]
               for f, d in (("poisson", 2), ("heat", 1), ("wave", 2), ("transport", 1))]
        exp_ = ["Elliptique", "Parabolique", "Hyperbolique", "Hyperbolique du premier ordre"]
        return got == exp_, ", ".join(got), ", ".join(exp_)

    @add("Solveurs", "Solveurs itératifs (CG, GMRES+ILU, BiCGSTAB) ≡ solveur direct")
    def _t_iter():
        base = run_solve(_cfg(family="poisson", dim=2, method="fem"))
        worst = 0.0
        for kind, prec in (("cg", "jacobi"), ("gmres", "ilu"), ("bicgstab", "jacobi")):
            r = run_solve(_cfg(family="poisson", dim=2, method="fem", solver={"kind": kind, "precond": prec, "tol": 1e-12}))
            worst = max(worst, float(np.max(np.abs(r["U"] - base["U"]))))
        return worst < 1e-8, f"écart max {worst:.1e}", "< 1e-8"

    results = []
    for i, (cat, name, fn) in enumerate(tests):
        if progress:
            progress(i / len(tests), name)
        t0 = time.perf_counter()
        try:
            ok, measured, expected = fn()
        except Exception as exc:  # noqa: BLE001
            ok, measured, expected = False, f"exception : {exc}", "—"
        results.append({"category": cat, "test": name, "passed": bool(ok), "measured": measured,
                        "expected": expected, "time": time.perf_counter() - t0})
    if progress:
        progress(1.0, "terminé")
    return results


# =============================================================================
# 11. VISUALISATION SCIENTIFIQUE (Plotly)
# =============================================================================
PALETTES: dict[str, dict[str, Any]] = {
    "dark": {
        "bg": "#0a1020", "surface": "#0f182b", "surface2": "#131f37", "border": "rgba(122,152,204,0.18)",
        "text": "#e2e9f5", "text2": "#a9b8cf", "muted": "#74859f", "grid": "rgba(148,163,184,0.12)",
        "axis": "rgba(148,163,184,0.30)", "accent": "#3b82f6", "accent2": "#22d3ee",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
        "good": "#0ca30c", "warn": "#fab219", "bad": "#d03b3b", "exact": "#e2e9f5",
        "seq": [[0.0, "#123566"], [0.18, "#15407c"], [0.36, "#1650a3"], [0.52, "#1f6fd6"], [0.68, "#2f8ff0"],
                [0.82, "#38b6f0"], [0.92, "#6fd6f2"], [1.0, "#c3f2fb"]],
        "div": [[0.0, "#9ec5f4"], [0.2, "#3987e5"], [0.5, "#3a3d47"], [0.8, "#e34948"], [1.0, "#f4a3a3"]],
        "band_bad": "rgba(208,59,59,0.12)", "band_good": "rgba(12,163,12,0.07)",
    },
    "light": {
        "bg": "#f5f7fb", "surface": "#ffffff", "surface2": "#eef2f8", "border": "rgba(15,30,60,0.12)",
        "text": "#111a2e", "text2": "#46536b", "muted": "#7a8599", "grid": "rgba(15,30,60,0.08)",
        "axis": "rgba(15,30,60,0.25)", "accent": "#1d5fd1", "accent2": "#0e9f95",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "good": "#0ca30c", "warn": "#fab219", "bad": "#d03b3b", "exact": "#111a2e",
        "seq": [[0.0, "#eaf5fd"], [0.15, "#c3e6f9"], [0.32, "#8fd0f3"], [0.5, "#4fb3ea"], [0.66, "#2590d8"],
                [0.8, "#1a6cc0"], [0.92, "#1650a3"], [1.0, "#0f3366"]],
        "div": [[0.0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8a89"], [1.0, "#b8302f"]],
        "band_bad": "rgba(208,59,59,0.10)", "band_good": "rgba(12,163,12,0.06)",
    },
}
FONT_STACK = "Inter, 'Source Sans Pro', system-ui, -apple-system, 'Segoe UI', sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "pde-lab-figure"}}
COLORSCALE_LABELS = {"auto": "Automatique (séquentiel ou divergent)", "seq": "Océan (séquentiel, bleu → cyan)",
                     "Viridis": "Viridis (perceptuel)", "Cividis": "Cividis (daltonisme)", "div": "Divergent bleu–rouge"}


def pal(theme: str) -> dict[str, Any]:
    return PALETTES.get(theme, PALETTES["dark"])


def style_fig(fig: go.Figure, theme: str, title: str | None = None, height: int | None = 420,
              xlabel: str | None = None, ylabel: str | None = None, legend: bool = True) -> go.Figure:
    P = pal(theme)
    axis = dict(gridcolor=P["grid"], zerolinecolor=P["axis"], linecolor=P["axis"], showline=True,
                ticks="outside", tickcolor=P["axis"], ticklen=4, title_font=dict(color=P["text2"], size=13),
                tickfont=dict(color=P["text2"], size=12))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=P["text2"], size=13), colorway=P["series"],
        # titre ET légende : la légende (au-dessus de la zone de tracé) passe sous la ligne du titre
        margin=dict(l=64, r=24, t=(96 if legend else 58) if title else (40 if legend else 24), b=56),
        hoverlabel=dict(bgcolor=P["surface2"], bordercolor=P["border"], font=dict(color=P["text"], family=FONT_STACK)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=P["text2"], size=12), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        showlegend=legend,
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=P["text"], size=15), x=0.01, xanchor="left", y=0.98))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)
    scene_axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=P["grid"], zerolinecolor=P["axis"],
                      color=P["text2"], showbackground=False)
    fig.update_scenes(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis)
    return fig


def colorscale_for(values: np.ndarray, theme: str, choice: str = "auto") -> tuple[Any, float | None]:
    """Choix de l'échelle : divergente centrée en 0 si le champ change nettement de signe."""
    P = pal(theme)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if choice == "auto":
        if v.size and v.min() < -0.05 * np.abs(v).max() and v.max() > 0.05 * np.abs(v).max():
            choice = "div"
        else:
            choice = "seq"
    if choice == "div":
        return P["div"], 0.0
    if choice == "seq":
        return P["seq"], None
    return choice, None


class P1Interpolator:
    """Évaluation exacte d'un champ P1 en des points quelconques (localisation par arbre k-d)."""

    def __init__(self, nodes: np.ndarray, elems: np.ndarray):
        self.nodes, self.elems = nodes, elems
        P = nodes[elems]
        self.P0 = P[:, 0]
        B = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)
        self.Binv = np.linalg.inv(B)
        self.tree = cKDTree(P.mean(axis=1))

    def locate(self, pts: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = min(k, len(self.elems))
        _, cand = self.tree.query(pts, k=k)
        if k == 1:
            cand = cand[:, None]
        d = pts[:, None, :] - self.P0[cand]
        l23 = np.einsum("nkij,nkj->nki", self.Binv[cand], d)
        bary = np.concatenate([(1 - l23.sum(-1))[..., None], l23], -1)
        inside = np.all(bary >= -1e-9, axis=-1)
        first = np.argmax(inside, axis=1)
        idx = np.arange(len(pts))
        return cand[idx, first], bary[idx, first], inside[idx, first]

    def __call__(self, U: np.ndarray, pts: np.ndarray) -> np.ndarray:
        el, b, found = self.locate(pts)
        vals = np.sum(b * U[self.elems[el]], axis=1)
        vals[~found] = np.nan
        return vals


def field_on_grid(res: dict, values: np.ndarray, ngrid: int = 141) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Champ 2D sur grille régulière : direct en DF, interpolation P1 exacte en EF (NaN hors du domaine)."""
    if res.get("elems") is None or res["method"] == "fdm":
        xs, ys = res["grid"]["xs"], res["grid"]["ys"]
        return xs, ys, np.asarray(values).reshape(len(ys), len(xs))
    nodes = res["nodes"]
    xs = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), ngrid)
    ys = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), ngrid)
    XX, YY = np.meshgrid(xs, ys)
    interp = P1Interpolator(nodes, res["elems"])
    Z = interp(np.asarray(values, dtype=float), np.stack([XX.ravel(), YY.ravel()], 1)).reshape(XX.shape)
    return xs, ys, Z


def fig_curves(x: np.ndarray, curves: list[dict], theme: str, title: str, xlabel: str, ylabel: str,
               height: int = 400, logy: bool = False, logx: bool = False) -> go.Figure:
    P = pal(theme)
    fig = go.Figure()
    for i, c in enumerate(curves):
        color = c.get("color") or P["series"][c.get("slot", i) % 8]
        mode = c.get("mode", "lines")
        fig.add_trace(go.Scatter(x=c.get("x", x), y=c["y"], name=c["name"], mode=mode,
                                 line=dict(color=color, width=c.get("width", 2), dash=c.get("dash", "solid")),
                                 marker=dict(size=c.get("msize", 7), color=color,
                                             line=dict(width=1.5, color=P["surface"])),
                                 hovertemplate=c.get("hover", "%{x:.4g}, %{y:.4g}<extra>" + c["name"] + "</extra>")))
    style_fig(fig, theme, title, height, xlabel, ylabel, legend=len(curves) > 1)
    if logy:
        fig.update_yaxes(type="log", exponentformat="power")
        tidy_log_axis(fig, "y", np.concatenate([np.ravel(np.asarray(c["y"], dtype=float)) for c in curves]))
    if logx:
        fig.update_xaxes(type="log", exponentformat="power")
        tidy_log_axis(fig, "x", np.ravel(np.asarray(x, dtype=float)))
    return fig


def tidy_log_axis(fig: go.Figure, axis: str, values: np.ndarray) -> None:
    """Axe logarithmique lisible : une graduation par décade dès que les données couvrent > 1,2 décade."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size >= 2 and math.log10(v.max() / v.min()) > 1.2:
        (fig.update_yaxes if axis == "y" else fig.update_xaxes)(dtick=1)


def fig_solution_1d(x: np.ndarray, U: np.ndarray, Uex: np.ndarray | None, theme: str, title: str,
                    ylabel: str = "u(x)", height: int = 400) -> go.Figure:
    P = pal(theme)
    order = np.argsort(x)
    x, U = x[order], np.asarray(U)[order]
    curves = []
    if Uex is not None:
        Uex = np.asarray(Uex)[order]
        xf = np.linspace(x.min(), x.max(), 600) if len(x) < 600 else x
        uf = np.interp(xf, x, Uex) if len(x) >= 600 else None
        curves.append({"x": xf if uf is not None else x, "y": uf if uf is not None else Uex,
                       "name": "solution exacte u", "color": P["muted"], "width": 2.5})
    curves.append({"y": U, "name": "solution numérique u_h", "slot": 0,
                   "mode": "lines+markers" if len(x) <= 80 else "lines", "width": 2})
    return fig_curves(x, curves, theme, title, "x", ylabel, height)


def fig_field_2d(res: dict, values: np.ndarray, theme: str, kind: str, title: str, zlabel: str = "u",
                 colorscale: str = "auto", show_mesh: bool = False, height: int = 520) -> go.Figure:
    P = pal(theme)
    vals = np.asarray(values, dtype=float)
    cs, mid = colorscale_for(vals, theme, colorscale)
    cbar = dict(title=dict(text=zlabel, font=dict(color=P["text2"])), tickfont=dict(color=P["text2"]),
                outlinewidth=0, thickness=14, len=0.8)
    fem = res.get("elems") is not None and res["method"] == "fem"
    fig = go.Figure()
    if kind == "surface":
        if fem and len(res["elems"]) <= 30000:
            nd, el = res["nodes"], res["elems"]
            fig.add_trace(go.Mesh3d(x=nd[:, 0], y=nd[:, 1], z=vals, i=el[:, 0], j=el[:, 1], k=el[:, 2],
                                    intensity=vals, colorscale=cs, cmid=mid, colorbar=cbar, flatshading=False,
                                    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25, roughness=0.6),
                                    hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>" + zlabel + "=%{z:.4g}<extra></extra>",
                                    name=zlabel))
            if show_mesh and len(el) <= 6000:
                ex, ey, ez = _edge_lines(nd, el, vals)
                fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color=P["text2"], width=1),
                                           opacity=0.35, hoverinfo="skip", name="arêtes"))
        else:
            xs, ys, Z = field_on_grid(res, vals, 121)
            fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale=cs, cmid=mid, colorbar=cbar,
                                     lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2),
                                     hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>" + zlabel + "=%{z:.4g}<extra></extra>"))
        style_fig(fig, theme, title, height, legend=False)
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title=zlabel,
                                     aspectmode="manual", aspectratio=dict(x=1, y=_aspect_y(res), z=0.6),
                                     camera=dict(eye=dict(x=1.2, y=-1.3, z=0.78))),
                          margin=dict(l=0, r=0, t=50, b=0))
        return fig
    xs, ys, Z = field_on_grid(res, vals, 161)
    if kind == "contour":
        fig.add_trace(go.Contour(x=xs, y=ys, z=Z, colorscale=cs, zmid=mid, colorbar=cbar, ncontours=18,
                                 contours=dict(coloring="heatmap", showlines=True),
                                 line=dict(width=0.6, color=P["surface"]),
                                 hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>" + zlabel + "=%{z:.4g}<extra></extra>"))
    else:
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=Z, colorscale=cs, zmid=mid, colorbar=cbar, zsmooth="best",
                                 hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>" + zlabel + "=%{z:.4g}<extra></extra>"))
    if show_mesh:
        if fem and len(res["elems"]) <= 12000:
            ex, ey = _edge_lines_2d(res["nodes"], res["elems"])
            fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines", line=dict(color=P["surface"], width=0.6),
                                       opacity=0.55, hoverinfo="skip", showlegend=False))
        elif not fem and res.get("grid") is not None and len(res["grid"]["xs"]) <= 130:
            gx, gy = _grid_lines(res["grid"]["xs"], res["grid"]["ys"])
            fig.add_trace(go.Scattergl(x=gx, y=gy, mode="lines", line=dict(color=P["surface"], width=0.6),
                                       opacity=0.5, hoverinfo="skip", showlegend=False))
    style_fig(fig, theme, title, height, "x", "y", legend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _aspect_y(res: dict) -> float:
    nd = res["nodes"]
    if nd.shape[1] < 2:
        return 1.0
    lx = nd[:, 0].max() - nd[:, 0].min()
    ly = nd[:, 1].max() - nd[:, 1].min()
    return float(np.clip(ly / max(lx, 1e-12), 0.25, 4.0))


def _edge_lines_2d(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.vstack([elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    xs = np.full(3 * len(e), np.nan)
    ys = np.full(3 * len(e), np.nan)
    xs[0::3], xs[1::3] = nodes[e[:, 0], 0], nodes[e[:, 1], 0]
    ys[0::3], ys[1::3] = nodes[e[:, 0], 1], nodes[e[:, 1], 1]
    return xs, ys


def _edge_lines(nodes: np.ndarray, elems: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.vstack([elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    out = []
    for comp in (nodes[:, 0], nodes[:, 1], z):
        a = np.full(3 * len(e), np.nan)
        a[0::3], a[1::3] = comp[e[:, 0]], comp[e[:, 1]]
        out.append(a)
    return out[0], out[1], out[2]


def _grid_lines(xs: np.ndarray, ys: np.ndarray) -> tuple[list, list]:
    gx, gy = [], []
    for x in xs:
        gx += [x, x, None]
        gy += [ys[0], ys[-1], None]
    for y in ys:
        gx += [xs[0], xs[-1], None]
        gy += [y, y, None]
    return gx, gy


NODE_CATEGORY_LABELS = {"interior": "nœud intérieur", "dirichlet": "Dirichlet", "neumann": "Neumann",
                        "robin": "Robin", "periodic": "périodique", "center": "nœud intérieur"}


def node_categories(prob: Problem, nodes: np.ndarray, side_nodes: dict[str, np.ndarray]) -> np.ndarray:
    cat = np.array(["interior"] * len(nodes), dtype=object)
    for typ in ("robin", "neumann", "dirichlet"):  # Dirichlet prioritaire aux coins
        for s, idx in side_nodes.items():
            if prob.bc.get(s, {}).get("type") == typ:
                cat[idx] = typ
    return cat


def fig_mesh(nodes: np.ndarray, elems: np.ndarray | None, cats: np.ndarray, theme: str, title: str,
             grid: dict | None = None, show_ids: bool = False, height: int = 520,
             show_nodes: bool = True) -> go.Figure:
    P = pal(theme)
    fig = go.Figure()
    d = nodes.shape[1]
    if d == 1 or (d == 2 and np.allclose(nodes[:, 1], 0) and elems is not None and elems.shape[1] == 2):
        x = nodes[:, 0]
        fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[0, 0], mode="lines", line=dict(color=P["axis"], width=2),
                                 hoverinfo="skip", showlegend=False))
        for slot, cname in enumerate(("interior", "dirichlet", "neumann", "robin")):
            m = cats == cname
            if not np.any(m):
                continue
            fig.add_trace(go.Scatter(x=x[m], y=np.zeros(m.sum()), mode="markers", name=NODE_CATEGORY_LABELS[cname],
                                     marker=dict(size=9 if len(x) < 200 else 5,
                                                 color=P["muted"] if cname == "interior" else P["series"][slot - 1],
                                                 line=dict(width=1.5, color=P["surface"])),
                                     hovertemplate="x = %{x:.4f}<extra>" + NODE_CATEGORY_LABELS[cname] + "</extra>"))
        if show_ids and len(x) <= 60:
            fig.add_trace(go.Scatter(x=x, y=np.full(len(x), 0.08), mode="text", text=[str(i) for i in range(len(x))],
                                     textfont=dict(size=10, color=P["text2"]), hoverinfo="skip", showlegend=False))
        style_fig(fig, theme, title, 230, "x", None)
        fig.update_yaxes(visible=False, range=[-0.3, 0.3])
        return fig
    if elems is not None:
        ex, ey = _edge_lines_2d(nodes, elems)
        fig.add_trace(go.Scattergl(x=ex, y=ey, mode="lines", line=dict(color=P["series"][0], width=0.8),
                                   opacity=0.75, hoverinfo="skip", name="arêtes"))
    elif grid is not None:
        gx, gy = _grid_lines(grid["xs"], grid["ys"])
        fig.add_trace(go.Scattergl(x=gx, y=gy, mode="lines", line=dict(color=P["series"][0], width=0.8),
                                   opacity=0.6, hoverinfo="skip", name="lignes de grille"))
    n = len(nodes)
    msize = 7 if n < 400 else (4 if n < 5000 else 2)
    for slot, cname in enumerate(("interior", "dirichlet", "neumann", "robin")):
        m = cats == cname
        if not np.any(m) or (not show_nodes and (cname == "interior" or elems is not None)):
            continue
        fig.add_trace(go.Scattergl(x=nodes[m, 0], y=nodes[m, 1], mode="markers", name=NODE_CATEGORY_LABELS[cname],
                                   marker=dict(size=msize, color=P["muted"] if cname == "interior" else P["series"][slot - 1]),
                                   hovertemplate="(%{x:.3f}, %{y:.3f})<extra>" + NODE_CATEGORY_LABELS[cname] + "</extra>"))
    if show_ids and n <= 150:
        fig.add_trace(go.Scatter(x=nodes[:, 0], y=nodes[:, 1], mode="text", text=[str(i) for i in range(n)],
                                 textposition="top right", textfont=dict(size=9, color=P["text2"]),
                                 hoverinfo="skip", showlegend=False))
        if elems is not None and len(elems) <= 200:
            c = nodes[elems].mean(axis=1)
            fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="text", text=[f"K{i}" for i in range(len(elems))],
                                     textfont=dict(size=8, color=P["accent2"]), hoverinfo="skip", showlegend=False))
    style_fig(fig, theme, title, height, "x", "y")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fig_spy(A: sp.spmatrix, theme: str, title: str, height: int = 430, max_points: int = 120_000) -> go.Figure:
    P = pal(theme)
    A = A.tocoo()
    n = A.shape[0]
    fig = go.Figure()
    if A.nnz <= max_points:
        fig.add_trace(go.Scattergl(x=A.col, y=A.row, mode="markers",
                                   marker=dict(size=max(1.5, min(7, 420 / max(n, 1))), color=P["series"][0],
                                               symbol="square"),
                                   hovertemplate="ligne %{y}, colonne %{x}<extra>non-zéro</extra>", name="non-zéros"))
        sub = ""
    else:
        nb = 200
        H, xe, ye = np.histogram2d(A.row, A.col, bins=nb, range=[[0, n], [0, n]])
        fig.add_trace(go.Heatmap(z=np.log10(H + 1), x=0.5 * (ye[1:] + ye[:-1]), y=0.5 * (xe[1:] + xe[:-1]),
                                 colorscale=P["seq"], showscale=True,
                                 colorbar=dict(title=dict(text="log₁₀(nnz+1)", font=dict(color=P["text2"])),
                                               tickfont=dict(color=P["text2"]), outlinewidth=0, thickness=12)))
        sub = " (densité des non-zéros par blocs)"
    style_fig(fig, theme, title + sub, height, "colonne j", "ligne i", legend=False)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


def fig_matrix_values(Ad: np.ndarray, theme: str, title: str, height: int = 420) -> go.Figure:
    P = pal(theme)
    vmax = float(np.max(np.abs(Ad))) or 1.0
    n = Ad.shape[0]
    text = [[("" if Ad[i, j] == 0 else f"{Ad[i, j]:.3g}") for j in range(Ad.shape[1])] for i in range(n)]
    fig = go.Figure(go.Heatmap(z=Ad, colorscale=P["div"], zmid=0, zmin=-vmax, zmax=vmax,
                               text=text if n <= 14 else None, texttemplate="%{text}" if n <= 14 else None,
                               textfont=dict(size=10, color=P["text"]),
                               colorbar=dict(outlinewidth=0, thickness=12, tickfont=dict(color=P["text2"])),
                               hovertemplate="A[%{y}, %{x}] = %{z:.5g}<extra></extra>"))
    style_fig(fig, theme, title, height, "j", "i", legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_loglog_convergence(rows: list[dict], keys: list[str], slopes: dict, theory: dict, theme: str,
                           title: str, height: int = 450) -> go.Figure:
    P = pal(theme)
    labels = {"l2": "‖e‖_L²", "h1": "|e|_H¹", "max": "‖e‖_∞"}
    fig = go.Figure()
    hs = np.array([r["h"] for r in rows])
    for i, k in enumerate(keys):
        es = np.array([r[k] for r in rows], dtype=float)
        lab = labels.get(k, k) + (f"  (pente {slopes[k]:.2f})" if k in slopes else "")
        fig.add_trace(go.Scatter(x=hs, y=es, mode="lines+markers", name=lab,
                                 line=dict(color=P["series"][i], width=2),
                                 marker=dict(size=9, color=P["series"][i], line=dict(width=2, color=P["surface"])),
                                 hovertemplate="h = %{x:.3e}<br>erreur = %{y:.3e}<extra>" + labels.get(k, k) + "</extra>"))
        p = (theory.get("orders") or {}).get(k)
        if p:
            ref = es[-1] * (hs / hs[-1]) ** p
            fig.add_trace(go.Scatter(x=hs, y=ref, mode="lines", name=f"référence ∝ h^{fmt(p, 3)}",
                                     line=dict(color=P["series"][i], width=1, dash="dot"), opacity=0.6,
                                     hoverinfo="skip"))
    style_fig(fig, theme, title, height, "pas du maillage h", "erreur")
    fig.update_xaxes(type="log", exponentformat="power", autorange="reversed")
    fig.update_yaxes(type="log", exponentformat="power")
    tidy_log_axis(fig, "y", np.array([r[k] for r in rows for k in keys], dtype=float))
    tidy_log_axis(fig, "x", hs)
    return fig


def fig_amplification(th: np.ndarray, items: list[tuple[str, np.ndarray]], theme: str, title: str,
                      show_real: bool = False, height: int = 420) -> go.Figure:
    P = pal(theme)
    fig = go.Figure()
    ymax = 1.15
    for i, (lab, G) in enumerate(items):
        y = np.real(G) if show_real else np.abs(G)
        ymax = max(ymax, float(np.nanmax(np.abs(y))) * 1.05 if np.all(np.isfinite(y)) else ymax)
        fig.add_trace(go.Scatter(x=th, y=y, mode="lines", name=lab, line=dict(color=P["series"][i % 8], width=2.2),
                                 hovertemplate="θ = %{x:.3f}<br>" + ("G" if show_real else "|G|") + " = %{y:.5f}<extra>" + lab + "</extra>"))
    ymax = min(ymax, 4.0)
    lo = -ymax if show_real else 0.0
    fig.add_hrect(y0=1.0, y1=ymax, fillcolor=P["band_bad"], line_width=0, layer="below")
    if show_real:
        fig.add_hrect(y0=lo, y1=-1.0, fillcolor=P["band_bad"], line_width=0, layer="below")
    fig.add_hline(y=1.0, line=dict(color=P["bad"], width=1))
    if show_real:
        fig.add_hline(y=-1.0, line=dict(color=P["bad"], width=1))
        fig.add_hline(y=0.0, line=dict(color=P["axis"], width=1))
    style_fig(fig, theme, title, height, "angle de phase θ = ξh (rad)", "G(θ)" if show_real else "|G(θ)|")
    fig.update_xaxes(tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ticktext=["−π", "−π/2", "0", "π/2", "π"])
    fig.update_yaxes(range=[lo, ymax])
    fig.add_annotation(x=0.99, y=1.0, xref="paper", yref="y", text="zone d'amplification |G| > 1", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=P["bad"], size=11))
    return fig


def fig_stability_region(vt: float | None, beta: float | None, z_pts: np.ndarray | None, theme: str, title: str,
                         vn_curve: np.ndarray | None = None, height: int = 460) -> go.Figure:
    """Région de stabilité absolue {z : |R(z)| ≤ 1} (convention U' = −μU, z = Δtμ) et spectre Δt·μᵢ."""
    P = pal(theme)
    fig = go.Figure()
    if vt is not None:
        zr = z_pts.real if z_pts is not None and len(z_pts) else np.array([0.0])
        xmax = max(4.5, float(np.max(zr)) * 1.15 if len(zr) else 4.5)
        xs = np.linspace(-0.5 * xmax, xmax, 260)
        zi = z_pts.imag if z_pts is not None and len(z_pts) else np.array([0.0])
        ym = max(2.5, float(np.max(np.abs(zi))) * 1.3)
        ys = np.linspace(-ym, ym, 220)
        XX, YY = np.meshgrid(xs, ys)
        R = np.abs(theta_R(XX + 1j * YY, vt))
        fig.add_trace(go.Contour(x=xs, y=ys, z=np.where(R <= 1, 1.0, 0.0), showscale=False,
                                 colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(12,163,12,0.20)"]],
                                 contours=dict(start=0.5, end=0.5, size=1, coloring="fill"), line=dict(width=0),
                                 hoverinfo="skip", name="région stable"))
        fig.add_trace(go.Contour(x=xs, y=ys, z=R, showscale=False, contours=dict(start=1, end=1, size=1, coloring="none"),
                                 line=dict(color=P["good"], width=2), hoverinfo="skip", name="|R(z)| = 1"))
        xl, yl = "Re(z),  z = Δt·μ", "Im(z)"
    else:
        xs = None
        xl, yl = "Re(w),  w = Δt²·μ", "Im(w)"
        bound = 4.0 / (1 - 4 * beta) if beta is not None and beta < 0.25 else None
        if bound:
            fig.add_vrect(x0=0, x1=bound, fillcolor=P["band_good"], line_width=0, layer="below")
            fig.add_vline(x=bound, line=dict(color=P["bad"], width=1.5))
            fig.add_annotation(x=bound, y=1, yref="paper", text=f"w = {fmt(bound)}", showarrow=False,
                               font=dict(color=P["bad"]), xanchor="left")
    if vn_curve is not None:
        fig.add_trace(go.Scatter(x=vn_curve.real, y=vn_curve.imag, mode="lines", name="symbole de von Neumann z(θ)",
                                 line=dict(color=P["series"][1], width=2)))
    if z_pts is not None and len(z_pts):
        fig.add_trace(go.Scattergl(x=np.real(z_pts), y=np.imag(z_pts), mode="markers", name="spectre discret Δt·μᵢ",
                                   marker=dict(size=6, color=P["series"][0], line=dict(width=1, color=P["surface"])),
                                   hovertemplate="Re = %{x:.4g}<br>Im = %{y:.4g}<extra>valeur propre</extra>"))
    style_fig(fig, theme, title, height, xl, yl)
    if xs is not None:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fig_spectrum(eigs: np.ndarray, exact: np.ndarray | None, theme: str, title: str, height: int = 400) -> go.Figure:
    P = pal(theme)
    ev = np.sort(np.real(eigs))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=np.arange(1, len(ev) + 1), y=ev, mode="markers", name="valeurs propres discrètes λ_h",
                               marker=dict(size=5, color=P["series"][0])))
    if exact is not None and len(exact):
        m = min(len(exact), len(ev))
        fig.add_trace(go.Scatter(x=np.arange(1, m + 1), y=exact[:m], mode="lines", name="valeurs propres exactes λ",
                                 line=dict(color=P["muted"], width=2)))
    style_fig(fig, theme, title, height, "indice m", "λ_m")
    fig.update_yaxes(type="log", exponentformat="power")
    return fig


def fig_stability_map(xv: np.ndarray, yv: np.ndarray, Z: np.ndarray, theme: str, title: str, xlabel: str,
                      ylabel: str, curves: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
                      point: tuple[float, float] | None = None, height: int = 430) -> go.Figure:
    P = pal(theme)
    Zc = np.clip(Z, 0, 2.0)
    stable = P["series"][2]
    fig = go.Figure(go.Heatmap(x=xv, y=yv, z=Zc, colorscale=[[0, stable], [0.5, stable], [0.50001, "#c0623f"], [1, P["bad"]]],
                               zmin=0, zmax=2, colorbar=dict(title=dict(text="max|G|", font=dict(color=P["text2"])),
                                                             tickfont=dict(color=P["text2"]), outlinewidth=0, thickness=12),
                               hovertemplate=xlabel + " = %{x:.3f}<br>" + ylabel + " = %{y:.3f}<br>max|G| = %{z:.4f}<extra></extra>"))
    fig.add_trace(go.Contour(x=xv, y=yv, z=Z, contours=dict(start=1 + 1e-9, end=1 + 1e-9, size=1, coloring="none"),
                             line=dict(color=P["text"], width=2), showscale=False, hoverinfo="skip", name="max|G| = 1"))
    for i, (lab, cx, cy) in enumerate(curves or []):
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", name=lab, line=dict(color=P["series"][(i + 3) % 8], width=2, dash="dot")))
    if point is not None:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers", name="paramètres actuels",
                                 marker=dict(size=13, color=P["accent2"], symbol="x", line=dict(width=2, color=P["surface"]))))
    style_fig(fig, theme, title, height, xlabel, ylabel)
    fig.update_xaxes(range=[xv.min(), xv.max()])
    fig.update_yaxes(range=[yv.min(), yv.max()])
    return fig


def fig_reference_triangle_functions(theme: str, height: int = 400) -> go.Figure:
    P = pal(theme)
    n = 14
    pts, tris = [], []
    idx = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = len(pts)
            pts.append((i / n, j / n))
    for i in range(n):
        for j in range(n - i):
            tris.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if j < n - i - 1:
                tris.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    pts = np.array(pts)
    tris = np.array(tris)
    lam = [1 - pts[:, 0] - pts[:, 1], pts[:, 0], pts[:, 1]]
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
                        subplot_titles=["λ₁ = 1 − ξ − η", "λ₂ = ξ", "λ₃ = η"], horizontal_spacing=0.02)
    for c in range(3):
        fig.add_trace(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=lam[c], i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                                intensity=lam[c], colorscale=P["seq"], showscale=False, flatshading=True,
                                hovertemplate="ξ=%{x:.2f}, η=%{y:.2f}<br>λ=%{z:.3f}<extra></extra>"), row=1, col=c + 1)
        verts = np.array([[0, 0], [1, 0], [0, 1]])
        fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1], z=[1.0 if k == c else 0.0 for k in range(3)],
                                   mode="markers+text", text=["A₁", "A₂", "A₃"], textposition="top center",
                                   marker=dict(size=4, color=P["accent2"]), textfont=dict(color=P["text"]),
                                   showlegend=False, hoverinfo="skip"), row=1, col=c + 1)
    style_fig(fig, theme, None, height, legend=False)
    cam = dict(eye=dict(x=1.25, y=-1.45, z=0.85))
    axis_ref = dict(showticklabels=False, title="", showgrid=True, gridcolor=P["grid"], showbackground=False)
    for s in ("scene", "scene2", "scene3"):
        fig.update_layout(**{s: dict(camera=cam, xaxis=dict(axis_ref, title="ξ"), yaxis=dict(axis_ref, title="η"),
                                     zaxis=dict(axis_ref, showticklabels=True, tickvals=[0, 0.5, 1], title=""),
                                     aspectratio=dict(x=1, y=1, z=0.75))})
    fig.update_annotations(font=dict(color=P["text"], size=13))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def fig_hat_functions(xn: np.ndarray, theme: str, highlight: int | None = None, coeffs: np.ndarray | None = None,
                      height: int = 360) -> go.Figure:
    P = pal(theme)
    fig = go.Figure()
    n = len(xn)
    show = range(n) if n <= 24 else [i for i in range(n) if i % max(1, n // 12) == 0]
    for i in show:
        y = np.zeros(n)
        y[i] = 1.0 if coeffs is None else coeffs[i]
        hl = (highlight == i)
        fig.add_trace(go.Scatter(x=xn, y=y, mode="lines", name=f"φ_{i}", showlegend=hl,
                                 line=dict(color=P["series"][1] if hl else P["series"][0], width=3 if hl else 1.3),
                                 opacity=1.0 if hl or highlight is None else 0.45,
                                 hovertemplate="x = %{x:.3f}<br>φ = %{y:.3f}<extra>φ_" + str(i) + "</extra>"))
    if coeffs is not None:
        fig.add_trace(go.Scatter(x=xn, y=coeffs, mode="lines+markers", name="u_h = Σ Uᵢ φᵢ",
                                 line=dict(color=P["text"], width=2.5), marker=dict(size=7)))
    style_fig(fig, theme, None, height, "x", "φᵢ(x)")
    return fig


def fig_stencil(kind: str, theme: str, height: int = 330) -> go.Figure:
    """Molécule de calcul (espace–temps ou 2D) du schéma."""
    P = pal(theme)
    stencils = {
        "lap1d": ([(-1, 0, "1"), (0, 0, "−2"), (1, 0, "1")], "j", "",
                  "Laplacien 1D : (U<sub>j−1</sub> − 2U<sub>j</sub> + U<sub>j+1</sub>) / h²"),
        "lap2d": ([(-1, 0, "−1"), (1, 0, "−1"), (0, -1, "−1"), (0, 1, "−1"), (0, 0, "4")], "i", "j",
                  "Laplacien 5 points : (4U<sub>ij</sub> − U<sub>i±1,j</sub> − U<sub>i,j±1</sub>) / h²"),
        "explicit": ([(-1, 0, "r"), (0, 0, "1−2r"), (1, 0, "r"), (0, 1, "Uⁿ⁺¹")], "j", "n",
                     "Euler explicite : U<sub>j</sub><sup>n+1</sup> = r U<sub>j−1</sub><sup>n</sup> + (1−2r) "
                     "U<sub>j</sub><sup>n</sup> + r U<sub>j+1</sub><sup>n</sup>"),
        "implicit": ([(-1, 1, "−r"), (0, 1, "1+2r"), (1, 1, "−r"), (0, 0, "Uⁿ")], "j", "n",
                     "Euler implicite : −r U<sub>j−1</sub><sup>n+1</sup> + (1+2r) U<sub>j</sub><sup>n+1</sup> "
                     "− r U<sub>j+1</sub><sup>n+1</sup> = U<sub>j</sub><sup>n</sup>"),
        "cn": ([(-1, 1, "−r/2"), (0, 1, "1+r"), (1, 1, "−r/2"), (-1, 0, "r/2"), (0, 0, "1−r"), (1, 0, "r/2")], "j", "n",
               "Crank–Nicolson : moyenne des niveaux n et n+1"),
        "leapfrog": ([(-1, 0, "σ²"), (0, 0, "2−2σ²"), (1, 0, "σ²"), (0, -1, "−1"), (0, 1, "Uⁿ⁺¹")], "j", "n",
                     "Ondes, saute-mouton : U<sub>j</sub><sup>n+1</sup> = 2U<sub>j</sub><sup>n</sup> − "
                     "U<sub>j</sub><sup>n−1</sup> + σ²(U<sub>j−1</sub><sup>n</sup> − 2U<sub>j</sub><sup>n</sup> + "
                     "U<sub>j+1</sub><sup>n</sup>)"),
        "upwind": ([(-1, 0, "λ"), (0, 0, "1−λ"), (0, 1, "Uⁿ⁺¹")], "j", "n", "Décentré amont (a > 0)"),
        "lax_wendroff": ([(-1, 0, "λ(1+λ)/2"), (0, 0, "1−λ²"), (1, 0, "−λ(1−λ)/2"), (0, 1, "Uⁿ⁺¹")], "j", "n", "Lax–Wendroff"),
        "lax_friedrichs": ([(-1, 0, "(1+λ)/2"), (1, 0, "(1−λ)/2"), (0, 1, "Uⁿ⁺¹")], "j", "n", "Lax–Friedrichs"),
        "ftcs": ([(-1, 0, "λ/2"), (0, 0, "1"), (1, 0, "−λ/2"), (0, 1, "Uⁿ⁺¹")], "j", "n", "Centré explicite (FTCS)"),
        "btcs": ([(-1, 1, "−λ/2"), (0, 1, "1"), (1, 1, "λ/2"), (0, 0, "Uⁿ")], "j", "n", "Centré implicite"),
    }
    pts, xl, yl, title = stencils.get(kind, stencils["lap1d"])
    fig = go.Figure()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    anchor = next(((p[0], p[1]) for p in pts if p[2].startswith("U")), (0, 0))
    for (x0, y0, _) in pts:
        if (x0, y0) != anchor:
            fig.add_trace(go.Scatter(x=[anchor[0], x0], y=[anchor[1], y0], mode="lines",
                                     line=dict(color=P["axis"], width=1), hoverinfo="skip", showlegend=False))
    colors = [P["series"][1] if p[2].startswith("U") else P["series"][0] for p in pts]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[p[2] for p in pts], textposition="top center",
                             marker=dict(size=22, color=colors, line=dict(width=2, color=P["surface"])),
                             textfont=dict(color=P["text"], size=13), hoverinfo="skip", showlegend=False))
    ylab = {"n": "niveau de temps", "j": "indice j"}.get(yl)
    style_fig(fig, theme, title, height, f"indice d'espace {xl}", ylab, legend=False)
    fig.update_xaxes(range=[-1.8, 1.8], tickvals=[-1, 0, 1], ticktext=[f"{xl}−1", xl, f"{xl}+1"], showgrid=False,
                     zeroline=False)
    if yl:
        lv = sorted(set(ys))
        fig.update_yaxes(range=[min(ys) - 0.7, max(ys) + 0.9], tickvals=lv,
                         ticktext=[{-1: f"{yl}−1", 0: yl, 1: f"{yl}+1"}[v] for v in lv], showgrid=False, zeroline=False)
    else:
        fig.update_yaxes(visible=False, range=[-1, 1])
    return fig


def fig_time_animation(x: np.ndarray, times: np.ndarray, frames: np.ndarray, exact: np.ndarray | None,
                       theme: str, title: str, height: int = 460) -> go.Figure:
    P = pal(theme)
    order = np.argsort(x)
    x = x[order]
    F = frames[:, order]
    E = exact[:, order] if exact is not None else None
    allv = F if E is None else np.concatenate([F, E])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = 0.08 * (hi - lo + 1e-12)
    data = []
    if E is not None:
        data.append(go.Scatter(x=x, y=E[0], mode="lines", name="solution exacte", line=dict(color=P["muted"], width=2.5)))
    data.append(go.Scatter(x=x, y=F[0], mode="lines", name="u_h", line=dict(color=P["series"][0], width=2.2)))
    frs = []
    for k in range(len(times)):
        d = []
        if E is not None:
            d.append(go.Scatter(x=x, y=E[k]))
        d.append(go.Scatter(x=x, y=F[k]))
        frs.append(go.Frame(data=d, name=f"{k}", layout=go.Layout(title_text=f"{title} — t = {times[k]:.4g}")))
    fig = go.Figure(data=data, frames=frs)
    style_fig(fig, theme, f"{title} — t = {times[0]:.4g}", height, "x", "u(x, t)")
    fig.update_yaxes(range=[lo - pad, hi + pad])
    steps = [dict(method="animate", args=[[f"{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                         transition=dict(duration=0))], label=f"{times[k]:.3g}")
             for k in range(len(times))]
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.2, xanchor="left", yanchor="top",
                          bgcolor=P["surface2"], bordercolor=P["border"], font=dict(color=P["text"]),
                          buttons=[dict(label="▶ Lecture", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=False), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, steps=steps, x=0.16, y=-0.13, len=0.84, xanchor="left", yanchor="top",
                      currentvalue=dict(prefix="t = ", font=dict(color=P["text2"])), font=dict(color=P["text2"]),
                      bgcolor=P["surface2"], bordercolor=P["border"], tickcolor=P["axis"])],
        margin=dict(b=110))
    return fig


def fig_spacetime(x: np.ndarray, times: np.ndarray, frames: np.ndarray, theme: str, title: str, zlabel: str = "u",
                  height: int = 430) -> go.Figure:
    P = pal(theme)
    order = np.argsort(x)
    Z = frames[:, order]
    cs, mid = colorscale_for(Z, theme)
    fig = go.Figure(go.Heatmap(x=x[order], y=times, z=Z, colorscale=cs, zmid=mid,
                               colorbar=dict(title=dict(text=zlabel, font=dict(color=P["text2"])),
                                             tickfont=dict(color=P["text2"]), outlinewidth=0, thickness=12),
                               hovertemplate="x = %{x:.3f}<br>t = %{y:.4g}<br>" + zlabel + " = %{z:.4g}<extra></extra>"))
    style_fig(fig, theme, title, height, "x", "t", legend=False)
    return fig


# =============================================================================
# 12. INTERFACE STREAMLIT — socle (thème, état, liaisons, cache, composants)
# =============================================================================

_MD_PROTECT = re.compile(r"(\$\$.*?\$\$|\$[^$\n]*\$|:material/[a-z0-9_]+:|`[^`]*`|<[^>]+>)", re.S)


def md_safe(text: Any) -> Any:
    """Échappe les « _ » d'un texte brut pour Markdown, sauf dans le LaTeX ($…$), les icônes, le code et le HTML."""
    if not isinstance(text, str) or "_" not in text:
        return text
    parts = _MD_PROTECT.split(text)
    return "".join(p if k % 2 else re.sub(r"(?<!\\)_", r"\\_", p) for k, p in enumerate(parts))


def ui_markdown(body: Any, *args: Any, **kwargs: Any):
    if not kwargs.get("unsafe_allow_html"):
        body = md_safe(body)
    return st.markdown(body, *args, **kwargs)


def ui_caption(body: Any, *args: Any, **kwargs: Any):
    if not kwargs.get("unsafe_allow_html"):
        body = md_safe(body)
    return st.caption(body, *args, **kwargs)


def ui_error(body: Any, *args: Any, **kwargs: Any):
    return st.error(md_safe(body), *args, **kwargs)


def ui_warning(body: Any, *args: Any, **kwargs: Any):
    return st.warning(md_safe(body), *args, **kwargs)


def ui_info(body: Any, *args: Any, **kwargs: Any):
    return st.info(md_safe(body), *args, **kwargs)


def ui_success(body: Any, *args: Any, **kwargs: Any):
    return st.success(md_safe(body), *args, **kwargs)
THEME_OPTIONS: dict[str, dict[str, Any]] = {
    "dark": {
        "theme.base": "dark", "theme.primaryColor": "#3b82f6", "theme.backgroundColor": "#0a1020",
        "theme.secondaryBackgroundColor": "#131f37", "theme.textColor": "#e2e9f5", "theme.borderColor": "#24324e",
        "theme.linkColor": "#60a5fa", "theme.codeBackgroundColor": "#0f182b",
        "theme.sidebar.backgroundColor": "#0c1426", "theme.sidebar.secondaryBackgroundColor": "#14203a",
        "theme.baseRadius": "0.55rem", "theme.showWidgetBorder": True,
    },
    "light": {
        "theme.base": "light", "theme.primaryColor": "#1d5fd1", "theme.backgroundColor": "#f5f7fb",
        "theme.secondaryBackgroundColor": "#e9eef6", "theme.textColor": "#111a2e", "theme.borderColor": "#d3dbe8",
        "theme.linkColor": "#1d5fd1", "theme.codeBackgroundColor": "#eef2f8",
        "theme.sidebar.backgroundColor": "#edf1f8", "theme.sidebar.secondaryBackgroundColor": "#e2e8f2",
        "theme.baseRadius": "0.55rem", "theme.showWidgetBorder": True,
    },
}
_THEME_MARK = "PDE Numerical Laboratory (main.py)"


def apply_runtime_theme(mode: str) -> bool:
    """Applique le thème via la configuration de Streamlit (sauf si l'utilisateur a son propre config.toml).
    Renvoie True si la configuration a changé (un nouveau rendu est alors nécessaire)."""
    try:
        from streamlit import config as _stcfg
        where = str(_stcfg.get_where_defined("theme.base"))
        if where not in ("<default>", _THEME_MARK):
            return False
        changed = False
        for key, val in THEME_OPTIONS[mode].items():
            if _stcfg.get_option(key) != val:
                _stcfg.set_option(key, val, where_defined=_THEME_MARK)
                changed = True
        return changed
    except Exception:  # noqa: BLE001 - repli : thème par défaut + CSS
        return False


def theme_mode() -> str:
    return st.session_state.get("theme_mode", "dark")


def inject_css(mode: str) -> None:
    P = pal(mode)
    dark = mode == "dark"
    accent_soft = "rgba(59,130,246,0.13)" if dark else "rgba(29,95,209,0.08)"
    accent_line = "rgba(96,165,250,0.40)" if dark else "rgba(29,95,209,0.30)"
    shadow = ("0 1px 0 rgba(255,255,255,0.025) inset, 0 10px 28px rgba(0,0,0,0.28)" if dark else
              "0 1px 2px rgba(15,30,60,0.05), 0 8px 24px rgba(15,30,60,0.06)")
    hero_bg = ("linear-gradient(180deg, #0b1630 0%, #0a1226 100%)" if dark else
               "linear-gradient(180deg, #e9f1fc 0%, #f4f8fd 100%)")
    css = f"""
<style>
:root {{
  --pdl-surface: {P['surface']}; --pdl-surface-2: {P['surface2']}; --pdl-border: {P['border']};
  --pdl-text: {P['text']}; --pdl-text-2: {P['text2']}; --pdl-muted: {P['muted']};
  --pdl-accent: {P['accent']}; --pdl-accent-2: {P['accent2']}; --pdl-good: {P['good']};
  --pdl-warn: {P['warn']}; --pdl-bad: {P['bad']}; --pdl-accent-soft: {accent_soft};
  --pdl-accent-line: {accent_line}; --pdl-shadow: {shadow};
}}
.block-container {{ padding-top: 0.9rem !important; padding-bottom: 3.5rem; max-width: 1520px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--pdl-border); }}
[data-testid="stSidebarNav"] a span {{ font-size: 0.93rem; }}
h1 {{ font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.2rem !important; font-weight: 650 !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; }}
h4 {{ font-size: 0.98rem !important; font-weight: 650 !important; }}
.pdl-header {{ display: flex; align-items: center; justify-content: space-between; gap: 18px;
  padding: 4px 2px 12px 2px; border-bottom: 1px solid var(--pdl-border); margin-bottom: 14px; }}
.pdl-brand {{ display: flex; align-items: center; gap: 14px; color: var(--pdl-accent); }}
.pdl-title {{ font-size: 1.42rem; font-weight: 700; color: var(--pdl-text); line-height: 1.15; letter-spacing: 0.01em; }}
.pdl-subtitle {{ font-size: 0.9rem; color: var(--pdl-text-2); font-style: italic; margin-top: 2px; }}
.pdl-pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px;
  border: 1px solid var(--pdl-border); background: var(--pdl-surface); color: var(--pdl-text-2);
  font-size: 0.82rem; white-space: nowrap; }}
.pdl-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.pdl-dot.ok {{ background: var(--pdl-good); box-shadow: 0 0 0 3px rgba(12,163,12,0.18); }}
.pdl-dot.warn {{ background: var(--pdl-warn); box-shadow: 0 0 0 3px rgba(250,178,25,0.18); }}
.pdl-dot.bad {{ background: var(--pdl-bad); box-shadow: 0 0 0 3px rgba(208,59,59,0.18); }}
.pdl-dot.idle {{ background: var(--pdl-accent); box-shadow: 0 0 0 3px var(--pdl-accent-soft); }}
.pdl-stepper {{ display: flex; gap: 6px; margin: 2px 0 16px 0; }}
.pdl-step {{ flex: 1; display: flex; align-items: center; gap: 10px; min-width: 0;
  padding: 9px 18px 9px 26px; background: var(--pdl-surface); color: var(--pdl-text-2); font-size: 0.9rem;
  clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%, 14px 50%); }}
.pdl-step:first-child {{ padding-left: 16px; clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%); }}
.pdl-step .num {{ width: 26px; height: 26px; flex: none; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; border: 1.5px solid currentColor; font-weight: 650; font-size: 0.85rem; }}
.pdl-step .lbl {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.pdl-step.done {{ color: var(--pdl-text); }}
.pdl-step.done .num {{ color: var(--pdl-good); border-color: var(--pdl-good); }}
.pdl-step.active {{ background: var(--pdl-accent); color: #ffffff; font-weight: 600; }}
.pdl-step.active .num {{ border-color: rgba(255,255,255,0.9); }}
div[class*="st-key-card_"] {{ background: var(--pdl-surface); border: 1px solid var(--pdl-border); border-radius: 14px;
  padding: 14px 18px 10px 18px; box-shadow: var(--pdl-shadow); }}
div[class*="st-key-card_"] h4, div[class*="st-key-card_"] h3 {{ padding-top: 0.1rem !important; margin-bottom: 0.2rem !important; }}
div[class*="st-key-pbox_"] {{ background: var(--pdl-accent-soft); border: 1px solid var(--pdl-accent-line);
  border-radius: 12px; padding: 10px 16px 6px 16px; }}
div[class*="st-key-run_"] button {{ min-height: 3.1rem; font-weight: 700; letter-spacing: 0.04em; }}
.pdl-kicker {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--pdl-accent-2);
  font-weight: 650; margin-bottom: 2px; }}
.pdl-pagetitle {{ font-size: 1.36rem; font-weight: 700; color: var(--pdl-text); margin: 0 0 4px 0; line-height: 1.2; }}
.pdl-lead {{ color: var(--pdl-text-2); font-size: 0.95rem; margin: 0 0 12px 0; max-width: 1050px; line-height: 1.5; }}
.pdl-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 650;
  border: 1px solid currentColor; white-space: nowrap; }}
.pdl-badge.green {{ color: {('#38c95a' if dark else '#127a12')}; background: rgba(12,163,12,0.10); }}
.pdl-badge.blue {{ color: {('#6aa8f5' if dark else '#1d5fd1')}; background: var(--pdl-accent-soft); }}
.pdl-badge.orange {{ color: {('#f0a63a' if dark else '#a35a00')}; background: rgba(250,178,25,0.10); }}
.pdl-badge.red {{ color: {('#f07070' if dark else '#b3261e')}; background: rgba(208,59,59,0.10); }}
.pdl-badge.gray {{ color: var(--pdl-text-2); background: var(--pdl-surface-2); }}
table.pdl-kv {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 2px 0 8px 0; }}
table.pdl-kv td {{ padding: 6px 8px; border-bottom: 1px solid var(--pdl-border); vertical-align: top; }}
table.pdl-kv td.k {{ color: var(--pdl-text-2); width: 44%; }}
table.pdl-kv td.v {{ color: var(--pdl-text); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem !important; }}
[data-testid="stMetricValue"] > div {{ overflow: hidden; text-overflow: ellipsis; }}
div[class*="st-key-pbox_"] .katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 2px 0; }}
.katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 3px 0 5px 0; }}
div[class*="st-key-card_"] .katex-display > .katex {{ font-size: 1.08em; }}
.pdl-hero {{ position: relative; border-radius: 16px; overflow: hidden; border: 1px solid var(--pdl-border);
  height: 158px; margin: 0 0 14px 0; background: {hero_bg}; }}
.pdl-hero svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.pdl-hero .txt {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 0 16px; }}
.pdl-hero .t1 {{ font-family: Georgia, 'Times New Roman', serif; font-style: italic; font-size: 1.75rem;
  color: var(--pdl-text); letter-spacing: 0.01em; }}
.pdl-hero .t2 {{ font-family: Georgia, 'Times New Roman', serif; font-style: italic; font-size: 1.02rem;
  color: var(--pdl-text-2); margin-top: 4px; }}
.pdl-side-card {{ border: 1px solid var(--pdl-border); border-radius: 12px; padding: 10px 12px; background: var(--pdl-surface);
  font-size: 0.82rem; color: var(--pdl-text-2); line-height: 1.45; }}
.pdl-side-card b {{ color: var(--pdl-text); }}
.pdl-prog {{ display: flex; flex-direction: column; gap: 4px; font-size: 0.84rem; margin: 2px 0 6px 0; }}
.pdl-prog .row {{ display: flex; align-items: center; gap: 8px; color: var(--pdl-text-2); }}
.pdl-prog .row.done {{ color: var(--pdl-text); }}
.pdl-prog .mk {{ width: 16px; height: 16px; border-radius: 50%; border: 1.5px solid var(--pdl-muted); flex: none;
  display: inline-flex; align-items: center; justify-content: center; }}
.pdl-prog .row.done .mk {{ border-color: var(--pdl-good); background: rgba(12,163,12,0.15); }}
.pdl-note {{ font-size: 0.85rem; color: var(--pdl-text-2); }}
@media (max-width: 1100px) {{ .pdl-step .lbl {{ display: none; }} .pdl-subtitle {{ display: none; }} }}
</style>
"""
    ui_markdown(css, unsafe_allow_html=True)


LOGO_SVG = """<svg width="48" height="44" viewBox="0 0 48 44" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<g fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round">
<polygon points="24,3 45,40 3,40"/><polyline points="13.5,21.5 34.5,21.5 24,40 13.5,21.5"/>
<line x1="18.75" y1="12.25" x2="29.25" y2="12.25"/><line x1="18.75" y1="12.25" x2="24" y2="21.5"/>
<line x1="29.25" y1="12.25" x2="24" y2="21.5"/><line x1="8.25" y1="30.75" x2="18.75" y2="30.75"/>
<line x1="29.25" y1="30.75" x2="39.75" y2="30.75"/><line x1="18.75" y1="30.75" x2="13.5" y2="40"/>
<line x1="29.25" y1="30.75" x2="34.5" y2="40"/></g>
<g fill="currentColor"><circle cx="24" cy="3" r="2"/><circle cx="45" cy="40" r="2"/><circle cx="3" cy="40" r="2"/>
<circle cx="13.5" cy="21.5" r="1.8"/><circle cx="34.5" cy="21.5" r="1.8"/><circle cx="24" cy="40" r="1.8"/>
<circle cx="24" cy="21.5" r="1.6"/></g></svg>"""

CHECK_SVG = ('<svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3.2L13 5" '
             'fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def hero_svg(mode: str) -> str:
    """Bandeau : superposition de modes de Fourier (ondes stationnaires) — purement décoratif mais calculé."""
    P = pal(mode)
    W, H = 1400, 160
    xs = np.linspace(0, W, 240)
    paths = []
    rng = np.random.default_rng(3)
    for k in range(14):
        m = 1 + k % 5
        amp = 18 + 22 * rng.random()
        ph = 2 * np.pi * rng.random()
        y = H / 2 + amp * np.sin(2 * np.pi * m * xs / W + ph) * np.cos(np.pi * xs / W * (k % 3 + 1)) \
            + 10 * np.sin(6 * np.pi * xs / W + k)
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, y))
        col = P["accent"] if k % 2 == 0 else P["accent2"]
        op = 0.10 + 0.22 * rng.random()
        paths.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.2" stroke-opacity="{op:.2f}"/>')
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            + "".join(paths) + "</svg>")


# ------------------------------------------------------------------ état
def init_state() -> None:
    s = st.session_state
    if "cfg" not in s:
        s.cfg = default_config()
    s.setdefault("theme_mode", "dark")
    s.setdefault("results", {})
    s.setdefault("flash", [])


def cfg_raw() -> dict:
    return st.session_state.cfg


def cfg_norm() -> dict:
    return normalize_config(st.session_state.cfg)[0]


def cfg_json_cur() -> str:
    return config_json(cfg_norm())


def cur_hash() -> str:
    return config_hash(cfg_norm())


def store_result(kind: str, data: Any, key: str = "") -> None:
    st.session_state.results[kind] = {"hash": cur_hash(), "key": key, "data": data, "time": time.time()}


def get_result(kind: str, key: str | None = None, allow_stale: bool = False) -> Any:
    r = st.session_state.results.get(kind)
    if r is None:
        return None
    if allow_stale or (r["hash"] == cur_hash() and (key is None or r["key"] == key)):
        return r["data"]
    return None


def result_is_stale(kind: str) -> bool:
    r = st.session_state.results.get(kind)
    return r is not None and r["hash"] != cur_hash()


def set_config(new_cfg: dict, message: str | None = None) -> None:
    st.session_state.cfg = normalize_config(new_cfg)[0]
    purge_widgets()
    if message:
        st.session_state.flash.append(message)


def purge_widgets() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("w::"):
            del st.session_state[k]


def _get_path(d: dict, path: str) -> Any:
    for p in path.split("."):
        d = d[p]
    return d


def _set_path(d: dict, path: str, val: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = val


def _on_change(wkey: str, path: str, structural: bool) -> None:
    val = st.session_state.get(wkey)
    if val is None:
        return
    _set_path(st.session_state.cfg, path, val)
    if structural:
        newcfg, notes = normalize_config(st.session_state.cfg)
        st.session_state.cfg = newcfg
        st.session_state.flash.extend(notes)
        purge_widgets()


def _wk(path: str, scope: str) -> str:
    return f"w::{scope}::{path}"


def w_text(label: str, path: str, scope: str, help: str | None = None, structural: bool = False,
           placeholder: str | None = None, disabled: bool = False, label_visibility: str = "visible") -> str:
    key = _wk(path, scope)
    return st.text_input(label, value=str(_get_path(cfg_raw(), path)), key=key, help=help, placeholder=placeholder,
                         on_change=_on_change, args=(key, path, structural), disabled=disabled,
                         label_visibility=label_visibility)


def w_number(label: str, path: str, scope: str, lo: float, hi: float, step: float, is_int: bool = False,
             fmt_: str | None = None, help: str | None = None, structural: bool = False, disabled: bool = False) -> Any:
    key = _wk(path, scope)
    cur = _get_path(cfg_raw(), path)
    try:
        cur = int(round(float(cur))) if is_int else float(cur)
    except (TypeError, ValueError):
        cur = lo
    cur = min(max(cur, lo), hi)
    if is_int:
        return st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(cur), step=int(step), key=key,
                               help=help, on_change=_on_change, args=(key, path, structural), disabled=disabled)
    return st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(cur), step=float(step),
                           format=fmt_ or "%.4g", key=key, help=help, on_change=_on_change,
                           args=(key, path, structural), disabled=disabled)


def w_select(label: str, path: str, scope: str, options: list, labels: dict | None = None, help: str | None = None,
             structural: bool = False, disabled: bool = False, horizontal_radio: bool = False) -> Any:
    key = _wk(path, scope)
    cur = _get_path(cfg_raw(), path)
    idx = options.index(cur) if cur in options else 0
    ff = (lambda k: labels.get(k, str(k))) if labels else str
    if horizontal_radio:
        return st.radio(label, options, index=idx, format_func=ff, key=key, help=help, horizontal=True,
                        on_change=_on_change, args=(key, path, structural), disabled=disabled)
    return st.selectbox(label, options, index=idx, format_func=ff, key=key, help=help,
                        on_change=_on_change, args=(key, path, structural), disabled=disabled)


def w_toggle(label: str, path: str, scope: str, help: str | None = None, structural: bool = False) -> bool:
    key = _wk(path, scope)
    return st.toggle(label, value=bool(_get_path(cfg_raw(), path)), key=key, help=help, on_change=_on_change,
                     args=(key, path, structural))


def w_slider(label: str, path: str, scope: str, lo: float, hi: float, step: float, help: str | None = None,
             structural: bool = False, fmt_: str | None = None) -> Any:
    key = _wk(path, scope)
    cur = float(_get_path(cfg_raw(), path))
    return st.slider(label, min_value=float(lo), max_value=float(hi), value=min(max(cur, lo), hi), step=float(step),
                     key=key, help=help, on_change=_on_change, args=(key, path, structural), format=fmt_)


# ------------------------------------------------------------------ cache
@st.cache_resource(max_entries=48, show_spinner=False)
def problem_cached(cfg_json: str) -> Problem:
    return Problem(json.loads(cfg_json))


def current_problem() -> Problem:
    return problem_cached(cfg_json_cur())


@st.cache_resource(max_entries=32, show_spinner=False)
def cached_classification(cfg_json: str) -> dict:
    return classify_problem(problem_cached(cfg_json))


@st.cache_resource(max_entries=32, show_spinner=False)
def cached_strong(cfg_json: str) -> dict:
    return strong_formulation(problem_cached(cfg_json))


@st.cache_resource(max_entries=32, show_spinner=False)
def cached_weak(cfg_json: str) -> dict:
    return weak_formulation(problem_cached(cfg_json))


@st.cache_resource(max_entries=32, show_spinner=False)
def cached_lax_milgram(cfg_json: str) -> dict:
    return lax_milgram_analysis(problem_cached(cfg_json))


@st.cache_resource(max_entries=32, show_spinner=False)
def cached_exact_check(cfg_json: str) -> dict | None:
    return problem_cached(cfg_json).exact_residuals()


@st.cache_data(max_entries=512, show_spinner=False)
def expr_check(src: str, allowed: tuple[str, ...]) -> tuple[bool, str]:
    try:
        return True, sympy.latex(parse_expression(src, allowed))
    except ExpressionError as exc:
        return False, str(exc)


@st.cache_data(max_entries=12, show_spinner=False)
def cached_solve(cfg_json: str) -> dict:
    return run_solve(json.loads(cfg_json))


@st.cache_data(max_entries=8, show_spinner=False)
def cached_assembly(cfg_json: str) -> dict:
    prob = Problem(json.loads(cfg_json))
    if not prob.ok:
        return {"ok": False, "errors": prob.errors}
    if prob.cls == "transport":
        return {"ok": False, "errors": ["Transport : schémas explicites/implicites sur grille 1D (pas d'assemblage "
                                        "elliptique) — voir les pages Discrétisation et Stabilité."]}
    try:
        sysm = build_system(prob, prob.cfg)
    except ConfigError as exc:
        return {"ok": False, "errors": [str(exc)]}
    out: dict[str, Any] = {"ok": True, "method": sysm.method, "dim": sysm.dim, "nodes": sysm.nodes, "n": sysm.n,
                           "K": sysm.K, "M": sysm.M, "M_cons": sysm.M_cons, "F": sysm.load(0.0),
                           "free": sysm.free, "dir_idx": sysm.dir_idx,
                           "dir_vals": sysm.dir_val(0.0) if len(sysm.dir_idx) else np.zeros(0),
                           "components": sysm.components, "info": sysm.info, "kernel": sysm.kernel, "W": sysm.W,
                           "h": mesh_size(sysm), "hmin": mesh_hmin(sysm)}
    if sysm.mesh is not None:
        out["elems"] = sysm.mesh.elems
        out["quality"] = sysm.mesh.quality()
        out["side_nodes"] = {s: sysm.mesh.side_nodes(s) for s in prob.sides}
        out["grid"] = None
        out["mesh_meta"] = sysm.mesh.meta
    else:
        g = sysm.grid
        out["elems"] = None
        out["grid"] = {"xs": g["xs"], "ys": g["ys"], "hx": g["hx"], "hy": g["hy"], "nx": g["nx"], "ny": g["ny"]}
        out["side_nodes"] = {s: np.nonzero(g["side_mask"][s])[0] for s in prob.sides}
        out["quality"] = {"h_max": max(g["hx"], g["hy"] if sysm.dim == 2 else 0), "h_min": min(g["hx"], g["hy"] if sysm.dim == 2 else g["hx"])}
    return out


@st.cache_data(max_entries=8, show_spinner=False)
def cached_matrix_props(cfg_json: str, which: str) -> dict:
    asm = cached_assembly(cfg_json)
    if not asm.get("ok"):
        return {}
    Fr = asm["free"]
    A = asm["K"] if which == "K" else asm["M"]
    if asm["kernel"] and which == "K":
        return matrix_properties(A, spd_check=True) | {"note": "sans élimination (problème de Neumann pur)"}
    return matrix_properties(A[Fr][:, Fr], spd_check=True)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_spectral(cfg_json: str, dt: float | None) -> dict:
    prob = Problem(json.loads(cfg_json))
    return spectral_analysis(prob, prob.cfg, dt)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_convergence(cfg_json: str, levels: int, n0: int | None, strategy: str | None) -> dict:
    return convergence_study(json.loads(cfg_json), levels, n0, strategy)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_temporal(cfg_json: str, levels: int) -> dict:
    return temporal_convergence(json.loads(cfg_json), levels)


@st.cache_data(max_entries=1, show_spinner=False)
def cached_validation(stamp: int) -> list[dict]:
    _ = stamp
    return run_validation_suite()


def time_plan_current(prob: Problem) -> dict | None:
    """Pas de temps et grandeurs de stabilité (sans résoudre), pour les pages d'analyse."""
    if not prob.ok or not prob.time_dependent:
        return None
    try:
        if prob.cls == "transport":
            nx = prob.cfg["mesh"]["nx"]
            h = (prob.xb - prob.xa) / nx
            plan = time_step_plan(prob, prob.cfg, h, nx + 1)
            lam = (prob.a or 1.0) * plan["dt"] / h
            return {"dt": plan["dt"], "nt": plan["nt"], "numbers": dict(plan, λ=lam), "h": h, "hmin": h,
                    "grid": {"xs": prob.xa + h * np.arange(nx + 1), "ys": np.array([0.0])}, "method": "fdm"}
        asm = cached_assembly(cfg_json_cur())
        if not asm.get("ok"):
            return None
        plan = time_step_plan(prob, prob.cfg, asm["hmin"], len(asm["free"]))
        return {"dt": plan["dt"], "nt": plan["nt"], "numbers": plan, "h": asm["h"], "hmin": asm["hmin"],
                "grid": asm.get("grid"), "method": asm["method"]}
    except ConfigError as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ composants
def page_header(kicker: str, title: str, lead: str | None = None) -> None:
    ui_markdown(f'<div class="pdl-kicker">{kicker}</div><div class="pdl-pagetitle">{title}</div>'
                + (f'<div class="pdl-lead">{lead}</div>' if lead else ""), unsafe_allow_html=True)


def card(key: str):
    return st.container(key=f"card_{key}")


def card_title(icon: str, text: str) -> None:
    ui_markdown(f"#### :material/{icon}: {text}")


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="pdl-badge {color}">{text}</span>'


def status_badge(status: str) -> str:
    lab, col = STATUS_META.get(status, (status, "gray"))
    return badge(lab, col)


def kv_table(rows: list[tuple[str, Any]]) -> None:
    import html as _html
    body = "".join(f'<tr><td class="k">{_html.escape(str(k))}</td><td class="v">{_html.escape(str(v))}</td></tr>'
                   for k, v in rows)
    ui_markdown(f'<table class="pdl-kv">{body}</table>', unsafe_allow_html=True)


def show_fig(fig: go.Figure, explain: str | None = None, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, width="stretch")
    if explain:
        ui_caption(explain)


def flash_messages() -> None:
    msgs = st.session_state.get("flash") or []
    for m in msgs:
        st.toast(md_safe(m), icon=":material/info:")
    st.session_state.flash = []


def expr_feedback(src: str, allowed: tuple[str, ...]) -> None:
    ok, info = expr_check(str(src), allowed)
    if ok:
        ui_caption(f":material/check_circle: ${info}$" if len(info) < 160 else ":material/check_circle: expression valide")
    else:
        ui_caption(f":red[:material/error: {info}]")


def df_numbers(rows: list[dict], fmt_cols: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c, f in fmt_cols.items():
        if c in df:
            df[c] = df[c].map(lambda v, f=f: "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v))
    return df


def need_valid_problem() -> Problem | None:
    prob = current_problem()
    if not prob.ok:
        ui_error("La définition du problème contient des erreurs : corrigez-les dans « Définition de l'EDP ».",
                 icon=":material/error:")
        for e in prob.errors:
            ui_markdown(f"- {e}")
        return None
    return prob


def nav_button(page_key: str, label: str, icon: str = ":material/arrow_forward:", primary: bool = False,
               key: str | None = None) -> None:
    pages = st.session_state.get("_pages", {})
    if page_key in pages:
        if st.button(label, icon=icon, type="primary" if primary else "secondary", key=key or f"nav_{page_key}_{label}"):
            st.switch_page(pages[page_key])


def page_link(page_key: str, label: str | None = None, icon: str | None = None) -> None:
    pages = st.session_state.get("_pages", {})
    if page_key in pages:
        st.page_link(pages[page_key], label=label, icon=icon)


def next_step(page_key: str, label: str) -> None:
    st.divider()
    with st.container(horizontal=True, horizontal_alignment="right"):
        nav_button(page_key, f"Étape suivante : {label}", primary=True, key=f"next_{page_key}")


def ui_problem_box(prob: Problem, key: str) -> None:
    """Encadré « Problème sélectionné » : formulation forte compacte."""
    with st.container(key=f"pbox_{key}"):
        ui_markdown(":material/lightbulb: **Problème sélectionné** — trouver $u$ tel que")
        lines = [pde_latex(prob, False).replace(r"\quad\text{dans }\Omega\times(0,T]", "")
                 .replace(r"\quad\text{dans }\Omega", "")]
        if prob.cls == "transport":
            lines.append(r"u(x,0) = " + _expr_lx(prob.u0_expr))
        else:
            for s in prob.sides[:4]:
                lines.append(bc_latex(prob, s) if prob.dim == 1 else bc_latex(prob, s).replace(r"\quad\text{sur }", r"\ \text{sur } "))
            if prob.time_dependent:
                lines.append(r"u(\cdot,0) = " + _expr_lx(prob.u0_expr))
        body = r"\small\begin{cases}" + r"\\ ".join(lines) + r"\end{cases}"
        st.latex(body)


# =============================================================================
# 12b. CATALOGUE D'EXEMPLES
# =============================================================================
_LSHAPE_U = "(x^2+y^2)^(1/3)*sin(2/3*(atan2(-y,-x)+pi))"
_SIDES4 = ("left", "right", "bottom", "top")


def _bc_all(typ: str, g: str = "0", beta: str = "1") -> dict:
    return {s: {"type": typ, "g": g, "beta": beta} for s in _SIDES4}


EXAMPLES: list[dict[str, Any]] = [
    {"id": "poisson1d", "title": "Poisson 1D — Test 1", "group": "Elliptique",
     "desc": "−u″ = π² sin(πx) sur ]0, 1[, u(0) = u(1) = 0. Solution exacte u = sin(πx).",
     "cfg": dict(name="Poisson 1D — Test 1", family="poisson", dim=1, domain={"shape": "interval", "xa": 0, "xb": 1},
                 f="pi^2*sin(pi*x)", exact="sin(pi*x)", manufactured=False, method="fdm", mesh={"nx": 32},
                 bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "0"}})},
    {"id": "poisson2d", "title": "Poisson 2D — Test 2", "group": "Elliptique",
     "desc": "−Δu = 2π² sin(πx) sin(πy) sur ]0,1[², Dirichlet homogène. u = sin(πx) sin(πy).",
     "cfg": dict(name="Poisson 2D — Test 2", family="poisson", dim=2, f="2*pi^2*sin(pi*x)*sin(pi*y)",
                 exact="sin(pi*x)*sin(pi*y)", manufactured=False, method="fem", mesh={"nx": 16, "ny": 16},
                 bc=_bc_all("dirichlet"))},
    {"id": "laplace2d", "title": "Laplace 2D — Dirichlet non homogène", "group": "Elliptique",
     "desc": "−Δu = 0, u = sinh(πx) sin(πy)/sinh(π) imposée au bord : espace de recherche affine (relèvement).",
     "cfg": dict(name="Laplace 2D", family="laplace", dim=2, exact="sinh(pi*x)*sin(pi*y)/sinh(pi)", method="fdm",
                 mesh={"nx": 32, "ny": 32}, bc=_bc_all("dirichlet", "sinh(pi*x)*sin(pi*y)/sinh(pi)"))},
    {"id": "dr2d", "title": "Diffusion–réaction 2D, conditions mixtes", "group": "Elliptique",
     "desc": "k = 1 + x² + y², c = 2, Dirichlet/Neumann/Robin ; solution manufacturée u = sin(πx) eʸ.",
     "cfg": dict(name="Diffusion–réaction 2D mixte", family="diffusion_reaction", dim=2,
                 coef={"k": "1+x^2+y^2", "c": "2"}, exact="sin(pi*x)*exp(y)", manufactured=True, method="fem",
                 mesh={"nx": 16, "ny": 16},
                 bc={"left": {"type": "dirichlet"}, "right": {"type": "neumann"},
                     "bottom": {"type": "robin", "beta": "1"}, "top": {"type": "dirichlet"}})},
    {"id": "robin1d", "title": "Diffusion–réaction 1D, Neumann + Robin", "group": "Elliptique",
     "desc": "k = 1 + x, c = 1 ; Neumann à gauche, Robin à droite ; u = cos(2x) + x (manufacturée).",
     "cfg": dict(name="Diffusion–réaction 1D N/R", family="diffusion_reaction", dim=1,
                 domain={"shape": "interval", "xa": 0, "xb": 1}, coef={"k": "1+x", "c": "1"}, exact="cos(2*x)+x",
                 manufactured=True, method="fdm", mesh={"nx": 32},
                 bc={"left": {"type": "neumann"}, "right": {"type": "robin", "beta": "2"}})},
    {"id": "neumann2d", "title": "Neumann pur 2D (compatibilité)", "group": "Elliptique",
     "desc": "−Δu = 2π² cos(πx) cos(πy), ∂u/∂n = 0 : noyau = constantes, condition ∫f = 0, solution à moyenne nulle.",
     "cfg": dict(name="Neumann pur 2D", family="poisson", dim=2, f="2*pi^2*cos(pi*x)*cos(pi*y)",
                 exact="cos(pi*x)*cos(pi*y)", manufactured=False, method="fem", mesh={"nx": 16, "ny": 16},
                 bc=_bc_all("neumann"))},
    {"id": "lshape", "title": "Laplace sur domaine en L (singularité)", "group": "Elliptique",
     "desc": "u = r^(2/3) sin(2θ/3) : coin rentrant, u ∉ H² ⇒ ordre EF réduit (H¹ ≈ 2/3). Éléments finis uniquement.",
     "cfg": dict(name="Domaine en L", family="laplace", dim=2, domain={"shape": "lshape"}, method="fem",
                 exact=_LSHAPE_U, mesh={"nx": 16, "ny": 16}, bc={"boundary": {"type": "dirichlet", "g": _LSHAPE_U}})},
    {"id": "heat1d", "title": "Chaleur 1D — Test 3 (Crank–Nicolson)", "group": "Parabolique",
     "desc": "u_t − u_xx = 0, u₀ = sin(πx) ; u = e^{−π²t} sin(πx). Schéma de Crank–Nicolson, r = 2.",
     "cfg": dict(name="Chaleur 1D — Test 3", family="heat", dim=1, domain={"shape": "interval", "xa": 0, "xb": 1},
                 coef={"kappa": "1"}, f="0", u0="sin(pi*x)", exact="exp(-pi^2*t)*sin(pi*x)", manufactured=False,
                 method="fdm", mesh={"nx": 32},
                 bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "0"}},
                 time={"T": 0.1, "theta": 0.5, "dt_mode": "number", "number": 2.0})},
    {"id": "heat1d_unstable", "title": "Chaleur 1D — Euler explicite instable (r = 0,55)", "group": "Parabolique",
     "desc": "Même problème avec Euler explicite et r = 0,55 > 1/2 : le mode θ = π est amplifié par |1 − 4r| = 1,2.",
     "cfg": dict(name="Chaleur 1D explicite instable", family="heat", dim=1, domain={"shape": "interval"},
                 coef={"kappa": "1"}, f="0", u0="sin(pi*x)", exact="exp(-pi^2*t)*sin(pi*x)", method="fdm",
                 mesh={"nx": 40}, bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "0"}},
                 time={"T": 0.25, "theta": 0.0, "dt_mode": "number", "number": 0.55})},
    {"id": "heat2d", "title": "Chaleur 2D (éléments finis, CN)", "group": "Parabolique",
     "desc": "u_t − Δu = 0, u = e^{−2π²t} sin(πx) sin(πy) ; M dU/dt + KU = 0, masse cohérente.",
     "cfg": dict(name="Chaleur 2D", family="heat", dim=2, coef={"kappa": "1"}, f="0", u0="sin(pi*x)*sin(pi*y)",
                 exact="exp(-2*pi^2*t)*sin(pi*x)*sin(pi*y)", method="fem", mesh={"nx": 16, "ny": 16},
                 bc=_bc_all("dirichlet"), time={"T": 0.05, "theta": 0.5, "dt_mode": "nt", "nt": 50})},
    {"id": "pdr2d", "title": "Diffusion–réaction instationnaire 2D (Robin)", "group": "Parabolique",
     "desc": "k = 1 + x/2, c = 1, Neumann/Dirichlet/Robin ; u = e^{−t} cos(πx/2)(1 + y²) manufacturée.",
     "cfg": dict(name="Diffusion–réaction 2D instationnaire", family="parabolic_dr", dim=2,
                 coef={"k": "1+x/2", "c": "1"}, exact="exp(-t)*cos(pi*x/2)*(1+y^2)", manufactured=True,
                 method="fdm", mesh={"nx": 24, "ny": 24},
                 bc={"left": {"type": "neumann"}, "right": {"type": "dirichlet"}, "bottom": {"type": "dirichlet"},
                     "top": {"type": "robin", "beta": "2"}},
                 time={"T": 0.2, "theta": 0.5, "dt_mode": "number", "number": 1.0})},
    {"id": "advdiff1d", "title": "Advection–diffusion 1D (Dirichlet dépendant de t)", "group": "Parabolique",
     "desc": "k = 0,1, b = 1 ; u = e^{−0,1π²t} sin(π(x − t)) : données au bord dépendant du temps.",
     "cfg": dict(name="Advection–diffusion 1D", family="advection_diffusion", dim=1, domain={"shape": "interval"},
                 coef={"k": "0.1", "bx": "1"}, exact="exp(-0.1*pi^2*t)*sin(pi*(x-t))", manufactured=True,
                 method="fdm", mesh={"nx": 64}, time={"T": 0.5, "theta": 0.5, "dt_mode": "dt", "dt": 0.005})},
    {"id": "advdiff_peclet", "title": "Advection dominante — Péclet de maille > 1", "group": "Parabolique",
     "desc": "k = 0,005, b = 1, u(0) = 0, u(1) = 1 : couche limite ; schéma centré avec Pe_h = 2,5 (oscillations).",
     "cfg": dict(name="Couche limite (Péclet)", family="advection_diffusion", dim=1, domain={"shape": "interval"},
                 coef={"k": "0.005", "bx": "1"}, f="0", u0="x", exact="", method="fdm", mesh={"nx": 40},
                 adv_scheme="centered",
                 bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "1"}},
                 time={"T": 2.0, "theta": 0.5, "dt_mode": "dt", "dt": 0.01})},
    {"id": "advdiff2d", "title": "Advection–diffusion 2D — panache", "group": "Parabolique",
     "desc": "Bosse gaussienne transportée par b = (1 ; 0,4) et diffusée (k = 0,01). Éléments finis, CN.",
     "cfg": dict(name="Panache 2D", family="advection_diffusion", dim=2, coef={"k": "0.01", "bx": "1", "by": "0.4"},
                 f="0", u0="exp(-80*((x-0.25)^2+(y-0.35)^2))", exact="", method="fem", mesh={"nx": 32, "ny": 32},
                 bc=_bc_all("dirichlet"), time={"T": 0.5, "theta": 0.5, "dt_mode": "nt", "nt": 100})},
    {"id": "wave1d", "title": "Ondes 1D — Test 5", "group": "Hyperbolique",
     "desc": "u_tt − u_xx = 0, u₀ = sin(πx), v₀ = 0 ; u = sin(πx) cos(πt). Saute-mouton, σ = 0,9.",
     "cfg": dict(name="Ondes 1D — Test 5", family="wave", dim=1, domain={"shape": "interval"}, coef={"cw": "1"},
                 f="0", u0="sin(pi*x)", v0="0", exact="sin(pi*x)*cos(pi*t)", method="fdm", mesh={"nx": 64},
                 bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "0"}},
                 time={"T": 2.0, "wave_scheme": "leapfrog", "dt_mode": "number", "number": 0.9})},
    {"id": "wave1d_pluck", "title": "Corde pincée (donnée triangulaire)", "group": "Hyperbolique",
     "desc": "u₀ triangulaire (non C¹) : séparation en deux ondes et réflexions aux extrémités fixes.",
     "cfg": dict(name="Corde pincée", family="wave", dim=1, domain={"shape": "interval"}, coef={"cw": "1"}, f="0",
                 u0="max(0, 1-abs(x-0.3)/0.15)", v0="0", exact="", method="fdm", mesh={"nx": 200},
                 bc={"left": {"type": "dirichlet", "g": "0"}, "right": {"type": "dirichlet", "g": "0"}},
                 time={"T": 2.0, "wave_scheme": "leapfrog", "dt_mode": "number", "number": 0.95, "frames": 100})},
    {"id": "wave2d", "title": "Membrane vibrante 2D", "group": "Hyperbolique",
     "desc": "u = sin(πx) sin(πy) cos(√2 πt) ; EF P1 masse condensée, saute-mouton explicite.",
     "cfg": dict(name="Membrane 2D", family="wave", dim=2, coef={"cw": "1"}, f="0", u0="sin(pi*x)*sin(pi*y)", v0="0",
                 exact="sin(pi*x)*sin(pi*y)*cos(sqrt(2)*pi*t)", method="fem", mesh={"nx": 24, "ny": 24},
                 bc=_bc_all("dirichlet"),
                 time={"T": 1.0, "wave_scheme": "leapfrog", "dt_mode": "number", "number": 0.5, "lumped": True})},
    {"id": "wave2d_pulse", "title": "Propagation d'une impulsion 2D", "group": "Hyperbolique",
     "desc": "Impulsion gaussienne : front circulaire de célérité c, réflexion sur les bords fixes.",
     "cfg": dict(name="Impulsion 2D", family="wave", dim=2, coef={"cw": "1"}, f="0",
                 u0="exp(-150*((x-0.5)^2+(y-0.5)^2))", v0="0", exact="", method="fdm", mesh={"nx": 80, "ny": 80},
                 bc=_bc_all("dirichlet"), time={"T": 0.6, "wave_scheme": "leapfrog", "dt_mode": "number",
                                                "number": 0.6, "frames": 60})},
    {"id": "transport_sin", "title": "Transport 1D — Test 4 (sinusoïde)", "group": "Transport",
     "desc": "u_t + u_x = 0 périodique, u₀ = sin(2πx) ; solution exacte translatée u₀(x − t). Décentré amont.",
     "cfg": dict(name="Transport 1D — Test 4", family="transport", dim=1, domain={"shape": "interval"},
                 coef={"a": "1"}, u0="sin(2*pi*x)", exact="", transport_bc="periodic", mesh={"nx": 64},
                 time={"T": 1.0, "transport_scheme": "upwind", "dt_mode": "number", "number": 0.8})},
    {"id": "transport_square", "title": "Transport d'un créneau (dissipation / dispersion)", "group": "Transport",
     "desc": "Créneau discontinu, Lax–Wendroff : oscillations dispersives ; comparer avec le décentré (diffusif).",
     "cfg": dict(name="Transport d'un créneau", family="transport", dim=1, domain={"shape": "interval"},
                 coef={"a": "1"}, u0="heaviside(x-0.2)-heaviside(x-0.4)", exact="", transport_bc="periodic",
                 mesh={"nx": 200}, time={"T": 1.0, "transport_scheme": "lax_wendroff", "dt_mode": "number",
                                         "number": 0.8, "frames": 80})},
    {"id": "transport_ftcs", "title": "Schéma centré explicite (FTCS) instable", "group": "Transport",
     "desc": "|G|² = 1 + λ² sin²θ > 1 : instabilité inconditionnelle, visible après quelques centaines de pas.",
     "cfg": dict(name="FTCS instable", family="transport", dim=1, domain={"shape": "interval"}, coef={"a": "1"},
                 u0="sin(2*pi*x)", exact="", transport_bc="periodic", mesh={"nx": 100},
                 time={"T": 2.0, "transport_scheme": "ftcs", "dt_mode": "number", "number": 0.5})},
    {"id": "transport_inflow", "title": "Transport avec condition d'entrée", "group": "Transport",
     "desc": "Impulsion gaussienne, condition imposée en x = 0 (entrée), aucune condition en sortie.",
     "cfg": dict(name="Transport avec entrée", family="transport", dim=1, domain={"shape": "interval"},
                 coef={"a": "1"}, u0="exp(-200*(x-0.3)^2)", exact="", transport_bc="inflow", mesh={"nx": 200},
                 time={"T": 0.5, "transport_scheme": "upwind", "dt_mode": "number", "number": 0.9})},
]


def example_config(ex: dict) -> dict:
    return normalize_config(deep_merge(default_config(), ex["cfg"]))[0]


def load_example(ex_id: str) -> None:
    ex = next(e for e in EXAMPLES if e["id"] == ex_id)
    set_config(example_config(ex), f"Exemple chargé : {ex['title']}")


# =============================================================================
# 12c. PAGES — TABLEAU DE BORD
# =============================================================================
def solve_current(show_errors: bool = True) -> dict | None:
    prob = current_problem()
    if not prob.ok:
        if show_errors:
            ui_error("Problème invalide : " + " ; ".join(prob.errors), icon=":material/error:")
        return None
    res = cached_solve(cfg_json_cur())
    store_result("solve", res)
    if not res["ok"] and show_errors:
        ui_error("Échec de la résolution : " + " ; ".join(res["errors"]), icon=":material/error:")
    return res


def run_full_workflow() -> None:
    prob = current_problem()
    cj = cfg_json_cur()
    with st.status("Exécution de la résolution complète…", expanded=True) as status:
        st.write("1/7 — Validation des données et classification de l'EDP")
        if not prob.ok:
            status.update(label="Problème invalide : corrigez la définition", state="error")
            for e in prob.errors:
                st.write(f"• {e}")
            return
        cached_classification(cj)
        st.write("2/7 — Formulations forte et faible, analyse d'existence et d'unicité")
        cached_strong(cj)
        cached_weak(cj)
        cached_lax_milgram(cj)
        st.write("3/7 — Construction du maillage et assemblage des matrices")
        if prob.cls != "transport":
            asm = cached_assembly(cj)
            if not asm.get("ok"):
                status.update(label="Échec de l'assemblage", state="error")
                st.write("; ".join(asm.get("errors", [])))
                return
            store_result("assembly", True)
        st.write("4/7 — Résolution numérique")
        res = cached_solve(cj)
        store_result("solve", res)
        if not res["ok"]:
            status.update(label="Échec de la résolution", state="error")
            st.write("; ".join(res["errors"]))
            return
        st.write("5/7 — Analyse de stabilité (von Neumann et spectre)")
        stab = stability_report(prob, res)
        store_result("stability", stab)
        if prob.time_dependent and prob.cls != "transport" and res.get("dofs", 1e9) <= 1500:
            store_result("spectral", cached_spectral(cj, float(res["dt"])), key=f"{res['dt']:.12g}")
        st.write("6/7 — Étude de convergence par raffinement")
        if res.get("exact_available"):
            levels = 4 if prob.dim == 1 else 3
            conv = cached_convergence(cj, levels, None, None)
            if conv.get("ok"):
                store_result("convergence", conv, key=f"{levels}|None|None")
        else:
            st.write("   (solution exacte inconnue : étude de convergence non applicable)")
        st.write("7/7 — Synthèse et interprétation")
        store_result("report", True)
        status.update(label="Résolution complète terminée", state="complete", expanded=False)


def _dash_result_views(prob: Problem, res: dict) -> None:
    mode = theme_mode()
    tabs = st.tabs(["Vue 3D" if prob.dim == 2 else "Solution", "Vue 2D" if prob.dim == 2 else "Erreur", "Analyse"])
    U = np.asarray(res["U"])
    with tabs[0]:
        if prob.dim == 2:
            show_fig(fig_field_2d(res, U, mode, "surface", "Solution numérique u_h", height=340), key="dash_surf")
        else:
            show_fig(fig_solution_1d(res["nodes"][:, 0], U, res.get("Uex"), mode, "Solution à t = T" if prob.time_dependent
                                     else "Solution", height=320), key="dash_1d")
    with tabs[1]:
        if prob.dim == 2:
            show_fig(fig_field_2d(res, U, mode, "heatmap", "Carte de u_h", height=340), key="dash_heat")
        elif res.get("Uex") is not None:
            x = res["nodes"][:, 0]
            o = np.argsort(x)
            show_fig(fig_curves(x[o], [{"y": (U - np.asarray(res["Uex"]))[o], "name": "u_h − u"}], mode,
                                "Erreur nodale", "x", "e(x)", height=320), key="dash_err")
        else:
            ui_info("Solution exacte non fournie : erreur non calculable.", icon=":material/info:")
    with tabs[2]:
        errs = res.get("errors") or {}
        rows = [("Degrés de liberté", res.get("dofs")), ("Temps total", f"{res['timings'].get('total', 0):.3f} s")]
        for k, lab in (("l2", "Erreur L²"), ("h1", "Semi-norme H¹ de l'erreur"), ("max", "Erreur max"),
                       ("l2_rel", "Erreur L² relative")):
            if errs.get(k) is not None:
                rows.append((lab, fmt(errs[k])))
        if res.get("blowup"):
            rows.append(("Instabilité", f"explosion au pas {res['blowup']['step']}"))
        kv_table(rows)


def page_dashboard() -> None:
    prob = current_problem()
    mode = theme_mode()
    ui_markdown(f'<div class="pdl-hero">{hero_svg(mode)}<div class="txt"><div class="t1">Résolvez. Analysez. Explorez.'
                '</div><div class="t2">Un environnement complet pour la résolution numérique des EDP</div></div></div>',
                unsafe_allow_html=True)
    res = get_result("solve")
    c = st.columns(4)
    c[0].metric("Problème", FAMILY_SHORT.get(prob.family, prob.meta["label"]), help=prob.meta["label"] + " — "
                + FAMILIES[prob.family]["summary"], border=True)
    c[1].metric("Méthode", "FEM P1" if prob.method == "fem" and prob.cls != "transport" else "Différences finies",
                border=True)
    geo = "intervalle" if prob.dim == 1 else ("domaine en L" if prob.shape == "lshape" else "rectangle")
    c[2].metric("Domaine", f"{prob.dim}D · {geo}", border=True)
    if res and res.get("ok"):
        c[3].metric("Degrés de liberté", f"{res['dofs']:,}".replace(",", " "), border=True)
    else:
        try:
            nn, _nt = estimate_cost(prob, prob.cfg) if prob.ok else (0, 0)
        except Exception:  # noqa: BLE001
            nn = 0
        c[3].metric("Nœuds du maillage (estimation)", f"≈ {nn:,}".replace(",", " "), border=True,
                    help="Avant résolution : nombre de nœuds prévu. Après résolution : nombre exact d'inconnues "
                         "(nœuds libres, hors conditions de Dirichlet).")
    # ---------- lancement
    with card("dash_run"):
        a, b = st.columns([1.25, 2])
        with a:
            with st.container(key="run_full"):
                if st.button("EXÉCUTER LA RÉSOLUTION COMPLÈTE", type="primary", icon=":material/rocket_launch:",
                             width="stretch", disabled=not prob.ok, help="Validation → assemblage → résolution → "
                             "stabilité → convergence → synthèse"):
                    run_full_workflow()
                    st.rerun()
        with b:
            ui_markdown('<div class="pdl-note">Ou exécuter les étapes séparément :</div>', unsafe_allow_html=True)
            s = st.columns(4)
            if s[0].button("Assembler", icon=":material/data_array:", width="stretch", disabled=not prob.ok or prob.cls == "transport"):
                asm = cached_assembly(cfg_json_cur())
                if asm.get("ok"):
                    store_result("assembly", True)
                else:
                    st.session_state.flash.append("; ".join(asm.get("errors", [])))
                st.rerun()
            if s[1].button("Résoudre", icon=":material/play_arrow:", width="stretch", disabled=not prob.ok):
                solve_current()
                st.rerun()
            if s[2].button("Stabilité", icon=":material/monitor_heart:", width="stretch",
                           disabled=not (prob.ok and prob.time_dependent)):
                r_ = get_result("solve") or solve_current()
                if r_ and r_.get("ok"):
                    store_result("stability", stability_report(prob, r_))
                st.rerun()
            if s[3].button("Convergence", icon=":material/trending_down:", width="stretch", disabled=not prob.ok):
                conv = cached_convergence(cfg_json_cur(), 4 if prob.dim == 1 else 3, None, None)
                if conv.get("ok"):
                    store_result("convergence", conv, key=f"{4 if prob.dim == 1 else 3}|None|None")
                else:
                    st.session_state.flash.append("Convergence : " + "; ".join(conv.get("errors", [])))
                st.rerun()
    # ---------- configuration + résultat
    left, right = st.columns([1.12, 1])
    with left:
        with card("dash_cfg"):
            card_title("tune", "Configuration du problème")
            a, b = st.columns([1, 1.3])
            with a:
                rows = [("Type d'EDP", prob.meta["label"]), ("Classe", CLASS_LABELS[prob.cls]),
                        ("Domaine Ω", prob.describe().split(" — ")[-1])]
                if prob.cls != "transport":
                    types = sorted({BC_LABELS[b_["type"]] for b_ in prob.bc.values()})
                    rows.append(("Conditions aux limites", ", ".join(types)))
                else:
                    rows.append(("Bord", "périodique" if prob.transport_bc == "periodic" else "entrée imposée"))
                if "f" in prob.meta["uses"]:
                    src_f = prob.cfg["f"] if not prob.manufactured else "dérivé de u (manufacturée)"
                    rows.append(("Terme source f", src_f if len(src_f) <= 60 else src_f[:57] + "…"))
                rows.append(("Solution exacte", "oui" if prob.exact is not None or prob.cls == "transport" else "non"))
                kv_table(rows)
            with b:
                if prob.ok:
                    ui_problem_box(prob, "dash")
                else:
                    ui_error("Définition invalide.", icon=":material/error:")
            nav_button("definition", "Modifier le problème", icon=":material/edit:", key="dash_edit")
    with right:
        with card("dash_res"):
            card_title("check_circle" if res and res.get("ok") else "hourglass_empty", "Résultat : solution numérique")
            if res and res.get("ok"):
                _dash_result_views(prob, res)
            elif res:
                ui_error("; ".join(res.get("errors", [])), icon=":material/error:")
            else:
                stale = get_result("solve", allow_stale=True)
                ui_info("Aucune résolution pour la configuration actuelle"
                        + (" (le résultat précédent est obsolète)." if stale else ".")
                        + " Lancez la résolution complète ci-dessus.", icon=":material/info:")
    # ---------- maillage / convergence / informations
    c1, c2, c3 = st.columns([1.05, 1.1, 0.95])
    with c1:
        with card("dash_mesh"):
            card_title("change_history", "Maillage et méthode")
            if prob.ok and prob.cls != "transport":
                asm = cached_assembly(cfg_json_cur()) if (get_result("assembly") or res) else None
                if asm and asm.get("ok") and asm["n"] <= 6000:
                    cats = node_categories(prob, asm["nodes"], asm["side_nodes"])
                    fig = fig_mesh(asm["nodes"], asm["elems"], cats, mode, None, grid=asm.get("grid"), height=260,
                                   show_nodes=prob.dim == 1)
                    fig.update_layout(showlegend=False, margin=dict(l=40, r=10, t=10, b=40))
                    show_fig(fig, key="dash_meshfig")
                    nel = len(asm["elems"]) if asm["elems"] is not None else 0
                    kv_table([("Éléments" if nel else "Mailles", nel or f"{asm['grid']['nx']}×{max(asm['grid']['ny'], 1)}"),
                              ("Nœuds", asm["n"]), ("Méthode", METHOD_LABELS[prob.method])])
                elif asm and asm.get("ok"):
                    kv_table([("Nœuds", asm["n"]), ("h", fmt(asm["h"]))])
                else:
                    ui_caption("Assemblez ou résolvez pour afficher le maillage.")
            elif prob.ok:
                kv_table([("Grille", f"{prob.cfg['mesh']['nx']} intervalles"),
                          ("Schéma", TRANSPORT_SCHEMES[prob.cfg['time']['transport_scheme']])])
    with c2:
        with card("dash_conv"):
            card_title("trending_down", "Convergence")
            conv = None
            r0 = st.session_state.results.get("convergence")
            if r0 and r0["hash"] == cur_hash():
                conv = r0["data"]
            if conv and conv.get("ok"):
                fig = fig_loglog_convergence(conv["rows"], conv["keys"], conv["slopes"], conv["theory"], mode, None, 270)
                fig.update_layout(margin=dict(l=56, r=10, t=10, b=44), showlegend=False)
                show_fig(fig, key="dash_convfig")
                lab = {"l2": "L²", "h1": "H¹", "max": "∞"}
                ui_caption("Pentes observées : " + ", ".join(
                    f"{lab.get(k, k)} {conv['slopes'][k]:.2f} (théorie {fmt(conv['theory']['orders'].get(k), 3)})"
                    for k in conv["keys"] if k in conv["slopes"]) + " — pointillés : pentes théoriques.")
            else:
                ui_caption("Étude de convergence non effectuée pour cette configuration.")
                nav_button("convergence", "Ouvrir l'étude", icon=":material/open_in_new:", key="dash_goconv")
    with c3:
        with card("dash_info"):
            card_title("info", "Informations")
            if res and res.get("ok"):
                if res.get("blowup"):
                    ui_error(f"Instabilité détectée (pas {res['blowup']['step']})", icon=":material/warning:")
                else:
                    ui_success("Résolution réussie", icon=":material/check_circle:")
                errs = res.get("errors") or {}
                rows = []
                if errs.get("l2") is not None:
                    rows.append(("Erreur L²", fmt(errs["l2"])))
                rows += [("Temps de calcul", f"{res['timings'].get('total', 0):.3f} s"),
                         ("Mémoire (structures)", f"{res.get('memory', 0) / 2 ** 20:.2f} Mio")]
                if res.get("dt"):
                    rows.append(("Δt / N_t", f"{fmt(res['dt'])} / {res['nt']}"))
                kv_table(rows)
                for w in res.get("warnings", [])[:2]:
                    ui_caption(f":orange[:material/warning: {w}]")
                nav_button("report", "Voir la synthèse", icon=":material/summarize:", key="dash_report")
            else:
                ui_caption("Système prêt. Aucun résultat pour la configuration courante.")
    # ---------- accès rapide
    with card("dash_nav"):
        card_title("apps", "Accès rapide aux étapes")
        keys = ["definition", "strong", "weak", "spaces", "existence", "discretization", "mesh", "matrices",
                "solve", "stability", "convergence", "visualization", "comparison", "report", "export", "library"]
        cols = st.columns(4)
        for i, k in enumerate(keys):
            with cols[i % 4]:
                page_link(k)


# =============================================================================
# 12d. PAGES — PROBLÈME
# =============================================================================
def page_definition() -> None:
    page_header("Module 1", "Définition et classification de l'EDP",
                "Choisissez une famille d'équations, la géométrie, les coefficients, les données et les conditions aux "
                "limites. Les expressions sont analysées par un interpréteur à liste blanche (aucune exécution de code).")
    cfg = cfg_raw()
    prob = current_problem()
    meta = FAMILIES[cfg["family"]]
    sv = ("x",) if cfg["dim"] == 1 else ("x", "y")
    td = meta["time_order"] > 0
    left, right = st.columns([1.35, 1])
    with left:
        with card("def_family"):
            card_title("category", "Famille d'EDP et méthode")
            fams = list(FAMILIES)
            w_select("Famille d'EDP", "family", "def", fams,
                     {k: f"{CLASS_LABELS[v['class']]} · {v['label']}" for k, v in FAMILIES.items()}, structural=True)
            ui_caption(meta["summary"])
            a, b = st.columns(2)
            with a:
                w_select("Dimension d'espace", "dim", "def", list(meta["dims"]), {1: "1D", 2: "2D"}, structural=True,
                         horizontal_radio=True, disabled=len(meta["dims"]) == 1)
            with b:
                w_select("Méthode numérique", "method", "def", list(meta["methods"]),
                         {"fdm": "Différences finies", "fem": "Éléments finis P1"}, structural=True,
                         horizontal_radio=True, disabled=len(meta["methods"]) == 1 or cfg["domain"]["shape"] == "lshape")
        with card("def_geom"):
            card_title("crop_square", "Géométrie du domaine")
            if cfg["dim"] == 2 and meta["class"] == "elliptic":
                w_select("Forme", "domain.shape", "def", ["rectangle", "lshape"],
                         {"rectangle": "Rectangle", "lshape": "Domaine en L (éléments finis)"}, structural=True,
                         horizontal_radio=True)
            lsh = cfg["domain"]["shape"] == "lshape"
            cols = st.columns(4 if cfg["dim"] == 2 else 2)
            with cols[0]:
                w_number("x_a", "domain.xa", "def", -100.0, 100.0, 0.1, disabled=lsh)
            with cols[1]:
                w_number("x_b", "domain.xb", "def", -100.0, 100.0, 0.1, disabled=lsh)
            if cfg["dim"] == 2:
                with cols[2]:
                    w_number("y_a", "domain.ya", "def", -100.0, 100.0, 0.1, disabled=lsh)
                with cols[3]:
                    w_number("y_b", "domain.yb", "def", -100.0, 100.0, 0.1, disabled=lsh)
            if lsh:
                ui_caption("Ω = (−1, 1)² privé du quadrant [0, 1] × [−1, 0] ; angle rentrant 3π/2 en (0, 0).")
        with card("def_coef"):
            card_title("functions", "Coefficients, paramètres et terme source")
            uses = meta["uses"]
            manuf = cfg["manufactured"] and bool(cfg["exact"].strip())
            cols = st.columns(2)
            i = 0
            fields = []
            if "k" in uses:
                fields.append(("Diffusion k(x" + (",y" if cfg["dim"] == 2 else "") + ")", "coef.k", sv,
                               "Coefficient de diffusion, doit vérifier k ≥ k₀ > 0."))
            if "c" in uses:
                fields.append(("Réaction c(x" + (",y" if cfg["dim"] == 2 else "") + ")", "coef.c", sv,
                               "c > 0 : absorption ; c < 0 : production."))
            if "kappa" in uses:
                fields.append(("Diffusivité κ (constante)", "coef.kappa", (), "κ > 0."))
            if "b" in uses:
                fields.append(("Vitesse b_x (constante)", "coef.bx", (), None))
                if cfg["dim"] == 2:
                    fields.append(("Vitesse b_y (constante)", "coef.by", (), None))
            if "cw" in uses:
                fields.append(("Célérité c (constante)", "coef.cw", (), "c > 0."))
            if "a" in uses:
                fields.append(("Vitesse de transport a (constante)", "coef.a", (), "a ≠ 0 ; le signe fixe le bord d'entrée."))
            for lab, path, allowed, hlp in fields:
                with cols[i % 2]:
                    val = w_text(lab, path, "def", help=hlp)
                    expr_feedback(val, allowed)
                i += 1
            if "f" in uses:
                fv = sv + (("t",) if td else ())
                val = w_text("Terme source f(" + ", ".join(fv) + ")", "f", "def", disabled=manuf,
                             help="Fonctions : sin, cos, tan, exp, log, sqrt, abs, sinh, cosh, tanh, atan, atan2, "
                                  "min, max, heaviside ; constantes pi, e ; puissance ^ ou **.")
                if manuf:
                    ui_caption(f"Dérivé de la solution manufacturée : $f = {latex_of(prob.f_expr)}$" if prob.ok else "")
                else:
                    expr_feedback(val, fv)
        if td:
            with card("def_ic"):
                card_title("schedule", "Conditions initiales et horizon de temps")
                a, b = st.columns([2, 1])
                manuf = cfg["manufactured"] and bool(cfg["exact"].strip())
                with a:
                    val = w_text("u₀(" + ", ".join(sv) + ")", "u0", "def", disabled=manuf)
                    if manuf:
                        ui_caption("Dérivée de la solution exacte : u₀ = u(·, 0).")
                    else:
                        expr_feedback(val, sv)
                    if meta["time_order"] == 2:
                        val = w_text("v₀(" + ", ".join(sv) + ") = ∂ₜu(·, 0)", "v0", "def", disabled=manuf)
                        if not manuf:
                            expr_feedback(val, sv)
                with b:
                    w_number("Temps final T", "time.T", "def", 1e-4, 1e3, 0.05)
        with card("def_bc"):
            card_title("border_outer", "Conditions aux limites")
            if meta["class"] == "transport":
                w_select("Type de bord", "transport_bc", "def", ["periodic", "inflow"],
                         {"periodic": "Périodique", "inflow": "Entrée imposée (bord amont)"}, horizontal_radio=True)
                ui_caption("Avec une condition d'entrée, la donnée au bord amont est u₀(x_amont − a t) (prolongement de "
                           "la formule) ou la solution exacte si elle est fournie. Aucune condition n'est imposée en sortie.")
            else:
                sides = sides_of(cfg)
                bt = list(meta["bc_types"])
                manuf = cfg["manufactured"] and bool(cfg["exact"].strip())
                cols = st.columns(2 if len(sides) > 1 else 1)
                for j, s in enumerate(sides):
                    with cols[j % len(cols)]:
                        ui_markdown(f"**Bord {SIDE_LABELS[s]}**" + (f" — ${GAMMA[s]}$" if cfg['dim'] == 2 else ""))
                        if s == "boundary":
                            ui_caption("Domaine en L : Dirichlet sur tout le bord.")
                        else:
                            w_select("Type", f"bc.{s}.type", "def", bt, BC_LABELS)
                        gv = sv + (("t",) if td else ())
                        val = w_text("Donnée g", f"bc.{s}.g", "def", disabled=manuf,
                                     help="Dirichlet : u = g ; Neumann : k ∂u/∂n = g ; Robin : k ∂u/∂n + βu = g "
                                          "(n normale sortante).")
                        if not manuf:
                            expr_feedback(val, gv)
                        if cfg["bc"][s]["type"] == "robin":
                            val = w_text("Coefficient β", f"bc.{s}.beta", "def")
                            expr_feedback(val, sv)
                if manuf:
                    ui_caption("Solution manufacturée active : les données g sont dérivées symboliquement de u "
                               "(Dirichlet : u ; Neumann : k ∂u/∂n ; Robin : k ∂u/∂n + βu).")
        with card("def_exact"):
            card_title("verified", "Solution exacte (facultative)")
            ev = sv + (("t",) if td else ())
            val = w_text("u_exacte(" + ", ".join(ev) + ")", "exact", "def",
                         help="Sert au calcul des erreurs et des ordres de convergence.")
            if val.strip():
                expr_feedback(val, ev)
            if meta["class"] != "transport":
                w_toggle("Solution manufacturée : dériver f, g et les conditions initiales de u_exacte", "manufactured",
                         "def", structural=True,
                         help="Méthode des solutions manufacturées : f = L(u), données de bord et initiales calculées "
                              "symboliquement. Permet de vérifier n'importe quelle configuration.")
            if prob.ok and prob.exact is not None:
                chk = cached_exact_check(cfg_json_cur())
                if chk:
                    worst = max(v for v in chk.values() if v == v)
                    if worst < 1e-8:
                        ui_success("La solution exacte vérifie l'EDP, les conditions aux limites et initiales "
                                   f"(résidu relatif max {worst:.1e}).", icon=":material/task_alt:")
                    else:
                        ui_warning("La solution exacte ne vérifie pas toutes les équations : "
                                   + ", ".join(f"{k} : {v:.1e}" for k, v in chk.items() if v > 1e-8)
                                   + ". Les erreurs calculées ne mesureraient pas l'erreur de discrétisation.",
                                   icon=":material/warning:")
    with right:
        with card("def_class"):
            card_title("schema", "Classification mathématique automatique")
            if prob.ok:
                cl = cached_classification(cfg_json_cur())
                ui_markdown(badge(cl["type"], "blue") + "&nbsp;" + badge("Linéaire", "green") + "&nbsp;"
                            + badge("Stationnaire" if cl["stationary"] else "Évolution", "gray"), unsafe_allow_html=True)
                kv_table([("Ordre des dérivées", f"{cl['order']} (espace : {cl['order_space']}, temps : {cl['order_time']})"),
                          ("Variables spatiales", f"{cl['n_space']} ({'x' if cl['n_space'] == 1 else 'x, y'})"),
                          ("Variable temporelle", "non" if cl["stationary"] else "t"),
                          ("k", f"[{fmt(cl['k_range'][0])}, {fmt(cl['k_range'][1])}]"
                                + (" (exact)" if cl["k_range"][2] else " (échantillonné)"))])
                if cl.get("principal") is not None:
                    ui_markdown("Matrice des coefficients principaux (variables "
                                + ", ".join(cl["variables"]) + ") :")
                    st.latex(r"A = " + cl["principal_latex"])
                    if cl.get("discriminant"):
                        dd = cl["discriminant"]
                        st.latex(rf"B^2 - 4AC = {dd['value']}\ {dd['sign'].replace('<', '<').replace('>', '>')}\ "
                                 r"\Rightarrow\ \text{" + cl["type"] + "}")
                ui_caption(cl.get("note", ""))
                ui_caption(cl["linear_reason"])
                ui_caption(cl["regularity"])
            else:
                ui_caption("Classification indisponible tant que la définition contient des erreurs.")
        with card("def_valid"):
            card_title("rule", "Validation")
            if prob.errors:
                for e in prob.errors:
                    ui_error(e, icon=":material/error:")
            else:
                ui_success("Toutes les expressions sont valides et le problème est dans le cadre traité.",
                           icon=":material/check_circle:")
            for w in prob.warnings:
                ui_warning(w, icon=":material/warning:")
            for n_ in prob.notes:
                ui_info(n_, icon=":material/info:")
        if prob.ok:
            ui_problem_box(prob, "def")
        with card("def_examples"):
            card_title("library_books", "Partir d'un exemple préconfiguré")
            ex = st.selectbox("Exemple", [e["id"] for e in EXAMPLES],
                              format_func=lambda i: next(f"{e['group']} · {e['title']}" for e in EXAMPLES if e["id"] == i),
                              key="def_ex_select")
            ui_caption(next(e["desc"] for e in EXAMPLES if e["id"] == ex))
            if st.button("Charger cet exemple", icon=":material/download_for_offline:", key="def_ex_load"):
                load_example(ex)
                st.rerun()
        with st.expander("Syntaxe des expressions", icon=":material/help:"):
            ui_markdown("- Variables : `x`, `y`, `t` (selon le contexte) ; constantes `pi`, `e`\n"
                        "- Fonctions : `sin cos tan exp log ln sqrt abs sinh cosh tanh atan asin acos atan2 sign "
                        "heaviside min max`\n- Opérateurs : `+ - * / ^` (ou `**`) ; multiplication implicite `2x`, "
                        "`2(x+1)`, `pi x`\n- Exemple : `2*pi^2*sin(pi*x)*sin(pi*y)`\n\nToute autre construction "
                        "(attributs, appels inconnus, chaînes, crochets…) est refusée.")
    next_step("strong", "formulation forte")


def page_strong() -> None:
    page_header("Module 2", "Formulation forte et analyse mathématique",
                "L'équation est écrite ponctuellement, avec son domaine, ses conditions aux limites et initiales. "
                "On distingue inconnues, données, paramètres et variables.")
    prob = need_valid_problem()
    if prob is None:
        return
    sf = cached_strong(cfg_json_cur())
    a, b = st.columns([1.3, 1])
    with a:
        with card("strong_eq"):
            card_title("functions", "Équation aux dérivées partielles")
            ui_markdown("Forme générique :")
            st.latex(sf["pde_generic"])
            ui_markdown("Avec les données du problème :")
            st.latex(sf["pde"])
            ui_markdown("Domaine :")
            st.latex(sf["domain"])
        with card("strong_bc"):
            card_title("border_outer", "Conditions aux limites" + (" et initiales" if sf["ics"] else ""))
            for lab, lx in sf["bcs"]:
                ui_markdown(f"**{lab}**")
                st.latex(lx)
            for lx in sf["ics"]:
                ui_markdown("**Condition initiale**")
                st.latex(lx)
            if prob.cls != "transport":
                ui_caption("n désigne la normale unitaire sortante. Dirichlet impose la valeur, Neumann le flux, "
                           "Robin (Fourier) une relation d'échange entre flux et valeur.")
        with card("strong_terms"):
            card_title("label", "Interprétation des termes")
            for lx, name, txt in sf["terms"]:
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.latex(lx)
                with c2:
                    ui_markdown(f"**{name}** — {txt}")
    with b:
        with card("strong_vars"):
            card_title("data_object", "Inconnues, données, paramètres, variables")
            ui_markdown("**Inconnue**")
            for u in sf["unknowns"]:
                ui_markdown(f"- {u}")
            ui_markdown("**Données**")
            for dd in sf["data"]:
                ui_markdown(f"- `{dd}`")
            ui_markdown("**Paramètres**")
            for p in sf["params"] or ["—"]:
                ui_markdown(f"- `{p}`")
            ui_markdown(f"**Variables spatiales** : {sf['space_vars']}  \n**Variable temporelle** : {sf['time_var']}")
        with card("strong_hyp"):
            card_title("rule", "Hypothèses mathématiques")
            for h in sf["hypotheses"]:
                st.latex(h)
            ui_caption("Ces hypothèses fondent l'analyse (formulation faible, existence, stabilité). Leur vérification "
                       "pour les données saisies est détaillée à la page « Existence et unicité ».")
        if prob.exact is not None:
            with card("strong_exact"):
                card_title("task_alt", "Vérification de la solution exacte")
                chk = cached_exact_check(cfg_json_cur()) or {}
                st.dataframe(pd.DataFrame({"Équation": list(chk), "Résidu relatif max": [f"{v:.2e}" for v in chk.values()],
                                           "Statut": ["vérifiée" if v < 1e-8 else "NON vérifiée" for v in chk.values()]}),
                             hide_index=True, width="stretch")
    next_step("weak", "formulation faible")


def page_weak() -> None:
    page_header("Module 3", "Formulation faible",
                "Passage pas à pas de la formulation forte à la formulation variationnelle, adapté aux conditions aux "
                "limites choisies (Dirichlet essentielle, Neumann naturelle, Robin mixte).")
    prob = need_valid_problem()
    if prob is None:
        return
    wf = cached_weak(cfg_json_cur())
    if prob.cls == "transport":
        with card("weak_tr"):
            card_title("waves", "Équation de transport : formulation faible au sens des distributions")
            for title, lx, txt in wf["transport"]:
                ui_markdown(f"**{title}**")
                st.latex(lx)
                ui_caption(txt)
        next_step("spaces", "espaces fonctionnels")
        return
    a, b = st.columns([1.45, 1])
    with a:
        with card("weak_steps"):
            card_title("integration_instructions", "Dérivation de la formulation faible")
            for i, step in enumerate(wf["steps"]):
                with st.expander(step["title"], expanded=i < 6):
                    st.latex(step["latex"])
                    ui_markdown(step["text"])
    with b:
        with card("weak_forms"):
            card_title("function", "Formes identifiées")
            st.latex(wf["a_latex"])
            st.latex(wf["L_latex"])
            if wf.get("instantiated"):
                ui_markdown("avec les données :")
                st.latex(r",\quad ".join(wf["instantiated"]))
            if prob.time_dependent:
                ui_caption("Pour un problème d'évolution, la formulation faible est posée en espace pour presque tout t ; "
                           "la dérivée temporelle est traitée au sens de la dualité V′–V.")
        with card("weak_spaces"):
            card_title("hub", "Espaces de recherche et de test")
            sp_ = wf["spaces"]
            st.latex(r"\text{Tests : }" + sp_["test"])
            st.latex(r"\text{Recherche : }" + sp_["trial"])
            ui_caption(sp_["comment"])
        with card("weak_peda"):
            card_title("school", "Comprendre chaque transformation")
            for q, ans in wf["pedagogy"]:
                with st.expander(q):
                    ui_markdown(ans)
    next_step("spaces", "espaces fonctionnels")


def _norm_calculator(prob: Problem) -> None:
    sv = prob.space_vars
    default = "x*(1-x)" if prob.dim == 1 else "x*(1-x)*y*(1-y)"
    src = st.text_input("Fonction v(" + ", ".join(sv) + ")", value=default, key="spaces_v")
    ok, info = expr_check(src, sv)
    if not ok:
        ui_caption(f":red[{info}]")
        return
    v = parse_expression(src, sv)
    try:
        fv = NumFunc(v)
        dvs = [NumFunc(sympy.diff(v, SYMBOLS[s])) for s in sv]
    except ExpressionError as exc:
        ui_error(f"v ∉ H¹(Ω) : sa dérivée au sens des distributions n'est pas une fonction ({exc}).",
                 icon=":material/block:")
        return

    def norms(panels: int) -> tuple[float, float]:
        xq, yq, wq = domain_quadrature(prob, panels=panels)
        l2 = float(np.sqrt(np.sum(wq * fv(xq, yq) ** 2)))
        g2 = sum(np.sum(wq * d(xq, yq) ** 2) for d in dvs)
        return l2, float(np.sqrt(g2))

    l2a, h1a = norms(16)
    l2b, h1b = norms(48)
    tr2 = 0.0
    for s in prob.sides:
        bx, by, bw = boundary_quadrature(prob, s)
        tr2 += float(np.sum(bw * fv(bx, by) ** 2))
    finite = all(math.isfinite(v_) for v_ in (l2b, h1b))
    conv = finite and abs(h1a - h1b) <= 1e-3 * max(h1b, 1e-12)
    rows = [("‖v‖_L²(Ω)", fmt(l2b, 6)), ("|v|_H¹(Ω) = ‖∇v‖_L²", fmt(h1b, 6)),
            ("‖v‖_H¹(Ω)", fmt(math.sqrt(l2b ** 2 + h1b ** 2), 6)),
            ("‖v‖_L²(∂Ω) (trace)" if prob.dim == 2 else "Σ v(bords)²  (trace ponctuelle)", fmt(math.sqrt(tr2), 6))]
    kv_table(rows)
    if not conv:
        ui_warning("L'intégrale de |∇v|² n'est pas stabilisée lorsque la quadrature est raffinée : ∇v n'est "
                   "probablement pas de carré intégrable (singularité), donc v ∉ H¹(Ω).", icon=":material/warning:")
    elif math.sqrt(tr2) < 1e-10:
        ui_success("Trace nulle sur ∂Ω et ∇v ∈ L² : v ∈ H₀¹(Ω).", icon=":material/check_circle:")
    else:
        ui_info("v ∈ H¹(Ω) mais sa trace est non nulle : v ∉ H₀¹(Ω).", icon=":material/info:")


def page_spaces() -> None:
    page_header("Module 3 bis", "Espaces fonctionnels (Sobolev)",
                "Définitions, normes, dérivées faibles, traces et inclusions — et le rôle précis de chaque espace "
                "dans la formulation faible du problème courant.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    a, b = st.columns(2)
    with a:
        with card("sp_l2"):
            card_title("square_foot", "L²(Ω) — énergie finie")
            st.latex(r"\begin{aligned}&L^2(\Omega)=\Big\{v \text{ mesurable} : \int_\Omega |v|^2\,d\mathbf x<\infty\Big\},\\ "
                     r"&(u,v)=\int_\Omega uv,\qquad \|v\|_{L^2}=(v,v)^{1/2}\end{aligned}")
            ui_caption("Espace de Hilbert des données (f, u₀) : ses éléments sont des classes de fonctions définies "
                       "presque partout — pas de valeur ponctuelle.")
        with card("sp_weak"):
            card_title("deblur", "Dérivée faible")
            st.latex(r"g_i = \partial_i v\ \text{au sens faible} \iff \int_\Omega v\,\partial_i\varphi = -\int_\Omega g_i\,"
                     r"\varphi\quad \forall\varphi\in C_c^\infty(\Omega)")
            ui_caption("C'est la formule d'intégration par parties prise comme définition : elle a un sens pour des "
                       "fonctions seulement intégrables. Exemple : |x − ½| a pour dérivée faible sign(x − ½) ∈ L².")
        with card("sp_h1"):
            card_title("hub", "H¹(Ω) et H₀¹(Ω)")
            st.latex(r"H^1(\Omega)=\{v\in L^2 : \nabla v\in L^2(\Omega)^d\},\quad \|v\|_{H^1}^2=\|v\|_{L^2}^2+\|\nabla v\|_{L^2}^2,"
                     r"\quad |v|_{H^1}=\|\nabla v\|_{L^2}")
            st.latex(r"H_0^1(\Omega)=\overline{C_c^\infty(\Omega)}^{\,H^1}=\{v\in H^1(\Omega) : \gamma_0 v = 0\}")
            ui_caption("H¹ est l'espace d'énergie : a(v, v) y est fini. H₀¹ incorpore la condition de Dirichlet homogène.")
        with card("sp_trace"):
            card_title("border_style", "Traces au bord")
            st.latex(r"\gamma_0 : H^1(\Omega)\to H^{1/2}(\partial\Omega)\subset L^2(\partial\Omega),\quad "
                     r"\|\gamma_0 v\|_{L^2(\partial\Omega)}\le C_{tr}\|v\|_{H^1(\Omega)}")
            ui_caption("Ω lipschitzien : l'application trace prolonge v ↦ v|∂Ω et est continue et surjective sur "
                       "H¹ᐟ²(∂Ω). Elle donne un sens aux conditions de Dirichlet (et le relèvement des données).")
    with b:
        with card("sp_incl"):
            card_title("account_tree", "Inclusions et inégalités clés")
            st.latex(r"H_0^1(\Omega)\ \subset\ H^1(\Omega)\ \subset\ L^2(\Omega)\ \subset\ H^{-1}(\Omega)")
            st.latex(r"\text{Poincaré : }\ \|v\|_{L^2}\le C_P\|\nabla v\|_{L^2}\quad\forall v\in H^1_{\Gamma_D}(\Omega),\ "
                     r"|\Gamma_D|>0")
            st.latex(r"\text{Rellich : } H^1(\Omega)\hookrightarrow L^2(\Omega)\ \text{compacte}\ (\Omega\ \text{borné})")
            st.latex(r"\begin{aligned}&d=1:\ H^1(I)\subset C^0(\bar I)\\ &d=2:\ H^1(\Omega)\not\subset C^0(\bar\Omega)\ "
                     r"\big(\text{ex. } \log|\log r|\big)\end{aligned}")
            ui_caption("En 1D les fonctions H¹ sont continues (valeurs ponctuelles définies) ; en 2D, non : seules les "
                       "traces sur des courbes ont un sens. C'est pourquoi les conditions de bord sont prises au sens des traces.")
        with card("sp_role"):
            card_title("assignment", "Rôle dans le problème courant")
            sp_ = function_spaces(prob) if prob.cls != "transport" else None
            if sp_:
                st.latex(r"\text{Tests : } " + sp_["test"])
                st.latex(r"\text{Recherche : } " + sp_["trial"])
                rows = [("Données f, u₀", "L²(Ω)"), ("Solution, fonctions tests", "H¹ (avec traces)"),
                        ("Donnée de Dirichlet g_D", "H¹ᐟ²(Γ_D)"), ("Données de Neumann/Robin", "L²(Γ_N ∪ Γ_R)"),
                        ("Forme a(·,·)", "continue sur H¹ × H¹"),
                        ("Dérivée ∂ₜu (évolution)", "L²(0, T ; V′)" if prob.time_dependent else "—")]
                kv_table(rows)
                ui_caption(sp_["comment"])
                if prob.method == "fem":
                    ui_markdown("**Discrétisation conforme :**")
                    st.latex(r"V_h=\{v_h\in C^0(\bar\Omega) : v_h|_K\in P_1(K)\ \forall K\in\mathcal T_h\}\cap V\ \subset\ V")
                    ui_caption("Les fonctions P1 continues sont dans H¹ (dérivée faible constante par élément) ; la "
                               "continuité inter-éléments est indispensable : un saut produirait une masse de Dirac.")
            else:
                ui_caption("Transport : solutions dans C([0,T]; L²(Ω)) ; pas de cadre H¹ elliptique.")
    with card("sp_tools"):
        card_title("calculate", "Calculateur de normes et dérivée faible illustrée")
        c1, c2 = st.columns([1, 1.25])
        with c1:
            ui_markdown("Saisissez une fonction : ses normes sont calculées par quadrature de Gauss composite, avec "
                        "test de convergence de l'intégrale (détection des singularités).")
            _norm_calculator(prob)
        with c2:
            choice = st.segmented_control("Exemple 1D", ["|x − ½|", "chapeau P1", "√x", "créneau"], default="|x − ½|",
                                          key="sp_wd_choice", required=True) or "|x − ½|"
            xs = np.linspace(0, 1, 801)
            if choice == "chapeau P1":
                v = np.maximum(0, 1 - np.abs(xs - 0.5) / 0.2)
                dv = np.where(np.abs(xs - 0.5) < 0.2, -np.sign(xs - 0.5) / 0.2, 0.0)
                note = "φᵢ ∈ H¹ : dérivée faible constante par morceaux, à sauts aux nœuds ⇒ φᵢ ∉ H²."
            elif choice == "√x":
                v = np.sqrt(xs)
                dv = np.where(xs > 0, 0.5 / np.sqrt(np.maximum(xs, 1e-12)), np.nan)
                note = "v′ = 1/(2√x) ∉ L²(0,1) car ∫ 1/(4x) dx diverge : √x ∈ L² mais √x ∉ H¹."
            elif choice == "créneau":
                v = (xs > 0.5).astype(float)
                dv = np.zeros_like(xs)
                note = "Dérivée au sens des distributions = δ(x − ½) (masse de Dirac), pas une fonction : v ∉ H¹."
            else:
                v = np.abs(xs - 0.5)
                dv = np.sign(xs - 0.5)
                note = "v ∈ H¹ (dérivée faible sign(x − ½) ∈ L²) mais v ∉ H² : v″ = 2δ(x − ½)."
            fig = fig_curves(xs, [{"y": v, "name": "v(x)"}, {"y": np.clip(dv, -6, 6), "name": "dérivée faible v′(x)"}],
                             mode, f"Fonction et dérivée faible — {choice}", "x", "valeur", height=330)
            show_fig(fig, note, key="sp_wd_fig")
    next_step("existence", "existence et unicité")


def _render_items(items: list[dict]) -> None:
    for it in items:
        c1, c2 = st.columns([3, 1.3])
        with c1:
            ui_markdown(f"**{it['title']}**")
        with c2:
            ui_markdown(status_badge(it["status"]), unsafe_allow_html=True)
        st.latex(it["latex"])
        ui_caption(it["detail"])


def page_existence() -> None:
    page_header("Module 4", "Existence et unicité",
                "Le théorème de Lax–Milgram n'est appliqué que lorsque ses hypothèses sont établies. Chaque hypothèse "
                "porte un statut : satisfaite (exacte), vérifiée numériquement, non vérifiée, ou cas non traité.")
    prob = need_valid_problem()
    if prob is None:
        return
    lm = cached_lax_milgram(cfg_json_cur())
    with card("ex_thm"):
        card_title("gavel", "Théorème de Lax–Milgram")
        st.latex(r"\begin{aligned}&V\ \text{Hilbert},\ a:V\times V\to\mathbb R\ \text{bilinéaire},\ L\in V'\ \text{tels que}\\"
                 r"&|a(u,v)|\le M\|u\|_V\|v\|_V,\qquad a(v,v)\ge\alpha\|v\|_V^2\ (\alpha>0),\qquad |L(v)|\le C\|v\|_V\\"
                 r"&\Longrightarrow\quad \exists!\,u\in V,\quad a(u,v)=L(v)\ \ \forall v\in V,\qquad \|u\|_V\le C/\alpha.\end{aligned}")
        ui_caption("À ne pas confondre avec le théorème d'équivalence de Lax–Richtmyer (consistance + stabilité ⇔ "
                   "convergence), qui concerne les schémas numériques d'évolution.")
    if lm["framework"] == "elliptic":
        a, b = st.columns([1.5, 1])
        with a:
            with card("ex_items"):
                card_title("checklist", "Vérification des hypothèses")
                _render_items(lm["items"])
        with b:
            with card("ex_concl"):
                card_title("flag", "Conclusion")
                if lm["applicable"]:
                    ui_markdown(status_badge("applicable"), unsafe_allow_html=True)
                    st.latex(lm["conclusion"])
                    ui_markdown("**Principe de minimisation (a symétrique)**")
                    st.latex(lm["energy"])
                    ui_markdown("**Lemme de Céa (éléments finis conformes, V_h ⊂ V)**")
                    st.latex(lm["cea"])
                    st.latex(lm["cea_sym"])
                else:
                    ui_markdown(status_badge("not_verified"), unsafe_allow_html=True)
                    ui_warning("Au moins une hypothèse n'est pas établie : aucune conclusion d'existence n'est affichée. "
                               "Le calcul numérique reste possible mais n'est pas couvert par cette analyse.",
                               icon=":material/warning:")
            with card("ex_const"):
                card_title("pin", "Constantes calculées")
                cst = lm["constants"]
                rows = [(k, fmt(v)) for k, v in (("k_min", cst.get("k_min")), ("k_max", cst.get("k_max")),
                                                  ("c_min", cst.get("c_min")), ("c_max", cst.get("c_max")),
                                                  ("λ₁ (Poincaré)", cst.get("lambda1")), ("α (coercivité)", cst.get("alpha")),
                                                  ("M (continuité)", cst.get("M")), ("‖f‖_L²", cst.get("f_l2")))
                        if v is not None]
                kv_table(rows)
            with card("ex_poinc"):
                card_title("insights", "Rôle de l'inégalité de Poincaré")
                st.latex(r"a(v,v)\ \ge\ k_{\min}\|\nabla v\|^2\ \ge\ k_{\min}\frac{\lambda_1}{1+\lambda_1}\|v\|_{H^1}^2"
                         r"\quad (c\ge 0)")
                ui_caption("Sans réaction, a(v, v) ne contrôle que ‖∇v‖ ; Poincaré (Γ_D de mesure non nulle) permet "
                           "de contrôler aussi ‖v‖_L², donc toute la norme H¹. Pour Neumann pur, les constantes "
                           "échappent à ce contrôle : d'où le quotient par ℝ et la condition de compatibilité.")
        with card("ex_discrete"):
            card_title("data_array", "Problème discret")
            ui_markdown("En éléments finis conformes, Lax–Milgram s'applique sur V_h ⊂ V avec les mêmes constantes : "
                        "la matrice de rigidité est symétrique définie positive, donc inversible. En différences finies, "
                        "l'inversibilité découle du caractère M-matrice (principe du maximum discret) — vérifié à la "
                        "page « Matrices et assemblage ».")
    else:
        with card("ex_evo"):
            card_title("update", "Problème d'évolution : " + lm.get("theorem", ""))
            ui_markdown(status_badge("not_treated") + "&nbsp; Lax–Milgram n'est pas le bon outil ici.",
                        unsafe_allow_html=True)
            ui_markdown(lm["why_not_lm"])
            _render_items(lm["items"])
            ui_markdown("**Conclusion**")
            st.latex(lm["conclusion"])
            ui_markdown("**Estimation d'énergie**")
            st.latex(lm["energy"])
            if lm.get("energy_detail"):
                ui_caption(lm["energy_detail"])
    next_step("discretization", "discrétisation")


# =============================================================================
# 12e. PAGES — DISCRÉTISATION, MAILLAGE, MATRICES, RÉSOLUTION, STABILITÉ
# =============================================================================
FD_OPERATORS = {
    "D+": ("Différence avant", r"\frac{u(x+h)-u(x)}{h}", r"u'(x)", 1, r"\frac h2 u''(\xi)"),
    "D-": ("Différence arrière", r"\frac{u(x)-u(x-h)}{h}", r"u'(x)", 1, r"-\frac h2 u''(\xi)"),
    "D0": ("Différence centrée", r"\frac{u(x+h)-u(x-h)}{2h}", r"u'(x)", 2, r"\frac{h^2}{6}u'''(\xi)"),
    "D2": ("Dérivée seconde centrée", r"\frac{u(x-h)-2u(x)+u(x+h)}{h^2}", r"u''(x)", 2, r"\frac{h^2}{12}u^{(4)}(\xi)"),
}


def _fd_consistency_table(src: str, x0: float, op: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    ok, _ = expr_check(src, ("x",))
    if not ok:
        return None
    e = parse_expression(src, ("x",))
    f = NumFunc(e)
    d1 = NumFunc(sympy.diff(e, X_SYM))
    d2 = NumFunc(sympy.diff(e, X_SYM, 2))
    hs = 0.2 / 2 ** np.arange(0, 9)
    rows, errs = [], []
    for h in hs:
        fx = lambda v: float(f(np.array([v]))[0])  # noqa: E731
        if op == "D+":
            approx, exact = (fx(x0 + h) - fx(x0)) / h, float(d1(np.array([x0]))[0])
        elif op == "D-":
            approx, exact = (fx(x0) - fx(x0 - h)) / h, float(d1(np.array([x0]))[0])
        elif op == "D0":
            approx, exact = (fx(x0 + h) - fx(x0 - h)) / (2 * h), float(d1(np.array([x0]))[0])
        else:
            approx, exact = (fx(x0 - h) - 2 * fx(x0) + fx(x0 + h)) / h ** 2, float(d2(np.array([x0]))[0])
        err = abs(approx - exact)
        errs.append(err)
        rows.append({"h": h, "approximation": approx, "exacte": exact, "erreur": err})
    errs = np.array(errs)
    ords = [None] + [math.log(errs[i - 1] / errs[i]) / math.log(2) if errs[i] > 0 and errs[i - 1] > 0 else None
                     for i in range(1, len(errs))]
    for r, o in zip(rows, ords):
        r["ordre observé"] = o
    return df_numbers(rows, {"h": "{:.3e}", "approximation": "{:.10g}", "exacte": "{:.10g}", "erreur": "{:.3e}",
                             "ordre observé": "{:.3f}"}), hs, errs


def _current_stencil(prob: Problem) -> str:
    cfg = prob.cfg
    if prob.cls == "transport":
        return cfg["time"]["transport_scheme"]
    if prob.cls == "hyperbolic":
        return "leapfrog"
    if prob.cls == "parabolic":
        vt = cfg["time"]["theta"]
        return "explicit" if vt == 0 else ("implicit" if vt == 1 else "cn")
    return "lap1d" if prob.dim == 1 else "lap2d"


def _scheme_latex(prob: Problem) -> list[str]:
    """Schéma discret complet écrit pour le problème courant."""
    cfg = prob.cfg
    out = []
    if prob.cls == "transport":
        sch = cfg["time"]["transport_scheme"]
        out.append({
            "upwind": r"U_j^{n+1} = U_j^n - \lambda\,(U_j^n - U_{j-1}^n)\quad (a>0),\qquad \lambda = \frac{a\Delta t}{h}",
            "ftcs": r"U_j^{n+1} = U_j^n - \frac\lambda2\,(U_{j+1}^n - U_{j-1}^n)",
            "lax_friedrichs": r"U_j^{n+1} = \frac{U_{j+1}^n + U_{j-1}^n}{2} - \frac\lambda2\,(U_{j+1}^n - U_{j-1}^n)",
            "lax_wendroff": r"U_j^{n+1} = U_j^n - \frac\lambda2(U_{j+1}^n - U_{j-1}^n) + \frac{\lambda^2}2(U_{j+1}^n - 2U_j^n + U_{j-1}^n)",
            "btcs": r"U_j^{n+1} + \frac\lambda2\,(U_{j+1}^{n+1} - U_{j-1}^{n+1}) = U_j^n",
        }[sch])
        out.append({"upwind": r"\tau = O(h + \Delta t)", "ftcs": r"\tau = O(h^2 + \Delta t)",
                    "lax_friedrichs": r"\tau = O(h + \Delta t + h^2/\Delta t)", "lax_wendroff": r"\tau = O(h^2 + \Delta t^2)",
                    "btcs": r"\tau = O(h^2 + \Delta t)"}[sch])
        return out
    lap = (r"\frac{U_{j-1}-2U_j+U_{j+1}}{h^2}" if prob.dim == 1 else
           r"\frac{U_{i-1,j}-2U_{ij}+U_{i+1,j}}{h_x^2} + \frac{U_{i,j-1}-2U_{ij}+U_{i,j+1}}{h_y^2}")
    if prob.cls == "elliptic":
        if prob.method == "fdm":
            if prob.k_expr == 1:
                out.append(r"-\left(" + lap + r"\right)" + ("" if prob.c_expr == 0 else r" + c_j U_j") + r" = f_j")
            else:
                out.append(r"-\frac{k_{j+1/2}(U_{j+1}-U_j) - k_{j-1/2}(U_j - U_{j-1})}{h^2} + c_jU_j = f_j,\quad "
                           r"k_{j\pm1/2} = k(x_j \pm h/2)")
            out.append(r"\tau_j = O(h^2)\ \text{(si } u\in C^4)")
        else:
            out.append(r"\text{Trouver } U\in\mathbb R^N:\ \sum_j a(\varphi_j,\varphi_i)\,U_j = L(\varphi_i)\quad\forall i\ "
                       r"\Longleftrightarrow\ KU = F")
        return out
    if prob.method == "fdm":
        out.append(r"\frac{U^{n+1}-U^n}{\Delta t} + \vartheta\,A U^{n+1} + (1-\vartheta)\,AU^n = \vartheta F^{n+1} + "
                   r"(1-\vartheta)F^n\quad (A = -\Delta_h\ \text{et termes d'ordre inférieur})" if prob.cls == "parabolic" else
                   r"\frac{U^{n+1}-2U^n+U^{n-1}}{\Delta t^2} + A\big(\beta U^{n+1} + (1-2\beta)U^n + \beta U^{n-1}\big) = F^n_\beta")
    else:
        out.append(r"(M + \vartheta\Delta t K)\,U^{n+1} = (M - (1-\vartheta)\Delta t K)\,U^n + \Delta t\,"
                   r"(\vartheta F^{n+1} + (1-\vartheta)F^n)" if prob.cls == "parabolic" else
                   r"(M + \beta\Delta t^2K)U^{n+1} = (2M - (1-2\beta)\Delta t^2K)U^n - (M+\beta\Delta t^2K)U^{n-1} + \Delta t^2F_\beta^n")
    if prob.cls == "parabolic":
        vt = cfg["time"]["theta"]
        out.append(r"\tau = O(h^2 + \Delta t^2)\ (\vartheta = \tfrac12)" if abs(vt - 0.5) < 1e-12 else r"\tau = O(h^2 + \Delta t)")
    else:
        out.append(r"\tau = O(h^2 + \Delta t^2)")
    return out


def page_discretization() -> None:
    page_header("Module 5", "Discrétisation",
                "Méthodes disponibles, approximation des dérivées, construction de Galerkin et schémas temporels. "
                "Les paramètres numériques choisis ici sont utilisés par toutes les étapes suivantes.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    cfg = cfg_raw()
    with card("disc_choice"):
        card_title("alt_route", "Méthodes disponibles pour ce problème")
        a, b = st.columns([1, 1.6])
        with a:
            w_select("Méthode", "method", "disc", list(prob.meta["methods"]),
                     {"fdm": "Différences finies (FDM)", "fem": "Éléments finis P1 (FEM)"}, structural=True,
                     horizontal_radio=True, disabled=len(prob.meta["methods"]) == 1 or prob.shape == "lshape")
            reasons = {
                "transport": "Transport : schémas de différences finies spécialisés (décentrement, CFL).",
                "lshape": "Géométrie non rectangulaire : seule la méthode des éléments finis s'applique.",
            }
            key = "transport" if prob.cls == "transport" else ("lshape" if prob.shape == "lshape" else None)
            ui_caption(reasons.get(key, "Les deux méthodes s'appliquent sur un domaine rectangulaire ; les éléments finis "
                                        "acceptent en plus des maillages non structurés et des coefficients peu réguliers."))
        with b:
            rows = []
            for f, m in FAMILIES.items():
                rows.append({"Famille": ("▸ " + m["label"] + " (courant)") if f == prob.family else m["label"],
                             "Classe": CLASS_LABELS[m["class"]],
                             "Dimensions": " / ".join(f"{d}D" for d in m["dims"]),
                             "DF": "oui" if "fdm" in m["methods"] else "—",
                             "EF P1": "oui" if "fem" in m["methods"] else "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)
    tabs = st.tabs(["Paramètres numériques", "Différences finies", "Éléments finis P1", "Schémas en temps"])
    with tabs[0]:
        with card("disc_params"):
            card_title("tune", "Paramètres du schéma")
            if not prob.time_dependent:
                ui_info("Problème stationnaire : pas de discrétisation temporelle. Le solveur linéaire se règle à la "
                        "page « Résolution numérique ».", icon=":material/info:")
                if prob.cls == "elliptic" and prob.method == "fdm":
                    ui_caption("Schéma à 3 points (1D) / 5 points (2D), coefficients k aux demi-points.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if prob.cls == "parabolic":
                        w_slider("Paramètre ϑ du θ-schéma", "time.theta", "disc", 0.0, 1.0, 0.05,
                                 help="ϑ = 0 : Euler explicite ; ϑ = 1/2 : Crank–Nicolson ; ϑ = 1 : Euler implicite.")
                        ui_caption({0.0: "Euler explicite", 0.5: "Crank–Nicolson", 1.0: "Euler implicite"}.get(
                            float(cfg["time"]["theta"]), "θ-schéma général"))
                        if prob.bx or prob.by:
                            if prob.method == "fdm":
                                w_select("Advection (DF)", "adv_scheme", "disc", ["centered", "upwind"],
                                         {"centered": "Centrée (ordre 2)", "upwind": "Décentrée amont (ordre 1)"})
                    elif prob.cls == "hyperbolic":
                        w_select("Schéma", "time.wave_scheme", "disc", list(WAVE_SCHEMES), WAVE_SCHEMES)
                    else:
                        w_select("Schéma de transport", "time.transport_scheme", "disc", list(TRANSPORT_SCHEMES),
                                 TRANSPORT_SCHEMES)
                    if prob.method == "fem" and prob.cls != "transport":
                        w_toggle("Masse condensée (lumping)", "time.lumped", "disc",
                                 help="Remplace M par la matrice diagonale des sommes de lignes : schéma explicite sans "
                                      "résolution de système, limite de stabilité moins sévère, précision O(h²) conservée.")
                with c2:
                    name = {"parabolic": "r = k Δt / h²", "hyperbolic": "σ = c Δt / h", "transport": "λ = |a| Δt / h"}[prob.cls]
                    w_select("Choix du pas de temps", "time.dt_mode", "disc", ["number", "dt", "nt"],
                             {"number": f"par le nombre {name}", "dt": "par Δt", "nt": "par le nombre de pas N_t"})
                    m_ = cfg["time"]["dt_mode"]
                    if m_ == "number":
                        w_number(name, "time.number", "disc", 1e-4, 1e3, 0.05, fmt_="%.4g")
                    elif m_ == "dt":
                        w_number("Δt", "time.dt", "disc", 1e-7, 10.0, 1e-4, fmt_="%.3e")
                    else:
                        w_number("N_t", "time.nt", "disc", 1, int(LIMITS["nt_max"]), 10, is_int=True)
                with c3:
                    w_number("Temps final T", "time.T", "disc", 1e-4, 1e3, 0.05)
                    w_number("Images stockées", "time.frames", "disc", 10, int(LIMITS["frames_max"]), 10, is_int=True,
                             help="Nombre d'instants conservés pour la visualisation de l'évolution.")
                plan = time_plan_current(current_problem())
                if plan and "error" not in plan:
                    nm = plan["numbers"]
                    ui_markdown(badge(f"Δt = {fmt(plan['dt'])}", "blue") + " " + badge(f"N_t = {plan['nt']}", "blue") + " "
                                + " ".join(badge(f"{k} = {fmt(v)}", "gray") for k, v in nm.items()
                                           if k in ("r", "σ", "λ", "Pe_h")), unsafe_allow_html=True)
                    ui_caption(f"Pas d'espace minimal h = {fmt(plan['hmin'])}. Les conditions de stabilité "
                               "correspondantes sont analysées à la page « Stabilité et CFL ».")
                elif plan:
                    ui_error(plan["error"], icon=":material/error:")
        with card("disc_scheme"):
            card_title("draw", "Schéma discret du problème courant")
            for lx in _scheme_latex(current_problem()):
                st.latex(lx)
            cap = "Molécule de calcul : points de la grille (espace × temps) intervenant dans une équation discrète."
            if prob.method == "fem" and prob.cls != "transport":
                cap += (" En éléments finis P1, la ligne i de K couple le nœud i à ses voisins du maillage : pour −Δ sur la "
                        "triangulation structurée « diagonale montante », on retrouve exactement ce stencil, à un "
                        "facteur de normalisation près (h en 1D, h² en 2D) ; sur un maillage quelconque, le stencil "
                        "dépend des angles (coefficients −½(cot α + cot β)).")
            show_fig(fig_stencil(_current_stencil(current_problem()), mode), cap, key="disc_stencil")
    with tabs[1]:
        a, b = st.columns([1.1, 1])
        with a:
            with card("fd_grid"):
                card_title("grid_on", "Grilles uniformes")
                st.latex(r"x_j = x_a + j\,h,\ \ h = \frac{x_b-x_a}{N},\qquad y_k = y_a + k\,h_y,\qquad t_n = n\,\Delta t")
                st.latex(r"U_j^n \approx u(x_j, t_n)")
                ui_markdown("**Approximation des dérivées et erreurs de troncature** (développements de Taylor) :")
                for key_, (name, fo, target, order, err) in FD_OPERATORS.items():
                    st.latex(rf"\text{{{name} : }}\ {fo} = {target} + {err}\qquad (\text{{ordre}}\ {order})")
                st.latex(r"\frac{\partial u}{\partial t}(x_j,t_n)\approx\frac{U_j^{n+1}-U_j^n}{\Delta t}\ (\text{ordre 1}),"
                         r"\qquad \frac{\partial^2 u}{\partial t^2}\approx\frac{U^{n+1}-2U^n+U^{n-1}}{\Delta t^2}\ (\text{ordre 2})")
                ui_caption("Bords de Neumann/Robin : traitement par demi-maille (équivalent au point fantôme centré), "
                           "qui préserve la symétrie de la matrice et l'ordre 2 global.")
        with b:
            with card("fd_check"):
                card_title("science", "Vérification numérique de la consistance")
                c1, c2, c3 = st.columns([1.4, 0.8, 1])
                src = c1.text_input("Fonction u(x)", value="exp(x)*sin(3*x)", key="fd_check_u")
                x0 = c2.number_input("x₀", value=0.4, step=0.1, key="fd_check_x0")
                op = c3.selectbox("Opérateur", list(FD_OPERATORS), format_func=lambda k: FD_OPERATORS[k][0], key="fd_check_op")
                out = _fd_consistency_table(src, float(x0), op)
                if out is None:
                    ui_caption(":red[Expression invalide]")
                else:
                    df, hs, errs = out
                    show_fig(fig_curves(hs, [{"y": errs, "name": "erreur", "mode": "lines+markers"},
                                             {"y": errs[0] * (hs / hs[0]) ** FD_OPERATORS[op][3],
                                              "name": f"pente {FD_OPERATORS[op][3]}", "slot": 2, "width": 1}],
                                        mode, "Erreur de consistance en fonction de h", "h", "|erreur|", 300, True, True),
                             "La pente en échelle log-log est l'ordre de consistance ; pour h très petit, l'erreur "
                             "d'arrondi (≈ ε/h ou ε/h²) finit par dominer.", key="fd_check_fig")
                    st.dataframe(df, hide_index=True, width="stretch", height=250)
    with tabs[2]:
        a, b = st.columns(2)
        with a:
            with card("fem_galerkin"):
                card_title("hive", "Méthode de Galerkin")
                st.latex(r"V_h\subset V,\ \dim V_h = N,\quad V_h = \mathrm{vect}(\varphi_1,\dots,\varphi_N)")
                st.latex(r"u_h = \sum_{j=1}^N U_j\,\varphi_j,\qquad a(u_h, v_h) = L(v_h)\ \ \forall v_h\in V_h")
                st.latex(r"\Longleftrightarrow\ \sum_j \underbrace{a(\varphi_j,\varphi_i)}_{K_{ij}} U_j = "
                         r"\underbrace{L(\varphi_i)}_{F_i}\ \ \forall i\quad\Longleftrightarrow\quad KU = F")
                st.latex(r"V_h\subset H_0^1(\Omega)\ \text{(Dirichlet homogène : nœuds du bord éliminés)}")
                ui_caption("Conformité : V_h ⊂ V, donc la coercivité de a est héritée en dimension finie (K symétrique "
                           "définie positive) et le lemme de Céa s'applique.")
        with b:
            with card("fem_p1"):
                card_title("change_history", "Élément fini P1 de Lagrange")
                st.latex(r"P_1(K) = \{\,a + bx + cy\ :\ a,b,c\in\mathbb R\,\},\quad \dim P_1(K) = 3")
                st.latex(r"F_K(\hat{\mathbf x}) = A_1 + B_K\hat{\mathbf x},\quad B_K = [A_2-A_1,\ A_3-A_1],\quad "
                         r"|K| = \tfrac12|\det B_K|")
                st.latex(r"\lambda_i(A_j) = \delta_{ij},\qquad \lambda_1+\lambda_2+\lambda_3 = 1,\qquad "
                         r"\nabla\lambda_i\ \text{constant sur } K")
                st.latex(r"K^{(e)}_{ij} = \int_K k\,\nabla\lambda_j\cdot\nabla\lambda_i = \bar k_K\,|K|\,\nabla\lambda_i\cdot\nabla\lambda_j,"
                         r"\qquad M^{(e)} = \frac{|K|}{12}\begin{pmatrix}2&1&1\\1&2&1\\1&1&2\end{pmatrix}")
                ui_caption("Les degrés de liberté sont les valeurs aux sommets ; la continuité aux interfaces est "
                           "automatique (une fonction affine sur une arête est fixée par ses deux extrémités).")
        with card("fem_semi"):
            card_title("timeline", "Semi-discrétisation en espace")
            st.latex(r"M\frac{dU}{dt} + KU = F(t)\ \ \text{(paraboliques)},\qquad M\frac{d^2U}{dt^2} + c^2KU = F(t)\ \ "
                     r"\text{(ondes)},\qquad KU = F\ \ \text{(elliptiques)}")
            ui_caption("M n'est pas l'identité : en explicite, chaque pas exige une résolution avec M, sauf si l'on "
                       "condense la masse (M ≈ M_L diagonale).")
    with tabs[3]:
        with card("time_schemes"):
            card_title("schedule", "Schémas temporels pour U′ = F(U)")
            st.latex(r"\text{Euler explicite : } U^{n+1} = U^n + \Delta t\,F(U^n)")
            st.latex(r"\text{Euler implicite : } U^{n+1} = U^n + \Delta t\,F(U^{n+1})")
            st.latex(r"\text{Crank–Nicolson : } U^{n+1} - U^n = \frac{\Delta t}{2}\big[F(U^n) + F(U^{n+1})\big]")
            st.latex(r"\text{θ-schéma : } U^{n+1} - U^n = \Delta t\big[(1-\vartheta)F(U^n) + \vartheta F(U^{n+1})\big]"
                     r"\quad(\text{paramètre noté }\alpha\text{ dans certains cours})")
            st.latex(r"\text{Version éléments finis : } (M + \vartheta\Delta t K)U^{n+1} = (M - (1-\vartheta)\Delta tK)U^n + "
                     r"\Delta t\big(\vartheta F^{n+1} + (1-\vartheta)F^n\big)")
            st.latex(r"\text{Ondes (Newmark, } \gamma=\tfrac12) : (M + \beta\Delta t^2K)U^{n+1} = (2M-(1-2\beta)\Delta t^2K)U^n "
                     r"- (M+\beta\Delta t^2K)U^{n-1} + \Delta t^2F_\beta^n")
            st.dataframe(pd.DataFrame([
                {"Schéma": "Euler explicite (ϑ = 0)", "Ordre en temps": 1, "Stabilité (chaleur)": "r ≤ 1/2 (DF)", "Coût par pas": "produit matrice-vecteur"},
                {"Schéma": "Crank–Nicolson (ϑ = 1/2)", "Ordre en temps": 2, "Stabilité (chaleur)": "inconditionnelle (A-stable)", "Coût par pas": "1 résolution (factorisation réutilisée)"},
                {"Schéma": "Euler implicite (ϑ = 1)", "Ordre en temps": 1, "Stabilité (chaleur)": "inconditionnelle (L-stable)", "Coût par pas": "1 résolution"},
                {"Schéma": "Saute-mouton (ondes)", "Ordre en temps": 2, "Stabilité (chaleur)": "σ ≤ 1 (CFL)", "Coût par pas": "explicite"},
                {"Schéma": "Newmark β = 1/4 (ondes)", "Ordre en temps": 2, "Stabilité (chaleur)": "inconditionnelle", "Coût par pas": "1 résolution"},
            ]), hide_index=True, width="stretch")
    next_step("mesh", "maillage")


def page_mesh() -> None:
    page_header("Module 5 bis", "Maillage et fonctions de forme",
                "Construction de la grille (différences finies) ou de la triangulation (éléments finis), raffinement, "
                "qualité, fonctions de base P1 et reconstruction u_h = Σ Uᵢ φᵢ.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    cfg = cfg_raw()
    a, b = st.columns([1, 2.1])
    with a:
        with card("mesh_params"):
            card_title("tune", "Paramètres")
            lim1, lim2 = int(LIMITS["n1d_max"]), int(LIMITS["n2d_max"])
            if prob.dim == 1:
                w_number("Nombre d'intervalles N", "mesh.nx", "mesh", 2, lim1, 4, is_int=True)
            else:
                w_number("Mailles en x (Nₓ)" + (" — pair" if prob.shape == "lshape" else ""), "mesh.nx", "mesh", 2, lim2, 2, is_int=True)
                if prob.shape != "lshape":
                    w_number("Mailles en y (N_y)", "mesh.ny", "mesh", 2, lim2, 2, is_int=True)
            if prob.method == "fem" and prob.cls != "transport":
                if prob.dim == 2:
                    w_select("Motif de triangulation", "mesh.pattern", "mesh", list(PATTERN_LABELS), PATTERN_LABELS)
                if prob.shape != "lshape":
                    w_select("Répartition des nœuds", "mesh.grading", "mesh", list(GRADING_LABELS), GRADING_LABELS)
                    if cfg["mesh"]["grading"] == "perturbed":
                        w_slider("Amplitude de perturbation (× h)", "mesh.perturb", "mesh", 0.0, 0.3, 0.01)
                        w_number("Graine aléatoire", "mesh.seed", "mesh", 0, 10000, 1, is_int=True)
                w_number("Raffinements uniformes (chaque élément → 2ᵈ)", "mesh.refine", "mesh", 0, 4, 1, is_int=True)
            else:
                ui_caption("Différences finies : grille uniforme (les formules d'ordre 2 supposent un pas constant).")
            show_ids = st.toggle("Afficher la numérotation (petits maillages)", value=False, key="mesh_ids")
    if prob.cls == "transport":
        with b:
            with card("mesh_tr"):
                card_title("grid_on", "Grille 1D du transport")
                nx = cfg["mesh"]["nx"]
                h = (prob.xb - prob.xa) / nx
                x = prob.xa + h * np.arange(nx + (0 if prob.transport_bc == "periodic" else 1))
                cats = np.array(["interior"] * len(x), dtype=object)
                if prob.transport_bc == "inflow":
                    cats[0 if (prob.a or 1) > 0 else -1] = "dirichlet"
                show_fig(fig_mesh(x[:, None], None, cats, mode, f"{len(x)} nœuds, h = {fmt(h)}"),
                         "Grille périodique : x_N ≡ x_0. Condition d'entrée : nœud amont imposé (bleu).", key="mesh_tr_fig")
        next_step("matrices", "matrices et assemblage")
        return
    asm = cached_assembly(cfg_json_cur())
    if not asm.get("ok"):
        with b:
            ui_error("; ".join(asm.get("errors", [])), icon=":material/error:")
        return
    store_result("assembly", True)
    cats = node_categories(prob, asm["nodes"], asm["side_nodes"])
    with b:
        with card("mesh_fig"):
            card_title("change_history" if prob.method == "fem" else "grid_on",
                       "Triangulation" if prob.method == "fem" and prob.dim == 2 else "Maillage / grille")
            n = asm["n"]
            if n > 40000:
                ui_info(f"{n} nœuds : affichage du maillage désactivé (trop dense pour être lisible).", icon=":material/info:")
            else:
                show_fig(fig_mesh(asm["nodes"], asm["elems"], cats, mode, None, grid=asm.get("grid"), show_ids=show_ids,
                                  height=480 if prob.dim == 2 else 240),
                         "Nœuds colorés selon la condition aux limites ; gris : nœuds intérieurs (inconnues). "
                         "Les coins appartenant à un bord de Dirichlet sont de type Dirichlet.", key="mesh_main")
    q = asm["quality"]
    c1, c2 = st.columns([1, 1.4])
    with c1:
        with card("mesh_stats"):
            card_title("analytics", "Statistiques")
            nel = len(asm["elems"]) if asm["elems"] is not None else None
            rows = [("Nœuds", asm["n"]), ("Éléments / mailles", nel if nel is not None else
                                         f"{asm['grid']['nx']} × {max(asm['grid']['ny'], 1)}"),
                    ("Inconnues (nœuds libres)", len(asm["free"])), ("Nœuds de Dirichlet", len(asm["dir_idx"])),
                    ("h_max", fmt(q.get("h_max"))), ("h_min", fmt(q.get("h_min")))]
            if "angle_min" in q:
                rows += [("Angle minimal", f"{q['angle_min']:.2f}°"), ("Angle maximal", f"{q['angle_max']:.2f}°"),
                         ("Régularité max h_K/(2ρ_K)", fmt(q["shape_reg"])),
                         ("Maillage non obtus", "oui" if q["nonobtuse"] else "non")]
            kv_table(rows)
            if "angle_min" in q:
                ui_caption("Non obtus ⇒ la matrice de rigidité de −Δ est une M-matrice (principe du maximum discret). "
                           "La régularité (angles minorés) conditionne les estimations d'interpolation.")
    with c2:
        if prob.method == "fem":
            with card("mesh_shape"):
                card_title("shapes", "Fonctions de forme locales")
                if prob.dim == 2:
                    show_fig(fig_reference_triangle_functions(mode),
                             "Sur le triangle de référence : chaque λᵢ vaut 1 au sommet Aᵢ et 0 aux deux autres.",
                             key="mesh_lambda")
                    P = asm["nodes"][asm["elems"][0]]
                    mesh0 = Mesh(P, np.array([[0, 1, 2]]), np.zeros((0, 2), int), np.zeros(0, int), ["boundary"])
                    G = p1_geometry(mesh0)["grad"][0]
                    lam = np.array([[1 + G[0] @ (P[j] - P[0]) if i == 0 else G[i] @ (P[j] - P[0]) for j in range(3)]
                                    for i in range(3)])
                    ui_markdown("Vérification sur l'élément K₀ du maillage (valeurs λᵢ(Aⱼ)) :")
                    st.dataframe(pd.DataFrame(np.round(lam, 14), index=["λ₁", "λ₂", "λ₃"], columns=["A₁", "A₂", "A₃"]),
                                 width="stretch")
                    ui_caption(f"Σᵢ ∇λᵢ = ({G.sum(axis=0)[0]:.1e}, {G.sum(axis=0)[1]:.1e}) ⇒ λ₁ + λ₂ + λ₃ ≡ 1.")
                else:
                    xn = np.sort(asm["nodes"][:, 0])
                    show_fig(fig_hat_functions(xn, mode, highlight=len(xn) // 2),
                             "Fonctions chapeaux φᵢ : affines par morceaux, φᵢ(xⱼ) = δᵢⱼ, support [x_{i−1}, x_{i+1}].",
                             key="mesh_hat")
    if prob.method == "fem":
        with card("mesh_recon"):
            card_title("layers", "Reconstruction u_h = Σᵢ Uᵢ φᵢ (surface affine par morceaux)")
            default = cfg["exact"] if cfg["exact"].strip() and "t" not in cfg["exact"] else (
                "sin(pi*x)*sin(pi*y)" if prob.dim == 2 else "sin(pi*x)")
            src = st.text_input("Fonction à interpoler aux nœuds", value=default, key="mesh_interp_src")
            ok, info = expr_check(src, prob.space_vars)
            if not ok:
                ui_caption(f":red[{info}]")
            else:
                fn = NumFunc(parse_expression(src, prob.space_vars))
                U = _eval_xy(fn, asm["nodes"])
                fake = {"method": "fem", "nodes": asm["nodes"], "elems": asm["elems"], "grid": None}
                if prob.dim == 2:
                    show_fig(fig_field_2d(fake, U, mode, "surface", "Interpolé P1 Π_h v", "Π_h v", show_mesh=True,
                                          height=470),
                             "Chaque triangle porte un morceau de plan : u_h est continue, affine par élément, et "
                             "coïncide avec v aux nœuds.", key="mesh_recon_fig")
                else:
                    xn = asm["nodes"][:, 0]
                    o = np.argsort(xn)
                    show_fig(fig_hat_functions(xn[o], mode, coeffs=U[o]),
                             "u_h = Σ Uᵢ φᵢ : combinaison des fonctions chapeaux pondérées par les valeurs nodales.",
                             key="mesh_recon_1d")
    next_step("matrices", "matrices et assemblage")


def _matrix_df(A: np.ndarray, r0: int = 0, c0: int = 0) -> pd.DataFrame:
    return pd.DataFrame(A, index=[str(r0 + i) for i in range(A.shape[0])],
                        columns=[str(c0 + j) for j in range(A.shape[1])]).map(lambda v: f"{v:.5g}")


def _mtx_bytes(A: sp.spmatrix) -> bytes:
    buf = io.BytesIO()
    sio.mmwrite(buf, A.tocoo())
    return buf.getvalue()


def page_matrices() -> None:
    page_header("Module 6", "Matrices et assemblage",
                "Matrices élémentaires, assemblage global, second membre, propriétés (symétrie, positivité, M-matrice, "
                "conditionnement) et structure creuse.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    if prob.cls == "transport":
        ui_info("Transport : chaque pas est une mise à jour explicite (ou un système bidiagonal/tridiagonal pour le "
                "schéma implicite), décrite à la page « Discrétisation ».", icon=":material/info:")
        return
    asm = cached_assembly(cfg_json_cur())
    if not asm.get("ok"):
        ui_error("; ".join(asm.get("errors", [])), icon=":material/error:")
        return
    store_result("assembly", True)
    Fr, Dd = asm["free"], asm["dir_idx"]
    K, M = asm["K"], asm["M"]
    KFF = K[Fr][:, Fr] if not asm["kernel"] else K
    with card("mat_system"):
        card_title("data_array", "Système discret assemblé")
        if prob.method == "fdm":
            st.latex(r"K = D\,A,\quad D = \mathrm{diag}(|V_j|)\ \text{(volumes de contrôle)},\quad A = D^{-1}K\ "
                     r"\text{(matrice des différences finies)}")
        if prob.cls == "elliptic":
            st.latex(r"K_{FF}\,U_F = F_F - K_{FD}\,G_D" if len(Dd) else r"KU = F")
        elif prob.cls == "parabolic":
            st.latex(r"M\frac{dU}{dt} + KU = F(t)" + (r",\quad M = D\ \text{(diagonale)}" if prob.method == "fdm" else ""))
        else:
            st.latex(r"M\frac{d^2U}{dt^2} + KU = F(t),\quad K = c^2 K_{\Delta}")
        names = {"K_diffusion": r"K_{ij}=\int_\Omega k\nabla\varphi_j\cdot\nabla\varphi_i", "M_masse": r"M_{ij}=\int_\Omega\varphi_j\varphi_i",
                 "M_reaction": r"\int_\Omega c\,\varphi_j\varphi_i", "B_advection": r"\int_\Omega(\mathbf b\cdot\nabla\varphi_j)\varphi_i",
                 "R_robin": r"\int_{\Gamma_R}\beta\varphi_j\varphi_i", "S_diffusion": r"\text{flux } k_{j\pm1/2}(U_{j\pm1}-U_j)/h",
                 "C_reaction": r"c_j|V_j|"}
        comps = asm["components"]
        ui_markdown("Contributions assemblées : " + ", ".join(f"${names.get(k, k)}$" for k in comps))
        c = st.columns(4)
        c[0].metric("Inconnues", f"{len(Fr):,}".replace(",", " "), border=True)
        c[1].metric("Nœuds de Dirichlet", len(Dd), border=True)
        c[2].metric("Non-zéros de K_FF", f"{KFF.nnz:,}".replace(",", " "), border=True)
        c[3].metric("Temps d'assemblage", f"{asm['info'].get('assembly_time', 0) * 1e3:.1f} ms", border=True)
    a, b = st.columns([1, 1.15])
    with a:
        with card("mat_props"):
            card_title("fact_check", "Propriétés de la matrice du système")
            if KFF.shape[0] > 70000:
                ui_info("Matrice trop grande pour l'analyse spectrale interactive.", icon=":material/info:")
                props = {}
            else:
                props = cached_matrix_props(cfg_json_cur(), "K")
            if props:
                rows = [("Dimension", f"{props['n']} × {props['n']}"), ("Non-zéros", props["nnz"]),
                        ("Densité", f"{props['density']:.2e}"), ("Non-zéros max par ligne", props["max_row_nnz"]),
                        ("Largeur de bande", props.get("bandwidth")),
                        ("Symétrique", "oui" if props["symmetric"] else "non"),
                        ("Diagonale > 0", "oui" if props["diag_positive"] else "non"),
                        ("Hors-diagonale ≤ 0", "oui" if props["offdiag_nonpositive"] else "non"),
                        ("Diagonale dominante", "oui" if props["diag_dominant"] else "non"),
                        ("Irréductible", "oui" if props["irreducible"] else "non"),
                        ("M-matrice", "oui" if props["m_matrix"] else "non (ou non établi)")]
                if props.get("spd") is not None:
                    rows += [("Définie positive", "oui" if props["spd"] else "non"),
                             ("λ_min", fmt(props.get("lam_min"))), ("λ_max", fmt(props.get("lam_max"))),
                             ("Conditionnement κ₂", fmt(props.get("cond2")))]
                kv_table(rows)
                msgs = []
                if props["symmetric"]:
                    msgs.append("Symétrie : héritée de la symétrie de a(u, v) (pas de terme d'advection).")
                elif "B_advection" in comps:
                    msgs.append("Non symétrique : le terme d'advection est antisymétrique (∫(b·∇φⱼ)φᵢ = −∫(b·∇φᵢ)φⱼ + bord).")
                if props.get("spd"):
                    msgs.append("Définie positive : traduction discrète de la coercivité ; Cholesky ou CG applicables.")
                if props["m_matrix"]:
                    msgs.append("M-matrice : A⁻¹ ≥ 0, principe du maximum discret, stabilité en norme ∞.")
                if props.get("cond2") and math.isfinite(props["cond2"]):
                    msgs.append(f"κ₂ = {fmt(props['cond2'])} : croît comme h⁻² ; il gouverne la vitesse des solveurs "
                                "itératifs (CG : ~√κ itérations).")
                if asm["kernel"]:
                    msgs.append("Neumann pur : K est singulière (K·1 = 0) ; on ajoute la contrainte ∫u_h = 0.")
                for m_ in msgs:
                    ui_caption("• " + m_)
    with b:
        with card("mat_spy"):
            card_title("blur_on", "Structure creuse (sparsity pattern)")
            show_fig(fig_spy(KFF, mode, f"K_FF — {KFF.shape[0]} × {KFF.shape[0]}, {KFF.nnz} non-zéros", 420),
                     "Chaque point est un coefficient non nul. Numérotation lexicographique : bande de largeur ≈ Nₓ en 2D ; "
                     "le remplissage (fill-in) de la factorisation LU dépend de cette structure.", key="mat_spy_fig")
    if prob.method == "fem":
        with card("mat_elem"):
            card_title("change_history", "Matrices élémentaires et assemblage")
            ne = len(asm["elems"])
            e = int(st.number_input("Élément K_e", min_value=0, max_value=ne - 1, value=0, step=1, key="mat_elem_idx"))
            mesh = Mesh(asm["nodes"], asm["elems"], np.zeros((0, prob.dim), int), np.zeros(0, int), ["boundary"])
            em = element_matrices(mesh, prob, e)
            c1, c2 = st.columns([1, 1.6])
            with c1:
                ui_markdown("**Géométrie**")
                coords = pd.DataFrame(em["coords"], columns=["x", "y"][:prob.dim])
                coords.insert(0, "nœud global", em["nodes"])
                coords.insert(0, "local", [f"A{i + 1}" for i in range(len(em["nodes"]))])
                st.dataframe(coords, hide_index=True, width="stretch")
                ui_markdown(f"Mesure |K| = {fmt(em['meas'], 6)} ; moyenne de k sur K : k̄ = {fmt(em['kbar'], 6)}")
                gdf = pd.DataFrame(em["grad"], columns=["∂/∂x", "∂/∂y"][:prob.dim], index=[f"∇λ{i + 1}" for i in range(len(em["nodes"]))])
                st.dataframe(gdf.map(lambda v: f"{v:.5g}"), width="stretch")
            with c2:
                t1, t2, t3 = st.tabs(["Rigidité K^(e)", "Masse M^(e)", "Second membre F^(e)"])
                with t1:
                    st.dataframe(_matrix_df(em["K"]), width="stretch")
                    ui_caption("K^(e)_ij = k̄ |K| ∇λⱼ·∇λᵢ (gradients constants). Somme de chaque ligne nulle : les "
                               "constantes sont dans le noyau local.")
                with t2:
                    st.dataframe(_matrix_df(em["M"]), width="stretch")
                    ui_caption(f"Somme des coefficients = |K| = {fmt(em['meas'], 6)}.")
                with t3:
                    st.dataframe(pd.DataFrame({"F^(e)_i": [f"{v:.6g}" for v in em["F"]]},
                                              index=[f"i = {k}" for k in em["nodes"]]), width="stretch")
                    ui_caption("F^(e)_i = ∫_K f λᵢ (quadrature de degré 4 en 2D, Gauss 3 points en 1D).")
            ui_caption("Assemblage : K_{g(i), g(j)} += K^(e)_{ij}, où g est la correspondance numérotation locale → "
                       "globale ci-dessus ; les contributions de bord (Neumann, Robin) sont assemblées sur les arêtes.")
    with card("mat_global"):
        card_title("table", "Matrice globale et second membre")
        F = asm["F"]
        n = K.shape[0]
        if n <= int(LIMITS["matrix_table_max"]):
            Kd = (K.toarray() if prob.method == "fem" else (sp.diags(1 / np.asarray(M.diagonal())) @ K).toarray())
            ui_markdown("Matrice complète (avant élimination des nœuds de Dirichlet)" +
                        (" — matrice DF A = D⁻¹K" if prob.method == "fdm" else ""))
            st.dataframe(_matrix_df(Kd), width="stretch")
            show_fig(fig_matrix_values(Kd, mode, "Valeurs des coefficients"), key="mat_vals")
        else:
            m_ = 10
            A_show = (sp.diags(1 / np.asarray(M.diagonal())) @ K).tocsr() if prob.method == "fdm" else K
            free_block = A_show[Fr][:, Fr][:m_, :m_].toarray() if len(Fr) else np.zeros((0, 0))
            ui_markdown(f"Extrait représentatif : bloc 10 × 10 de la matrice {'DF A' if prob.method == 'fdm' else 'K'} "
                        f"restreinte aux inconnues (dimension totale {len(Fr)}).")
            st.dataframe(_matrix_df(free_block), width="stretch")
        ui_markdown("Second membre F (premières composantes, nœuds libres) :")
        st.dataframe(pd.DataFrame({"nœud": Fr[:12], "F": [f"{v:.6g}" for v in F[Fr][:12]]}).T, width="stretch")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("K (Matrix Market)", data=lambda: _mtx_bytes(K), file_name="K.mtx", mime="text/plain",
                           icon=":material/download:", on_click="ignore", width="stretch")
        d2.download_button("M (Matrix Market)", data=lambda: _mtx_bytes(M), file_name="M.mtx", mime="text/plain",
                           icon=":material/download:", on_click="ignore", width="stretch")
        d3.download_button("F (CSV)", data=lambda: pd.DataFrame({"F": F}).to_csv(index_label="noeud"),
                           file_name="F.csv", mime="text/csv", icon=":material/download:", on_click="ignore", width="stretch")
        if n <= int(LIMITS["dense_csv_max"]):
            d4.download_button("K dense (CSV)", data=lambda: pd.DataFrame(K.toarray()).to_csv(index=False),
                               file_name="K_dense.csv", mime="text/csv", icon=":material/download:", on_click="ignore",
                               width="stretch")
        else:
            d4.caption(f"Export dense désactivé (n > {int(LIMITS['dense_csv_max'])}).")
    next_step("solve", "résolution numérique")


def page_solve() -> None:
    page_header("Module 7", "Résolution numérique",
                "Solveurs directs (factorisation creuse) et itératifs pour les problèmes stationnaires ; θ-schéma, "
                "saute-mouton et Newmark pour les problèmes d'évolution, avec la matrice de masse correctement traitée.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    cfg = cfg_raw()
    with card("solve_params"):
        card_title("settings", "Paramètres du solveur")
        if not prob.time_dependent:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                w_select("Solveur linéaire", "solver.kind", "solve", list(SOLVER_LABELS), SOLVER_LABELS)
            with c2:
                w_select("Préconditionneur", "solver.precond", "solve", list(PRECOND_LABELS), PRECOND_LABELS,
                         disabled=cfg["solver"]["kind"] == "direct")
            with c3:
                w_number("Tolérance relative", "solver.tol", "solve", 1e-14, 1e-2, 1e-10, fmt_="%.1e",
                         disabled=cfg["solver"]["kind"] == "direct")
            with c4:
                w_number("Itérations max", "solver.maxiter", "solve", 10, 100000, 100, is_int=True,
                         disabled=cfg["solver"]["kind"] == "direct")
            ui_caption("Direct : SuperLU avec renumérotation COLAMD (robuste). CG : matrices symétriques définies "
                       "positives. GMRES/BiCGSTAB : cas général. Neumann pur : système augmenté résolu en direct.")
        else:
            plan = time_plan_current(prob)
            if plan and "error" not in plan:
                desc = (f"ϑ = {cfg['time']['theta']:g}" if prob.cls == "parabolic" else
                        WAVE_SCHEMES[cfg["time"]["wave_scheme"]] if prob.cls == "hyperbolic" else
                        TRANSPORT_SCHEMES[cfg["time"]["transport_scheme"]])
                ui_markdown(badge(desc, "blue") + " " + badge(f"Δt = {fmt(plan['dt'])}", "gray") + " " +
                            badge(f"N_t = {plan['nt']}", "gray") + " " +
                            " ".join(badge(f"{k} = {fmt(v)}", "gray") for k, v in plan["numbers"].items()
                                     if k in ("r", "σ", "λ", "Pe_h")), unsafe_allow_html=True)
                ui_caption("Les systèmes linéaires de chaque pas (implicites ou masse cohérente) sont factorisés une "
                           "seule fois (SuperLU), puis réutilisés à chaque pas.")
            elif plan:
                ui_error(plan["error"], icon=":material/error:")
            nav_button("discretization", "Modifier le schéma en temps", icon=":material/tune:", key="solve_go_disc")
        if st.button("Résoudre", type="primary", icon=":material/play_arrow:", key="solve_btn"):
            with st.spinner("Résolution en cours…"):
                solve_current()
    res = get_result("solve")
    if res is None:
        stale = get_result("solve", allow_stale=True)
        ui_info("Aucune solution pour la configuration actuelle" + (" (résultat précédent obsolète)" if stale else "")
                + ".", icon=":material/info:")
        return
    if not res["ok"]:
        ui_error("; ".join(res["errors"]), icon=":material/error:")
        return
    for w in res.get("warnings", []):
        ui_warning(w, icon=":material/warning:")
    if res.get("blowup"):
        b_ = res["blowup"]
        ui_error(f"Explosion numérique détectée au pas {b_['step']} (t = {fmt(b_['time'])}, max|u| = {fmt(b_['value'])}). "
                 "Le schéma est instable pour ces paramètres : voir « Stabilité et CFL ».", icon=":material/warning:")
    c = st.columns(5)
    c[0].metric("Inconnues", f"{res['dofs']:,}".replace(",", " "), border=True)
    c[1].metric("Assemblage", f"{res['timings'].get('assembly', 0) * 1e3:.1f} ms", border=True)
    if prob.time_dependent:
        c[2].metric("Boucle en temps", f"{res['timings'].get('time_loop', 0):.3f} s", border=True)
        c[3].metric("Pas effectués", f"{res.get('steps_done', res.get('nt'))} / {res.get('nt')}", border=True)
    else:
        sv = res["solver"]
        c[2].metric("Résolution", f"{sv.get('time', 0) * 1e3:.1f} ms", border=True)
        c[3].metric("Résidu relatif", f"{sv.get('residual', 0):.1e}", border=True)
    errs = res.get("errors") or {}
    c[4].metric("Erreur L² relative", fmt(errs.get("l2_rel")) if errs else "—", border=True)
    a, b = st.columns([1.35, 1])
    with a:
        with card("solve_fig"):
            card_title("insights", "Solution calculée")
            U = np.asarray(res["U"])
            if prob.dim == 1:
                show_fig(fig_solution_1d(res["nodes"][:, 0], U, res.get("Uex"), mode,
                                         "Solution numérique et solution exacte" + (f" à t = {fmt(res.get('t_final', prob.T))}"
                                                                                     if prob.time_dependent else "")),
                         "Trait gris : solution exacte ; bleu : solution numérique aux nœuds.", key="solve_sol")
            else:
                show_fig(fig_field_2d(res, U, mode, "surface", "Solution numérique u_h", height=470), key="solve_sol2d")
    with b:
        with card("solve_info"):
            card_title("receipt_long", "Rapport du solveur")
            rows = []
            if not prob.time_dependent:
                sv = res["solver"]
                rows += [("Méthode", sv.get("kind")), ("Matrice symétrique", "oui" if sv.get("symmetric") else "non"),
                         ("Itérations", sv.get("iterations")), ("Résidu ‖b − Ax‖/‖b‖", f"{sv.get('residual', 0):.2e}")]
                if sv.get("fill"):
                    rows.append(("Remplissage nnz(L+U)", f"{sv['fill']} (nnz(A) = {sv.get('nnz')})"))
                if sv.get("note"):
                    rows.append(("Remarque", sv["note"]))
                if res.get("kernel"):
                    rows.append(("Multiplicateur de Lagrange", fmt(sv.get("multiplier"))))
            else:
                rows += [("Schéma", res.get("scheme")), ("Δt", fmt(res.get("dt"))), ("N_t", res.get("nt")),
                         ("Factorisation", res.get("factor_info"))]
                for k, v in (res.get("numbers") or {}).items():
                    if k in ("r", "σ", "λ", "Pe_h", "CFL_adv"):
                        rows.append((k, fmt(v)))
            for k, lab in (("l2", "‖e‖_L²"), ("h1", "|e|_H¹"), ("max", "‖e‖_∞"), ("l2_rel", "erreur L² relative")):
                if errs.get(k) is not None:
                    rows.append((lab, fmt(errs[k])))
            kv_table(rows)
        sv = res.get("solver") or {}
        if sv.get("history"):
            with card("solve_hist"):
                card_title("show_chart", "Historique de convergence du solveur itératif")
                h = np.array(sv["history"])
                show_fig(fig_curves(np.arange(1, len(h) + 1), [{"y": h, "name": "résidu relatif"}], mode, None,
                                    "itération", "résidu relatif", 300, logy=True),
                         "GMRES : résidu préconditionné ; CG/BiCGSTAB : résidu vrai ‖b − Ax_k‖/‖b‖.", key="solve_hist_fig")
    if prob.time_dependent and res.get("hist"):
        h = res["hist"]
        c1, c2 = st.columns(2)
        with c1:
            with card("solve_norm"):
                if "energy" in h:
                    card_title("bolt", "Énergie discrète")
                    show_fig(fig_curves(h["t"], [{"y": h["energy"], "name": "E^{n+1/2}"}], mode, None, "t", "énergie", 300),
                             "½VᵀM̃V + ½ŪᵀKŪ : exactement conservée par le schéma (en l'absence de source).", key="solve_energy")
                else:
                    card_title("trending_down", "Norme L² de u_h au cours du temps")
                    show_fig(fig_curves(h["t"], [{"y": h["norm"], "name": "‖u_h(t)‖_L²"}], mode, None, "t", "norme",
                                        300, logy=bool(np.all(np.asarray(h["norm"]) > 0))), key="solve_normfig")
        with c2:
            if h.get("err") is not None:
                with card("solve_errt"):
                    card_title("query_stats", "Erreur au cours du temps")
                    E = np.asarray(h["err"])
                    show_fig(fig_curves(E[:, 0], [{"y": E[:, 1], "name": "‖e‖ discrète L²"},
                                                  {"y": E[:, 2], "name": "‖e‖_∞ nodale"}], mode, None, "t", "erreur",
                                        300, logy=bool(np.all(E[:, 1:] > 0))),
                             "Accumulation des erreurs de troncature, contrôlée par la stabilité.", key="solve_errt_fig")
    nav_button("report", "Synthèse et interprétation des résultats", icon=":material/summarize:", key="solve_to_report")
    next_step("stability", "stabilité et CFL")


# ------------------------------------------------------------------ stabilité
def _vn_theory() -> None:
    a, b = st.columns([1.1, 1])
    with a:
        with card("vn_theory"):
            card_title("school", "Principe de l'analyse de von Neumann")
            st.latex(r"e_j^n = G^n\,e^{ij\theta},\qquad \theta = \xi h\in[-\pi,\pi]")
            ui_markdown(r"- **Mode de Fourier** : la perturbation $e$ (erreur d'arrondi, de troncature) est décomposée "
                        r"en ondes discrètes $e^{ij\theta}$." "\n"
                        r"- **Nombre d'onde** $\xi$, angle de phase $\theta = \xi h$ : $\theta\approx 0$ basses "
                        r"fréquences (bien résolues), $\theta = \pi$ mode « dent de scie » $(-1)^j$, le plus oscillant "
                        r"représentable." "\n"
                        r"- **Facteur d'amplification** $G(\theta)$ : multiplicateur d'un mode à chaque pas, "
                        r"$e^{n+1} = G\,e^n$." "\n"
                        r"- **Signe de G** : $G>0$ amortissement ou croissance monotone ; $G<0$ le mode change de signe "
                        r"à chaque pas (oscillation en temps)." "\n"
                        r"- $|G|<1$ : mode amorti ; $|G| = 1$ : conservé ; $|G|>1$ : amplifié — après $n$ pas, $|G|^n$.")
            st.latex(r"\text{Critère : } |G(\theta)|\le 1\ \ (\text{ou } 1 + C\Delta t)\quad\forall\theta\in[-\pi,\pi]")
            ui_caption("Valable pour un schéma linéaire à coefficients constants sur grille uniforme périodique "
                       "(ou dont la frontière ne modifie pas le spectre) : c'est alors une condition nécessaire et "
                       "suffisante de stabilité L².")
    with b:
        with card("vn_heat"):
            card_title("thermostat", "Chaleur 1D, Euler explicite : démonstration de r ≤ 1/2")
            st.latex(r"\begin{aligned}&U_j^{n+1} = U_j^n + r\,(U_{j+1}^n - 2U_j^n + U_{j-1}^n),\\ "
                     r"&r = \frac{\kappa\,\Delta t}{h^2},\qquad e_j^n = G^n e^{ij\theta}\end{aligned}")
            st.latex(r"\begin{aligned}G &= 1 + r\,(e^{i\theta} - 2 + e^{-i\theta})\\ &= 1 - 2r(1-\cos\theta)"
                     r" = 1 - 4r\sin^2\tfrac\theta2\end{aligned}")
            st.latex(r"\begin{aligned}|G|\le 1 &\iff -1\le 1-4r\sin^2\tfrac\theta2\le 1\\ "
                     r"&\iff 4r\sin^2\tfrac\theta2\le 2\quad\forall\theta\end{aligned}")
            st.latex(r"\text{pire cas } \theta = \pi\ (\sin^2\tfrac\pi2 = 1):\qquad \boxed{\,r\le\tfrac12\,}")
            ui_caption("Si r > 1/2, G(π) = 1 − 4r < −1 : le mode dent de scie change de signe et croît à chaque pas — "
                       "c'est l'instabilité oscillante typique. Pour Euler implicite, G = 1/(1 + 4r sin²(θ/2)) ∈ ]0, 1] "
                       "pour tout r : stabilité inconditionnelle.")


def _vn_explorer() -> None:
    mode = theme_mode()
    th = np.linspace(-np.pi, np.pi, 1201)
    fam = st.segmented_control("Famille de schémas", ["Chaleur 1D", "Chaleur 2D", "Advection–diffusion 1D", "Ondes 1D",
                                                       "Transport"], default="Chaleur 1D", key="vnx_family",
                               required=True) or "Chaleur 1D"
    if fam == "Chaleur 1D":
        c1, c2 = st.columns([1, 2])
        with c1:
            vt = st.slider("ϑ (0 = explicite, ½ = CN, 1 = implicite)", 0.0, 1.0, 0.0, 0.05, key="vnx_theta")
            rs_txt = st.text_input("Valeurs de r à comparer", value="0.25, 0.5, 0.6", key="vnx_rs")
            try:
                rs = [float(v) for v in rs_txt.replace(";", ",").split(",") if v.strip()][:6]
            except ValueError:
                rs = [0.25, 0.5, 0.6]
                ui_caption(":red[Liste invalide]")
            lim = math.inf if vt >= 0.5 else 1 / (2 * (1 - 2 * vt))
            st.latex(r"G(\theta) = \frac{1 - 4(1-\vartheta) r\sin^2(\theta/2)}{1 + 4\vartheta r\sin^2(\theta/2)}")
            ui_markdown("Stabilité : " + ("**inconditionnelle**" if vt >= 0.5 else f"**r ≤ {fmt(lim)}**"))
        with c2:
            items = [(f"r = {r:g}", parabolic_symbol(th, r, vt=vt)[0]) for r in rs]
            show_fig(fig_amplification(th, items, mode, "Facteur d'amplification G(θ) (réel)", show_real=True),
                     "Bandes rouges : |G| > 1 (modes amplifiés). G réel : le signe indique si le mode oscille en temps.",
                     key="vnx_heat_G")
        c3, c4 = st.columns(2)
        with c3:
            rr = np.linspace(0.01, 2.0, 160)
            tt = np.linspace(0, 1, 120)
            RR, TT = np.meshgrid(rr, tt)
            # G est monotone en s = sin²(θ/2) ∈ [0, 1] : max_θ |G| = max(|G(0)|, |G(π)|) = max(1, |G(π)|)
            Z = np.maximum(1.0, np.abs(1 - 4 * (1 - TT) * RR) / (1 + 4 * TT * RR))
            cx = np.linspace(0.01, 2.0, 200)
            cy = 0.5 - 1 / (4 * cx)
            m_ = cy >= 0
            show_fig(fig_stability_map(rr, tt, Z, mode, "Diagramme de stabilité (pire mode θ = π)",
                                       "r", "ϑ", [("r = 1/(2(1−2ϑ))", cx[m_], cy[m_])], (rs[0] if rs else 0.5, vt)),
                     "Vert : max|G| ≤ 1 (stable) ; rouge : instable. La frontière est r(1 − 2ϑ) = 1/2.", key="vnx_heat_map")
        with c4:
            r_sim = st.select_slider("r pour la simulation de perturbation", options=sorted(set(rs + [0.5])),
                                     value=rs[-1] if rs else 0.6, key="vnx_rsim")
            pe = perturbation_experiment("parabolic", {"r": r_sim, "theta": vt}, n=64, nsteps=80)
            show_fig(fig_curves(pe["steps"], [{"y": pe["norms"], "name": "‖eⁿ‖/‖e⁰‖ (simulation)"},
                                              {"y": np.minimum(pe["pred"], 1e30), "name": "max|G|ⁿ (prédiction)", "slot": 1, "width": 1.5}],
                                mode, "Évolution d'une perturbation aléatoire", "pas n", "amplification", 330, logy=True),
                     f"Schéma réel appliqué en espace physique (grille périodique, 64 points). max|G| = {fmt(pe['gmax'], 6)}.",
                     key="vnx_heat_pert")
    elif fam == "Chaleur 2D":
        c1, c2 = st.columns([1, 2])
        with c1:
            vt = st.slider("ϑ", 0.0, 1.0, 0.0, 0.05, key="vnx2_theta")
            rx = st.slider("rₓ = κΔt/hₓ²", 0.0, 1.0, 0.2, 0.01, key="vnx2_rx")
            ry = st.slider("r_y = κΔt/h_y²", 0.0, 1.0, 0.2, 0.01, key="vnx2_ry")
            st.latex(r"G = \frac{1 - 4(1-\vartheta)(r_x s_x + r_y s_y)}{1 + 4\vartheta(r_xs_x + r_ys_y)},\ s = \sin^2\tfrac\theta2")
            ui_markdown("Explicite : **rₓ + r_y ≤ 1/2** (et non r ≤ 1/2 dans chaque direction).")
        with c2:
            tx = np.linspace(-np.pi, np.pi, 181)
            TX, TY = np.meshgrid(tx, tx)
            G, _ = parabolic_symbol(TX, rx, vt=vt, th_y=TY, ry=ry)
            fig = go.Figure(go.Heatmap(x=tx, y=tx, z=np.abs(G), colorscale=pal(mode)["seq"],
                                       colorbar=dict(title=dict(text="|G|"), outlinewidth=0, thickness=12)))
            fig.add_trace(go.Contour(x=tx, y=tx, z=np.abs(G), contours=dict(start=1, end=1, size=1, coloring="none"),
                                     line=dict(color=pal(mode)["bad"], width=2), showscale=False))
            style_fig(fig, mode, f"|G(θₓ, θ_y)| — max = {np.max(np.abs(G)):.4f}", 430, "θₓ", "θ_y", legend=False)
            show_fig(fig, "Contour rouge : |G| = 1. Le pire mode est (θₓ, θ_y) = (π, π).", key="vnx2_fig")
    elif fam == "Advection–diffusion 1D":
        c1, c2 = st.columns([1, 2])
        with c1:
            adv = st.radio("Advection", ["centered", "upwind"], format_func=lambda k: {"centered": "centrée",
                           "upwind": "décentrée amont"}[k], horizontal=True, key="vnx_ad_adv")
            r = st.slider("r = kΔt/h²", 0.0, 1.0, 0.3, 0.01, key="vnx_ad_r")
            lam = st.slider("λ = bΔt/h", -1.5, 1.5, 0.5, 0.01, key="vnx_ad_l")
            ui_markdown("Explicite centré : **r ≤ 1/2 et λ² ≤ 2r** ; décentré : **2r + |λ| ≤ 1**.")
        with c2:
            G, z = parabolic_symbol(th, r, lam, 0.0, 0.0, adv)
            show_fig(fig_amplification(th, [("|G(θ)|", G)], mode, f"|G(θ)| — max = {np.max(np.abs(G)):.4f}"),
                     key="vnx_ad_G")
        rr = np.linspace(0.0, 0.8, 161)
        ll = np.linspace(-1.5, 1.5, 161)
        RR, LL = np.meshgrid(rr, ll)
        tt = np.linspace(0, np.pi, 181)
        Z = np.zeros_like(RR)
        for t_ in tt:
            z_ = 4 * RR * np.sin(t_ / 2) ** 2 + (np.abs(LL) * (1 - np.cos(t_)) if adv == "upwind" else 0) + 1j * LL * np.sin(t_)
            Z = np.maximum(Z, np.abs(1 - z_))
        curves = ([("λ² = 2r", np.linspace(0, 0.8, 100), np.sqrt(2 * np.linspace(0, 0.8, 100))),
                   ("λ = −√(2r)", np.linspace(0, 0.8, 100), -np.sqrt(2 * np.linspace(0, 0.8, 100)))]
                  if adv == "centered" else [("2r + |λ| = 1", np.linspace(0, 0.5, 50), 1 - 2 * np.linspace(0, 0.5, 50)),
                                              ("2r − λ = 1", np.linspace(0, 0.5, 50), -(1 - 2 * np.linspace(0, 0.5, 50)))])
        show_fig(fig_stability_map(rr, ll, Z, mode, "Diagramme de stabilité dans le plan (r, λ)", "r", "λ", curves, (r, lam)),
                 "Calcul numérique de max_θ |G| sur une grille de paramètres ; les courbes sont les frontières analytiques.",
                 key="vnx_ad_map")
    elif fam == "Ondes 1D":
        c1, c2 = st.columns([1, 2])
        with c1:
            beta = st.radio("Schéma", [0.0, 0.25], format_func=lambda b_: "saute-mouton (β = 0)" if b_ == 0 else "Newmark (β = 1/4)",
                            horizontal=True, key="vnx_w_beta")
            sig_txt = st.text_input("Valeurs de σ = cΔt/h", value="0.5, 1.0, 1.1", key="vnx_w_sig")
            try:
                sigs = [float(v) for v in sig_txt.replace(";", ",").split(",") if v.strip()][:6]
            except ValueError:
                sigs = [0.5, 1.0, 1.1]
            st.latex(r"G^2 - 2\big(1 - 2\sigma^2\sin^2\tfrac\theta2\big)G + 1 = 0\ \ (\beta=0)")
            ui_markdown("Racines de module 1 si |1 − 2σ² sin²(θ/2)| ≤ 1, soit **σ ≤ 1** : schéma non dissipatif.")
        with c2:
            items = []
            for s_ in sigs:
                g1, g2 = wave_roots(4 * s_ ** 2 * np.sin(th / 2) ** 2, beta)
                items.append((f"σ = {s_:g}", np.maximum(np.abs(g1), np.abs(g2))))
            show_fig(fig_amplification(th, items, mode, "max |G±(θ)|"), key="vnx_w_G")
        pe = perturbation_experiment("wave", {"sigma": sigs[-1] if sigs else 1.1, "beta": beta}, n=64, nsteps=80)
        show_fig(fig_curves(pe["steps"], [{"y": pe["norms"], "name": "‖eⁿ‖/‖e⁰‖"}, {"y": np.minimum(pe["pred"], 1e30),
                                                                                     "name": "max|G|ⁿ", "slot": 1}],
                            mode, f"Perturbation pour σ = {sigs[-1] if sigs else 1.1:g}", "pas n", "amplification", 320,
                            logy=True), key="vnx_w_pert")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            schemes = st.multiselect("Schémas", list(TRANSPORT_SCHEMES), default=["upwind", "lax_friedrichs", "lax_wendroff", "ftcs"],
                                     format_func=lambda k: TRANSPORT_SCHEMES[k], key="vnx_t_sch")
            lam = st.slider("λ = aΔt/h", -1.5, 1.5, 0.8, 0.01, key="vnx_t_lam")
            ui_markdown("Décentré, Lax–Friedrichs, Lax–Wendroff : **|λ| ≤ 1** ; FTCS : instable ; centré implicite : "
                        "inconditionnellement stable.")
        with c2:
            items = [(TRANSPORT_SCHEMES[s], transport_symbol(s, lam, th)) for s in schemes]
            show_fig(fig_amplification(th, items, mode, "Dissipation : |G(θ)|"), key="vnx_t_G")
        c3, c4 = st.columns(2)
        with c3:
            thp = np.linspace(1e-3, np.pi, 600)
            curves = []
            for i, s in enumerate(schemes):
                G = transport_symbol(s, lam, thp)
                ratio = -np.angle(G) / (lam * thp) if lam != 0 else np.ones_like(thp)
                curves.append({"x": thp, "y": ratio, "name": TRANSPORT_SCHEMES[s], "slot": i})
            show_fig(fig_curves(thp, curves, mode, "Dispersion : vitesse de phase discrète / a", "θ", "c_num / a", 330),
                     "1 = propagation exacte. Un rapport < 1 retarde les hautes fréquences (oscillations derrière les fronts).",
                     key="vnx_t_phase")
        with c4:
            lams = np.linspace(0.01, 1.6, 200)
            curves = []
            for i, s in enumerate(schemes):
                curves.append({"x": lams, "y": [float(np.max(np.abs(transport_symbol(s, l_, th)))) for l_ in lams],
                               "name": TRANSPORT_SCHEMES[s], "slot": i})
            fig = fig_curves(lams, curves, mode, "Critère CFL : max_θ |G| en fonction de λ", "λ", "max |G|", 330)
            fig.add_hline(y=1, line=dict(color=pal(mode)["bad"], width=1))
            fig.add_vline(x=1, line=dict(color=pal(mode)["muted"], width=1))
            show_fig(fig, key="vnx_t_cfl")


def page_stability() -> None:
    page_header("Module 8", "Stabilité numérique : von Neumann, CFL, analyse spectrale",
                "Une véritable analyse mathématique : facteurs d'amplification, critères démontrés, diagrammes de "
                "stabilité, vérification par simulation, et analyse spectrale des matrices réelles (K z = λ M z).")
    prob = current_problem()
    mode = theme_mode()
    tabs = st.tabs(["Théorie", "Explorateur interactif", "Problème courant", "Analyse spectrale", "Limites du critère"])
    with tabs[0]:
        _vn_theory()
    with tabs[1]:
        _vn_explorer()
    with tabs[2]:
        if not prob.ok:
            ui_error("Problème invalide.", icon=":material/error:")
        elif not prob.time_dependent:
            ui_info("Problème stationnaire : pas de stabilité temporelle à étudier. La stabilité du schéma spatial est "
                    "la borne uniforme de ‖A_h⁻¹‖ (coercivité ou principe du maximum discret).", icon=":material/info:")
        else:
            res = get_result("solve")
            plan = res if (res and res.get("ok")) else time_plan_current(prob)
            if not plan or "error" in plan:
                ui_error((plan or {}).get("error", "Paramètres temporels invalides."), icon=":material/error:")
            else:
                stab = stability_report(prob, plan)
                store_result("stability", stab)
                a, b = st.columns([1.25, 1])
                with a:
                    with card("stab_crit"):
                        card_title("rule", "Critères évalués avec les paramètres réels")
                        for c in stab["criteria"]:
                            c1, c2 = st.columns([3, 1.1])
                            c1.markdown(f"**{c['name']}** — {c['value']}")
                            c2.markdown(badge("satisfaite" if c["ok"] else "non satisfaite", "green" if c["ok"] else "red"),
                                        unsafe_allow_html=True)
                            st.latex(c["latex"])
                            ui_caption(c["kind"])
                        if "max_G" in stab:
                            ui_markdown(f"Évaluation numérique : **max_θ |G(θ)| = {stab['max_G']:.6f}**")
                        for n_ in stab["notes"]:
                            ui_caption(n_)
                with b:
                    with card("stab_valid"):
                        card_title("checklist", "Cadre de validité de von Neumann")
                        for lab, ok in stab["vn_valid"]:
                            ui_markdown(("✓ " if ok else "✗ ") + lab)
                        if not all(ok for _, ok in stab["vn_valid"]):
                            ui_caption("Hors du cadre strict (bords non périodiques, coefficients variables, maillage non "
                                       "uniforme) : le critère devient une condition nécessaire (coefficients gelés). "
                                       "L'analyse spectrale des matrices réelles (onglet suivant) donne le verdict exact "
                                       "pour le problème semi-discret.")
                th = np.linspace(-np.pi, np.pi, 1201)
                with card("stab_G"):
                    card_title("show_chart", "Facteur d'amplification avec les paramètres actuels")
                    if prob.cls == "parabolic":
                        if prob.method == "fdm":
                            G, _ = parabolic_symbol(th, stab["r"], stab["lam"], 0, stab["theta"], prob.cfg.get("adv_scheme", "centered"))
                            items = [("DF : |G(θ)|", G)]
                        else:
                            items = [("P1 masse cohérente", fem1d_symbol(th, stab["r"], stab["theta"], False)[0]),
                                     ("P1 masse condensée", fem1d_symbol(th, stab["r"], stab["theta"], True)[0])]
                        show_fig(fig_amplification(th, items, mode, f"|G(θ)| — r = {fmt(stab['r'])}"), key="stab_cur_G")
                    elif prob.cls == "hyperbolic":
                        g1, g2 = wave_roots(4 * stab["sigma"] ** 2 * np.sin(th / 2) ** 2, stab["beta"])
                        show_fig(fig_amplification(th, [("max|G±|", np.maximum(np.abs(g1), np.abs(g2)))], mode,
                                                   f"σ = {fmt(stab['sigma'])}"), key="stab_cur_G")
                    else:
                        sch = prob.cfg["time"]["transport_scheme"]
                        show_fig(fig_amplification(th, [(TRANSPORT_SCHEMES[sch], transport_symbol(sch, stab["lam"], th))],
                                                   mode, f"λ = {fmt(stab['lam'])}"), key="stab_cur_G")
    with tabs[3]:
        if not prob.ok or prob.cls == "transport":
            ui_info("Analyse spectrale disponible pour les problèmes assemblés (elliptiques, paraboliques, ondes). Pour le "
                    "transport sur grille périodique, les modes de Fourier sont exactement les vecteurs propres : "
                    "l'analyse de von Neumann est l'analyse spectrale.", icon=":material/info:")
        else:
            res = get_result("solve")
            plan = res if (res and res.get("ok")) else time_plan_current(prob)
            dt = float(plan["dt"]) if plan and "dt" in plan else None
            ui_markdown("On calcule les valeurs propres généralisées du problème semi-discret, restreint aux nœuds libres :")
            st.latex(r"K z = \mu\,M z\qquad\Longrightarrow\qquad U^{n+1} = R(\Delta t\,\mu)\,U^n\ \text{mode par mode}")
            ui_caption("Contrairement à von Neumann (symbole sur grille infinie), cette analyse inclut les vraies conditions "
                       "aux limites, les coefficients variables et le maillage réel. Pour K et M symétriques (M définie "
                       "positive), la stabilité ⇔ |R(Δt μᵢ)| ≤ 1 pour tout i.")
            asm = cached_assembly(cfg_json_cur())
            nfree = len(asm["free"]) if asm.get("ok") else 0
            auto = nfree <= 1500
            if auto or st.button("Calculer le spectre (ARPACK pour les grandes tailles)", key="spec_btn"):
                with st.spinner("Calcul des valeurs propres…"):
                    sa = cached_spectral(cfg_json_cur(), dt)
                store_result("spectral", sa, key=f"{dt}")
                if "error" in sa:
                    ui_error(sa["error"], icon=":material/error:")
                else:
                    c = st.columns(4)
                    c[0].metric("μ_min", fmt(sa["lam_min"]), border=True)
                    c[1].metric("μ_max", fmt(sa["lam_max"]), border=True)
                    if sa.get("rho") is not None:
                        c[2].metric("max |R(Δt μᵢ)|", f"{sa['rho']:.6f}", border=True,
                                    delta="stable" if sa["rho"] <= 1 + 1e-10 else "instable",
                                    delta_color="normal" if sa["rho"] <= 1 + 1e-10 else "inverse")
                    if sa.get("dt_max"):
                        c[3].metric("Δt maximal stable", fmt(sa["dt_max"]), border=True,
                                    delta=f"Δt actuel = {fmt(dt)}", delta_color="off")
                    elif sa.get("cond_K"):
                        c[3].metric("κ₂(K)", fmt(sa["cond_K"]), border=True)
                    a, b = st.columns(2)
                    with a:
                        show_fig(fig_spectrum(sa["eigs"], sa.get("exact_spectrum"), mode,
                                              "Spectre discret" + (" (complet)" if sa.get("complete") else " (extrémités, ARPACK)")),
                                 "Les petites valeurs propres approchent celles de l'opérateur continu ; les grandes "
                                 "(∝ h⁻²) sont des artefacts de la discrétisation qui pilotent la stabilité des schémas explicites.",
                                 key="spec_fig")
                    with b:
                        if prob.time_dependent and dt:
                            if prob.time_order == 1:
                                z = dt * sa["eigs"]
                                vn = None
                                if prob.method == "fdm" and prob.dim == 1:
                                    _, vn = parabolic_symbol(np.linspace(-np.pi, np.pi, 400), plan["numbers"].get("r", 0),
                                                             prob.bx * dt / plan["hmin"], 0, 0,
                                                             prob.cfg.get("adv_scheme", "centered"))
                                show_fig(fig_stability_region(float(prob.cfg["time"]["theta"]), None, z, mode,
                                                              "Région de stabilité du θ-schéma et spectre Δt·μᵢ", vn),
                                         "Zone verte : |R(z)| ≤ 1. Stabilité ⇔ tous les points bleus dans la zone verte.",
                                         key="spec_region")
                            else:
                                beta = 0.0 if prob.cfg["time"]["wave_scheme"] == "leapfrog" else 0.25
                                w = dt * dt * sa["eigs"]
                                show_fig(fig_stability_region(None, beta, w.astype(complex), mode,
                                                              "Saute-mouton : stabilité ⇔ Δt²μᵢ ≤ 4"),
                                         "Pour β = 1/4 (Newmark), aucune borne : stabilité inconditionnelle.", key="spec_region_w")
                        else:
                            ui_caption(f"Problème stationnaire : κ₂(K) = {fmt(sa.get('cond_K'))}, plus petite valeur "
                                       f"propre de K = {fmt(sa.get('K_min_eig'))} (> 0 ⇔ K définie positive).")
                    if sa.get("lam_max_consistent") is not None:
                        lc, ll = sa["lam_max_consistent"], sa["lam_max_lumped"]
                        ui_markdown("**Masse cohérente ou condensée (EF)** — limite de stabilité d'Euler explicite Δt ≤ 2/μ_max :")
                        kv_table([("μ_max (M cohérente)", fmt(lc)), ("Δt_max (cohérente)", fmt(2 / lc)),
                                  ("μ_max (M condensée)", fmt(ll)), ("Δt_max (condensée)", fmt(2 / ll)),
                                  ("Rapport", fmt(lc / ll))])
                        ui_caption("En 1D uniforme, μ_max ≈ 12k/h² (cohérente) contre 4k/h² (condensée) : la condensation "
                                   "assouplit la condition d'un facteur 3 et rend le schéma explicite sans résolution.")
            else:
                ui_caption(f"{nfree} inconnues : calcul à la demande.")
    with tabs[4]:
        with card("vn_limits"):
            card_title("report", "Ce que prouve — et ne prouve pas — l'analyse de von Neumann")
            ui_markdown(
                "- **Cadre** : schéma linéaire, coefficients constants, grille uniforme, conditions périodiques "
                "(ou frontières qui ne détruisent pas la structure de Fourier). Dans ce cadre, |G| ≤ 1 + CΔt est "
                "nécessaire et suffisant pour la stabilité L².\n"
                "- **Frontières** : avec Dirichlet/Neumann, le critère reste nécessaire ; pour les matrices symétriques "
                "(ou symétrisables), le rayon spectral de la matrice d'itération donne le verdict exact (exemple : en "
                "1D avec Dirichlet, la limite exacte est r ≤ 1/(2cos²(πh/2)), qui tend vers 1/2).\n"
                "- **Coefficients variables** : on « gèle » les coefficients (k_max) — critère local, heuristique.\n"
                "- **Matrices non normales** (advection) : ρ(T) ≤ 1 ne suffit pas à exclure une croissance transitoire ; "
                "on recourt à une méthode d'énergie (estimation ‖Uⁿ‖ ≤ C‖U⁰‖) ou aux pseudo-spectres.\n"
                "- **Non linéaire** : von Neumann ne s'applique qu'au problème linéarisé.\n"
                "- **Éléments finis** : l'analyse naturelle est spectrale (K z = μ M z) et énergétique, pas de Fourier.")
            st.latex(r"\text{Méthode d'énergie (Euler implicite, } K\ \text{SDP) : } (U^{n+1}-U^n, U^{n+1})_M + \Delta t\,"
                     r"(KU^{n+1},U^{n+1}) = 0\ \Rightarrow\ \|U^{n+1}\|_M\le\|U^n\|_M")
            st.latex(r"\text{Consistance}\ +\ \text{Stabilité}\ \Longleftrightarrow\ \text{Convergence}\qquad"
                     r"\text{(Lax–Richtmyer, problème linéaire bien posé)}")
    next_step("convergence", "convergence et erreurs")


# =============================================================================
# 12f. PAGES — CONVERGENCE, VISUALISATION, COMPARAISON, SYNTHÈSE, EXPORT
# =============================================================================
def _conv_dataframe(conv: dict) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(conv["rows"]):
        row = {"niveau": r["level"], "N": r["n"], "h": r["h"], "inconnues": r["dofs"]}
        if r.get("dt"):
            row["Δt"] = r["dt"]
            row["N_t"] = r["nt"]
        for k in conv["keys"]:
            row[f"erreur {k.upper()}"] = r[k]
            row[f"ordre {k.upper()}"] = conv["orders"][k][i]
        row["temps (s)"] = r["time"]
        rows.append(row)
    return pd.DataFrame(rows)


def page_convergence() -> None:
    page_header("Module 9", "Consistance, erreurs et convergence",
                "Trois notions distinctes — consistance, stabilité, convergence — reliées par le théorème de "
                "Lax–Richtmyer ; mesure des erreurs et ordres expérimentaux par raffinement successif.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    tabs = st.tabs(["Théorie", "Erreurs du calcul courant", "Convergence en espace", "Convergence en temps"])
    with tabs[0]:
        a, b = st.columns(2)
        with a:
            with card("cv_three"):
                card_title("school", "Consistance, stabilité, convergence")
                st.latex(r"\begin{aligned}&\text{Consistance : } \tau_h = L_h(\Pi_h u) - f_h\ \xrightarrow[h,\Delta t\to 0]{}\ 0\\"
                         r"&\quad(\text{erreur de troncature } O(h^p + \Delta t^q))\end{aligned}")
                st.latex(r"\begin{aligned}&\text{Stabilité : } \|A_h^{-1}\|\le C\ \text{(stationnaire)},\\ "
                         r"&\phantom{\text{Stabilité : }}\|T^n\|\le C\ \ (0\le n\Delta t\le T)\end{aligned}")
                st.latex(r"\text{Convergence : } \|u - u_h\|\xrightarrow[h\to 0]{} 0")
                st.latex(r"\boxed{\ \text{Consistance} + \text{Stabilité}\ \Longleftrightarrow\ \text{Convergence}\ }")
                ui_caption("Théorème d'équivalence de Lax–Richtmyer.")
                ui_caption("Le théorème d'équivalence de Lax–Richtmyer concerne les schémas d'évolution linéaires pour un "
                           "problème bien posé ; il ne doit pas être confondu avec le théorème de Lax–Milgram (existence et "
                           "unicité pour a(u, v) = L(v)). En stationnaire : ‖e‖ ≤ ‖A_h⁻¹‖ ‖τ_h‖.")
        with b:
            with card("cv_estimates"):
                card_title("functions", "Estimations d'erreur")
                st.latex(r"\begin{aligned}&\text{DF : } \|u - U\|_\infty\le\|A_h^{-1}\|_\infty\,\|\tau_h\|_\infty\le C\,h^2\|u\|_{C^4}\\"
                         r"&\quad(\text{principe du maximum discret})\end{aligned}")
                st.latex(r"\begin{aligned}&\text{EF (Céa) : } \|u-u_h\|_{H^1}\le\frac M\alpha\inf_{v_h}\|u-v_h\|_{H^1}\\"
                         r"&\phantom{\text{EF (Céa) : } \|u-u_h\|_{H^1}}\le\frac M\alpha\,C\,h\,|u|_{H^2}\end{aligned}")
                st.latex(r"\begin{aligned}&\text{EF (Aubin–Nitsche, }\Omega\text{ convexe) : }\\ &\|u-u_h\|_{L^2}\le C\,h^2|u|_{H^2}\end{aligned}")
                st.latex(r"\text{Évolution : } \max_n\|u(t_n)-U^n\|\le C\,(h^2 + \Delta t^{q})")
                th = theoretical_orders(prob)
                ui_markdown("**Pour la configuration courante :**")
                ui_markdown(th["text"])
                ui_warning("Conditions : " + th["conditions"], icon=":material/rule:")
    with tabs[1]:
        res = get_result("solve")
        if not res or not res.get("ok"):
            ui_info("Résolvez d'abord le problème (tableau de bord ou page Résolution).", icon=":material/info:")
            if st.button("Résoudre maintenant", icon=":material/play_arrow:", key="cv_solve"):
                solve_current()
                st.rerun()
        elif not res.get("errors"):
            ui_info("Solution exacte non fournie : les erreurs ne peuvent pas être calculées. Activez une solution "
                    "exacte ou manufacturée à la page Définition.", icon=":material/info:")
        else:
            errs = res["errors"]
            labels = {"l2": "‖e‖_L² " + ("(continue, quadrature)" if prob.method == "fem" and prob.cls != "transport" else "(discrète)"),
                      "l2_rel": "erreur L² relative", "h1": "|e|_H¹ " + ("(continue)" if prob.method == "fem" else "(discrète)"),
                      "h1_rel": "erreur H¹ relative", "max": "‖e‖_∞", "max_nodal": "max nodal", "max_quad": "max aux points de quadrature",
                      "l2_discrete": "L² discrète (poids nodaux)", "l1": "‖e‖_L¹", "l1_rel": "L¹ relative", "max_rel": "max relative",
                      "max_nodal_rel": "max nodale relative"}
            a, b = st.columns([1, 1.5])
            with a:
                with card("cv_errtab"):
                    card_title("table", "Normes de l'erreur e = u − u_h" + (f" à t = {fmt(res.get('t_final'))}" if prob.time_dependent else ""))
                    kv_table([(labels.get(k, k), fmt(v, 6)) for k, v in errs.items() if not k.startswith("norm_")])
                    st.download_button("Télécharger (CSV)", data=lambda: pd.DataFrame([errs]).to_csv(index=False),
                                       file_name="erreurs.csv", mime="text/csv", icon=":material/download:", on_click="ignore")
            with b:
                with card("cv_errfig"):
                    card_title("blur_linear", "Distribution de l'erreur")
                    e = np.asarray(res["U"]) - np.asarray(res["Uex"])
                    if prob.dim == 1:
                        x = res["nodes"][:, 0]
                        o = np.argsort(x)
                        show_fig(fig_curves(x[o], [{"y": e[o], "name": "u_h − u"}], mode, None, "x", "erreur", 360),
                                 key="cv_err1d")
                    else:
                        show_fig(fig_field_2d(res, e, mode, "heatmap", None, "u_h − u", "div", height=420),
                                 "Échelle divergente centrée en 0 : bleu = sous-estimation, rouge = surestimation.",
                                 key="cv_err2d")
    with tabs[2]:
        if prob.exact is None and prob.cls != "transport":
            ui_info("Une solution exacte (ou manufacturée) est nécessaire pour mesurer l'erreur.", icon=":material/info:")
        else:
            with card("cv_ctrl"):
                card_title("tune", "Paramètres de l'étude")
                c1, c2, c3, c4 = st.columns(4)
                levels = c1.slider("Nombre de niveaux", 2, 6, 4 if prob.dim == 1 else 3, key="cv_levels")
                default_n0 = max(4 if prob.dim == 2 else 8, prob.cfg["mesh"]["nx"] // 2 ** (levels - 1))
                n0 = int(c2.number_input("Maillage initial N₀", 2, 512, int(default_n0), step=2, key="cv_n0"))
                strat = None
                if prob.time_dependent:
                    opts = ["number", "dt_h", "dt_h2", "fixed_dt"]
                    dflt = default_dt_strategy(prob)
                    strat = c3.selectbox("Couplage Δt / h", opts, index=opts.index(dflt), format_func=DT_STRATEGY_LABELS.get,
                                         key="cv_strat")
                c4.markdown("&nbsp;")
                run = c4.button("Lancer l'étude", type="primary", icon=":material/play_arrow:", key="cv_run")
                ui_caption("Chaque niveau divise h par 2 (éléments finis : raffinement uniforme emboîté). Les niveaux dont le "
                           "coût estimé dépasse la limite de sécurité ne sont pas calculés.")
            key = f"{levels}|{n0}|{strat}"
            if run:
                with st.spinner("Étude de convergence en cours…"):
                    conv = cached_convergence(cfg_json_cur(), levels, n0, strat)
                store_result("convergence", conv, key=key)
            conv = get_result("convergence")
            if conv is not None and not conv.get("ok"):
                ui_error("; ".join(conv.get("errors", [])), icon=":material/error:")
            elif conv:
                if conv.get("skipped"):
                    ui_warning(conv["skipped"], icon=":material/warning:")
                if any(r_.get("blowup") for r_ in conv["rows"]):
                    ui_error("Des calculs ont explosé pendant l'étude : le schéma est instable pour le nombre de stabilité "
                             "maintenu d'un niveau à l'autre. Sans stabilité il n'y a pas de convergence (Lax–Richtmyer) : "
                             "les pentes affichées n'ont pas de sens. Choisissez un pas de temps stable.",
                             icon=":material/warning:")
                a, b = st.columns([1.2, 1])
                with a:
                    show_fig(fig_loglog_convergence(conv["rows"], conv["keys"], conv["slopes"], conv["theory"], mode,
                                                    "log(erreur) en fonction de log(h)"),
                             "Pentes observées (moindres carrés sur les 3 derniers niveaux) ; lignes pointillées : pentes "
                             "théoriques.", key="cv_fig")
                with b:
                    with card("cv_orders"):
                        card_title("speed", "Ordres expérimentaux")
                        th = conv["theory"]
                        for k in conv["keys"]:
                            p_obs = conv["slopes"].get(k)
                            p_th = th["orders"].get(k)
                            ok = p_obs is not None and p_th is not None and abs(p_obs - p_th) < 0.25
                            ui_markdown(f"**{k.upper()}** : pente observée {p_obs:.3f} — théorique {fmt(p_th, 3) if p_th else '—'} "
                                        + badge("conforme" if ok else "à analyser", "green" if ok else "orange"),
                                        unsafe_allow_html=True)
                        ui_caption(th["text"])
                        ui_caption("Conditions : " + th["conditions"])
                df = _conv_dataframe(conv)
                fmts = {c: "{:.3e}" for c in df.columns if c.startswith("erreur") or c in ("h", "Δt")}
                fmts.update({c: "{:.3f}" for c in df.columns if c.startswith("ordre")})
                fmts["temps (s)"] = "{:.3f}"
                st.dataframe(df_numbers(df.to_dict("records"), fmts), hide_index=True, width="stretch")
                st.download_button("Télécharger les résultats de convergence (CSV)", data=lambda: df.to_csv(index=False),
                                   file_name="convergence.csv", mime="text/csv", icon=":material/download:", on_click="ignore")
    with tabs[3]:
        if not prob.time_dependent or prob.cls == "transport":
            ui_info("Étude temporelle à maillage fixe disponible pour les problèmes paraboliques et les ondes.",
                    icon=":material/info:")
        else:
            ui_markdown("À maillage fixe, Δt est divisé par 2 à chaque niveau. L'erreur est mesurée par rapport à une "
                        "solution de référence calculée avec la **même discrétisation spatiale** et un pas 8 fois plus "
                        "petit (Crank–Nicolson ou Newmark, ordre 2) : l'erreur spatiale, commune, s'élimine.")
            lv = st.slider("Niveaux", 3, 6, 5, key="tc_levels")
            if st.button("Lancer l'étude temporelle", icon=":material/play_arrow:", key="tc_run"):
                with st.spinner("Calcul…"):
                    tc = cached_temporal(cfg_json_cur(), lv)
                store_result("tconv", tc, key=str(lv))
            tc = get_result("tconv")
            if tc and not tc.get("ok"):
                ui_error("; ".join(tc["errors"]), icon=":material/error:")
            elif tc:
                dts = np.array([r["dt"] for r in tc["rows"]])
                es = np.array([r["err"] for r in tc["rows"]])
                p_exp = 2 if (prob.time_order == 2 or abs(prob.cfg["time"]["theta"] - 0.5) < 1e-12) else 1
                fig = fig_curves(dts, [{"y": es, "name": "erreur temporelle", "mode": "lines+markers"},
                                       {"y": es[-1] * (dts / dts[-1]) ** p_exp, "name": f"pente {p_exp}", "slot": 2, "width": 1}],
                                 mode, "Erreur de discrétisation en temps", "Δt", "‖U_Δt(T) − U_ref(T)‖", 400, True, True)
                show_fig(fig, f"Référence : {tc['reference']} ; {tc['dofs']} inconnues.", key="tc_fig")
                st.dataframe(df_numbers([{"Δt": r["dt"], "erreur": r["err"], "ordre": o, "temps (s)": r["time"]}
                                         for r, o in zip(tc["rows"], tc["orders"])],
                                        {"Δt": "{:.3e}", "erreur": "{:.3e}", "ordre": "{:.3f}", "temps (s)": "{:.3f}"}),
                             hide_index=True, width="stretch")
    next_step("visualization", "visualisation")


def page_visualization() -> None:
    page_header("Module 10", "Visualisation scientifique",
                "Solution, solution exacte et erreur ; vues 3D, cartes, courbes de niveau, coupes et évolution temporelle. "
                "Tous les graphiques sont zoomables ; l'icône appareil photo exporte en PNG.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    res = get_result("solve")
    if not res or not res.get("ok"):
        ui_info("Aucune solution pour la configuration actuelle.", icon=":material/info:")
        if st.button("Résoudre maintenant", type="primary", icon=":material/play_arrow:", key="vis_solve"):
            solve_current()
            st.rerun()
        return
    has_ex = res.get("Uex") is not None
    td = prob.time_dependent and res.get("frames") is not None
    with card("vis_ctrl"):
        c1, c2, c3, c4 = st.columns(4)
        fields = ["u_h"] + (["u exacte", "erreur u_h − u", "|u_h − u|"] if has_ex else [])
        field = c1.selectbox("Champ", fields, key="vis_field")
        if prob.dim == 2:
            view = c2.selectbox("Vue", ["Surface 3D", "Carte (heatmap)", "Courbes de niveau", "Coupe 1D"], key="vis_view")
            cmap = c3.selectbox("Échelle de couleurs", list(COLORSCALE_LABELS), format_func=COLORSCALE_LABELS.get, key="vis_cmap")
            show_mesh = c4.toggle("Superposer le maillage", value=False, key="vis_mesh")
        else:
            view = c2.selectbox("Vue", ["Profil", "Animation", "Espace–temps"] if td else ["Profil"], key="vis_view1")
            cmap, show_mesh = "auto", False
        k = None
        if td:
            times = res["times"]
            k = st.select_slider("Instant", options=list(range(len(times))), value=len(times) - 1,
                                 format_func=lambda i: f"t = {times[i]:.4g}", key="vis_t")
    if td:
        Uh = np.asarray(res["frames"][k], dtype=float)
        Ue = np.asarray(res["exact_frames"][k], dtype=float) if res.get("exact_frames") is not None else None
        tlab = f" — t = {res['times'][k]:.4g}"
    else:
        Uh = np.asarray(res["U"])
        Ue = np.asarray(res["Uex"]) if has_ex else None
        tlab = ""
    vals = {"u_h": Uh, "u exacte": Ue, "erreur u_h − u": (Uh - Ue) if Ue is not None else None,
            "|u_h − u|": np.abs(Uh - Ue) if Ue is not None else None}[field]
    zl = {"u_h": "u_h", "u exacte": "u", "erreur u_h − u": "e", "|u_h − u|": "|e|"}[field]
    if prob.dim == 1:
        x = res["nodes"][:, 0]
        if view == "Animation":
            show_fig(fig_time_animation(x, res["times"], res["frames"], res.get("exact_frames"), mode, "Évolution de u_h"),
                     "Bouton Lecture ou curseur : propagation/diffusion au cours du temps (trait gris : solution exacte).",
                     key="vis_anim")
        elif view == "Espace–temps":
            show_fig(fig_spacetime(x, res["times"], res["frames"], mode, "Diagramme espace–temps de u_h"),
                     "Pour le transport, les caractéristiques x − a t = cste apparaissent comme des droites.", key="vis_xt")
        else:
            if field == "u_h":
                fig = fig_solution_1d(x, Uh, Ue, mode, "Solution numérique et exacte" + tlab)
            else:
                o = np.argsort(x)
                fig = fig_curves(x[o], [{"y": vals[o], "name": field}], mode, field + tlab, "x", zl)
            show_fig(fig, key="vis_1d")
    else:
        if view == "Coupe 1D":
            c1, c2 = st.columns([1, 3])
            with c1:
                axis = st.radio("Coupe", ["y = constante", "x = constante"], key="vis_cut_axis")
                lo, hi = (prob.ya, prob.yb) if axis.startswith("y") else (prob.xa, prob.xb)
                cval = st.slider("Valeur", float(lo), float(hi), float(0.5 * (lo + hi)), key="vis_cut_val")
            s = np.linspace(prob.xa if axis.startswith("y") else prob.ya, prob.xb if axis.startswith("y") else prob.yb, 400)
            pts = np.stack([s, np.full_like(s, cval)], 1) if axis.startswith("y") else np.stack([np.full_like(s, cval), s], 1)
            if res["method"] == "fem":
                interp = P1Interpolator(res["nodes"], res["elems"])
                lines = [{"x": s, "y": interp(Uh, pts), "name": "u_h (interpolation P1 exacte)"}]
                if Ue is not None:
                    lines.insert(0, {"x": s, "y": interp(Ue, pts), "name": "Π_h u (interpolé de la solution exacte)",
                                     "color": pal(mode)["muted"], "width": 2.5})
            else:
                xs, ys = res["grid"]["xs"], res["grid"]["ys"]
                if axis.startswith("y"):
                    j = int(np.argmin(np.abs(ys - cval)))
                    s, sel = xs, slice(j * len(xs), (j + 1) * len(xs))
                    lines = [{"x": s, "y": Uh[sel], "name": f"u_h (y = {ys[j]:.3f})", "mode": "lines+markers"}]
                    if Ue is not None:
                        lines.insert(0, {"x": s, "y": Ue[sel], "name": "u exacte", "color": pal(mode)["muted"], "width": 2.5})
                else:
                    i = int(np.argmin(np.abs(xs - cval)))
                    s = ys
                    lines = [{"x": s, "y": Uh[i::len(xs)], "name": f"u_h (x = {xs[i]:.3f})", "mode": "lines+markers"}]
                    if Ue is not None:
                        lines.insert(0, {"x": s, "y": Ue[i::len(xs)], "name": "u exacte", "color": pal(mode)["muted"], "width": 2.5})
            with c2:
                show_fig(fig_curves(s, lines, mode, "Coupe" + tlab, "abscisse le long de la coupe", "u", 420), key="vis_cut")
        else:
            kind = {"Surface 3D": "surface", "Carte (heatmap)": "heatmap", "Courbes de niveau": "contour"}[view]
            cm = "div" if field == "erreur u_h − u" and cmap == "auto" else cmap
            fig = fig_field_2d(res, vals, mode, kind, f"{field}{tlab}", zl, cm, show_mesh, height=560)
            show_fig(fig, "Éléments finis : surface exactement affine par triangle (Mesh3d) ; cartes et isolignes "
                          "obtenues par interpolation P1 exacte sur une grille fine." if res["method"] == "fem" else
                     "Différences finies : valeurs nodales sur la grille.", key="vis_2d")
            st.download_button("Exporter cette figure (HTML interactif)", data=lambda: fig.to_html(include_plotlyjs="cdn"),
                               file_name="figure.html", mime="text/html", icon=":material/download:", on_click="ignore")
    if td and prob.dim == 2:
        ui_caption("Évolution temporelle : déplacez le curseur « Instant » ; les images sont celles stockées pendant le calcul.")
    next_step("comparison", "comparaison des méthodes")


def _compare_cfg(base: dict, **upd: Any) -> dict:
    return normalize_config(deep_merge(base, upd))[0]


def page_comparison() -> None:
    page_header("Module 10 bis", "Comparaison des méthodes",
                "Mêmes données, méthodes différentes : degrés de liberté, temps, structure des matrices, erreurs, "
                "convergence et conditions de stabilité. Les méthodes ne sont pas présentées comme équivalentes : "
                "leurs erreurs ne se mesurent pas dans les mêmes normes.")
    prob = need_valid_problem()
    if prob is None:
        return
    mode = theme_mode()
    base = cfg_norm()
    labels = []
    if "fdm" in prob.meta["methods"] and "fem" in prob.meta["methods"] and prob.shape != "lshape":
        labels.append("DF vs EF")
    if prob.cls == "parabolic":
        labels.append("Schémas en temps")
    if prob.cls == "hyperbolic":
        labels.append("Schémas pour les ondes")
    if prob.cls == "transport":
        labels.append("Schémas de transport")
    if not labels:
        ui_info("Aucune comparaison pertinente pour cette configuration (domaine en L : éléments finis uniquement).",
                icon=":material/info:")
        return
    tabs = st.tabs(labels)
    for name, tab in zip(labels, tabs):
        with tab:
            if name == "DF vs EF":
                c1, c2 = st.columns([1, 3])
                with c1:
                    n = int(st.number_input("Mailles par direction N", 4, 128 if prob.dim == 2 else 2048,
                                            int(base["mesh"]["nx"]), step=4, key="cmp_n"))
                    go_ = st.button("Comparer", type="primary", icon=":material/compare_arrows:", key="cmp_go")
                    conv_go = st.button("Comparer aussi la convergence", icon=":material/trending_down:", key="cmp_conv",
                                        disabled=prob.exact is None)
                with c2:
                    ui_caption("Grille DF (N+1)ᵈ nœuds ; EF P1 sur triangulation structurée de mêmes nœuds (motif choisi). "
                               "Erreurs : max nodale (commune aux deux méthodes) et L² discrète ; pour les EF, la norme L² "
                               "continue est aussi donnée.")
                if go_ or get_result("cmp_dffe", key=str(n)):
                    out = {}
                    for meth in ("fdm", "fem"):
                        c = _compare_cfg(base, method=meth, mesh={"nx": n, "ny": n, "refine": 0, "grading": "uniform"})
                        out[meth] = cached_solve(config_json(c))
                    store_result("cmp_dffe", True, key=str(n))
                    if not all(r["ok"] for r in out.values()):
                        ui_error("; ".join(sum((r.get("errors", []) for r in out.values()), [])), icon=":material/error:")
                    else:
                        rows = []
                        for meth, r in out.items():
                            e = r.get("errors") or {}
                            rows.append({"Méthode": METHOD_LABELS[meth], "Inconnues": r["dofs"],
                                         "Non-zéros K_FF": r["matrix"]["nnz"],
                                         "Assemblage (ms)": 1e3 * r["timings"].get("assembly", 0),
                                         "Résolution (ms)": 1e3 * (r["timings"].get("solve", 0) + r["timings"].get("time_loop", 0)),
                                         "Erreur max nodale": e.get("max_nodal"), "Erreur L² discrète": e.get("l2_discrete"),
                                         "Erreur L² continue": e.get("l2") if meth == "fem" else None,
                                         "Symétrique": (r.get("matrix") or {}).get("symmetric")})
                        st.dataframe(df_numbers(rows, {"Assemblage (ms)": "{:.1f}", "Résolution (ms)": "{:.1f}",
                                                        "Erreur max nodale": "{:.3e}", "Erreur L² discrète": "{:.3e}",
                                                        "Erreur L² continue": "{:.3e}"}), hide_index=True, width="stretch")
                        a, b = st.columns(2)
                        for col, meth in zip((a, b), ("fdm", "fem")):
                            with col:
                                c = _compare_cfg(base, method=meth, mesh={"nx": n, "ny": n, "refine": 0, "grading": "uniform"})
                                asm = cached_assembly(config_json(c))
                                if asm.get("ok"):
                                    Fr = asm["free"]
                                    show_fig(fig_spy(asm["K"][Fr][:, Fr], mode, f"{'DF' if meth == 'fdm' else 'EF'} : K_FF", 360),
                                             key=f"cmp_spy_{meth}")
                        if prob.dim == 1:
                            x1, x2 = out["fdm"]["nodes"][:, 0], out["fem"]["nodes"][:, 0]
                            o1, o2 = np.argsort(x1), np.argsort(x2)
                            curves = [{"x": x1[o1], "y": out["fdm"]["U"][o1], "name": "DF"},
                                      {"x": x2[o2], "y": out["fem"]["U"][o2], "name": "EF P1", "slot": 1}]
                            show_fig(fig_curves(x1, curves, mode, "Solutions DF et EF", "x", "u", 360), key="cmp_sol1d")
                        ui_markdown("**Différences de fond** : la méthode DF approche l'opérateur (ponctuellement, grille "
                                    "uniforme) ; la méthode EF approche l'espace (Galerkin, maillage quelconque) — d'où une "
                                    "matrice de masse non diagonale, des estimations en normes d'énergie et la conformité "
                                    "V_h ⊂ H¹. Sur grille cartésienne et pour −Δ à k constant, les rigidités coïncident à un "
                                    "facteur h² près (motif « diagonale montante ») : les écarts proviennent alors du second "
                                    "membre (ponctuel vs intégré) et, en évolution, de la matrice de masse.")
                        if prob.time_dependent and max(r["dofs"] for r in out.values()) <= 1500:
                            ui_markdown("**Stabilité des schémas explicites (limite Δt_max = 2/μ_max)** :")
                            srows = []
                            for meth, lump in (("fdm", False), ("fem", False), ("fem", True)):
                                c = _compare_cfg(base, method=meth, mesh={"nx": n, "ny": n, "refine": 0, "grading": "uniform"},
                                                 time={"lumped": lump})
                                sa = cached_spectral(config_json(c), None)
                                if "lam_max" in sa:
                                    lm_ = sa["lam_max"]
                                    lim = (2 / lm_ if prob.time_order == 1 else 2 / math.sqrt(lm_))
                                    srows.append({"Discrétisation": "DF" if meth == "fdm" else ("EF masse condensée" if lump else
                                                                                               "EF masse cohérente"),
                                                  "μ_max": lm_, "Δt_max (explicite)": lim})
                            st.dataframe(df_numbers(srows, {"μ_max": "{:.4e}", "Δt_max (explicite)": "{:.4e}"}),
                                         hide_index=True, width="stretch")
                if conv_go:
                    with st.spinner("Deux études de convergence…"):
                        convs = {}
                        for meth in ("fdm", "fem"):
                            c = _compare_cfg(base, method=meth, mesh={"refine": 0, "grading": "uniform"})
                            convs[meth] = cached_convergence(config_json(c), 4 if prob.dim == 1 else 3, None, None)
                    store_result("cmp_conv", convs)
                convs = get_result("cmp_conv")
                if convs and all(v.get("ok") for v in convs.values()):
                    fig = go.Figure()
                    P = pal(mode)
                    for i, (meth, cv) in enumerate(convs.items()):
                        hs = [r["h"] for r in cv["rows"]]
                        for j, kk in enumerate(("max", "l2")):
                            fig.add_trace(go.Scatter(x=hs, y=[r[kk] for r in cv["rows"]], mode="lines+markers",
                                                     name=f"{'DF' if meth == 'fdm' else 'EF'} — {kk.upper()} (pente {cv['slopes'].get(kk, float('nan')):.2f})",
                                                     line=dict(color=P["series"][2 * i + j], width=2), marker=dict(size=8)))
                    style_fig(fig, mode, "Convergence comparée", 420, "h", "erreur")
                    fig.update_xaxes(type="log", autorange="reversed", exponentformat="power")
                    fig.update_yaxes(type="log", exponentformat="power")
                    tidy_log_axis(fig, "y", np.array([r[kk] for cv in convs.values() for r in cv["rows"]
                                                      for kk in ("max", "l2")], dtype=float))
                    show_fig(fig, "Les deux méthodes convergent à l'ordre 2 en L² ici ; la norme ∞ des EF peut présenter un "
                                  "facteur |log h| en 2D.", key="cmp_convfig")
            elif name == "Schémas en temps":
                ui_caption("Même maillage et même Δt pour les trois schémas : on observe précision, coût et stabilité.")
                if st.button("Comparer Euler explicite, Crank–Nicolson et Euler implicite", type="primary", key="cmp_t_go"):
                    store_result("cmp_time", True)
                if get_result("cmp_time"):
                    out = {vt: cached_solve(config_json(_compare_cfg(base, time={"theta": vt}))) for vt in (0.0, 0.5, 1.0)}
                    rows, curves = [], []
                    P = pal(mode)
                    for i, (vt, r) in enumerate(out.items()):
                        if not r["ok"]:
                            ui_error("; ".join(r["errors"]), icon=":material/error:")
                            continue
                        e = r.get("errors") or {}
                        stb = stability_report(Problem(_compare_cfg(base, time={"theta": vt})), r)
                        rows.append({"Schéma": {0.0: "Euler explicite", 0.5: "Crank–Nicolson", 1.0: "Euler implicite"}[vt],
                                     "max|G|": stb.get("max_G", stb.get("max_G_fem1d")),
                                     "Explosion": f"pas {r['blowup']['step']}" if r.get("blowup") else "non",
                                     "Erreur L² (T)": e.get("l2"), "Temps (s)": r["timings"].get("time_loop", 0)})
                        if r["hist"].get("err") is not None:
                            E = np.asarray(r["hist"]["err"])
                            curves.append({"x": E[:, 0], "y": np.maximum(E[:, 1], 1e-300), "name": rows[-1]["Schéma"], "slot": i})
                    st.dataframe(df_numbers(rows, {"max|G|": "{:.6f}", "Erreur L² (T)": "{:.3e}", "Temps (s)": "{:.3f}"}),
                                 hide_index=True, width="stretch")
                    if curves:
                        show_fig(fig_curves(curves[0]["x"], curves, mode, "Erreur L² au cours du temps", "t", "erreur", 380,
                                            logy=True), key="cmp_t_fig")
                    _ = P
            elif name == "Schémas pour les ondes":
                if st.button("Comparer saute-mouton et Newmark", type="primary", key="cmp_w_go"):
                    store_result("cmp_wave", True)
                if get_result("cmp_wave"):
                    rows, curves = [], []
                    for i, ws in enumerate(("leapfrog", "newmark")):
                        r = cached_solve(config_json(_compare_cfg(base, time={"wave_scheme": ws})))
                        if not r["ok"]:
                            continue
                        E = np.asarray(r["hist"]["energy"])
                        e = r.get("errors") or {}
                        rows.append({"Schéma": WAVE_SCHEMES[ws], "Explosion": "oui" if r.get("blowup") else "non",
                                     "Dérive relative d'énergie": float((E.max() - E.min()) / abs(E[0])) if len(E) and E[0] else None,
                                     "Erreur L² (T)": e.get("l2"), "Temps (s)": r["timings"].get("time_loop", 0)})
                        curves.append({"x": r["hist"]["t"], "y": E, "name": WAVE_SCHEMES[ws], "slot": i})
                    st.dataframe(df_numbers(rows, {"Dérive relative d'énergie": "{:.2e}", "Erreur L² (T)": "{:.3e}",
                                                    "Temps (s)": "{:.3f}"}), hide_index=True, width="stretch")
                    if curves:
                        show_fig(fig_curves(curves[0]["x"], curves, mode, "Énergie discrète", "t", "E", 360), key="cmp_w_fig")
            else:
                lam = st.slider("λ = aΔt/h commun", 0.05, 1.2, float(min(base["time"]["number"], 1.2)), 0.05, key="cmp_tr_lam")
                sel = st.multiselect("Schémas", list(TRANSPORT_SCHEMES), default=["upwind", "lax_friedrichs", "lax_wendroff", "btcs"],
                                     format_func=TRANSPORT_SCHEMES.get, key="cmp_tr_sel")
                if st.button("Comparer", type="primary", key="cmp_tr_go"):
                    store_result("cmp_tr", True, key=f"{lam}|{','.join(sel)}")
                if get_result("cmp_tr", key=f"{lam}|{','.join(sel)}"):
                    rows, curves = [], []
                    xref = None
                    for i, sc in enumerate(sel):
                        r = cached_solve(config_json(_compare_cfg(base, time={"transport_scheme": sc, "dt_mode": "number",
                                                                              "number": lam})))
                        if not r["ok"]:
                            continue
                        x = r["nodes"][:, 0]
                        xref = x
                        e = r["errors"]
                        vn = r.get("vonneumann_check") or {}
                        rows.append({"Schéma": TRANSPORT_SCHEMES[sc], "Erreur L²": e["l2"], "Erreur max": e["max"],
                                     "Amplitude mode dominant": vn.get("meas_amp"), "Explosion": "oui" if r.get("blowup") else "non"})
                        curves.append({"x": x, "y": r["U"], "name": TRANSPORT_SCHEMES[sc], "slot": i})
                    if xref is not None:
                        r0 = cached_solve(config_json(_compare_cfg(base, time={"dt_mode": "number", "number": lam})))
                        curves.insert(0, {"x": xref, "y": r0["Uex"], "name": "solution exacte", "color": pal(mode)["muted"], "width": 3})
                        show_fig(fig_curves(xref, curves, mode, "Profils à t = T", "x", "u", 430),
                                 "Décentré et Lax–Friedrichs : diffusion numérique (fronts étalés) ; Lax–Wendroff : dispersion "
                                 "(oscillations parasites) ; FTCS : instabilité.", key="cmp_tr_fig")
                    st.dataframe(df_numbers(rows, {"Erreur L²": "{:.3e}", "Erreur max": "{:.3e}", "Amplitude mode dominant": "{:.6f}"}),
                                 hide_index=True, width="stretch")
    next_step("report", "synthèse et interprétation")


def page_report() -> None:
    page_header("Module 13", "Synthèse scientifique et interprétation",
                "Rapport structuré de la dernière résolution ; l'interprétation n'utilise que des grandeurs effectivement "
                "calculées (aucune affirmation inventée).")
    prob = need_valid_problem()
    if prob is None:
        return
    res = get_result("solve")
    if not res or not res.get("ok"):
        ui_info("Aucune résolution pour la configuration actuelle.", icon=":material/info:")
        if st.button("Exécuter la résolution complète", type="primary", icon=":material/rocket_launch:", key="rep_run"):
            run_full_workflow()
            st.rerun()
        return
    store_result("report", True)
    stab = stability_report(prob, res) if prob.time_dependent else None
    conv = get_result("convergence")
    mprops = None
    if not prob.time_dependent and prob.cls != "transport" and res.get("dofs", 1e9) <= 6000:
        mprops = cached_matrix_props(cfg_json_cur(), "K")
    rep = build_report(prob, res, stab, conv if conv and conv.get("ok") else None, mprops, get_result("spectral"))
    secs = rep["sections"]
    cols = st.columns(2)
    for i, (title, items) in enumerate(secs):
        with cols[i % 2]:
            with card(f"rep_{i}"):
                ui_markdown(f"#### {title}")
                rows = [(k, v) for k, v in items if k != "Formulation"]
                if rows:
                    kv_table(rows)
                for k, v in items:
                    if k == "Formulation":
                        st.latex(v)
    with card("rep_interp"):
        card_title("psychology", "INTERPRÉTATION MATHÉMATIQUE")
        for p in rep["interpretation"]:
            ui_markdown(f"- {p}")
    st.download_button("Télécharger le rapport (Markdown)", data=lambda: report_markdown(rep, cfg_norm()),
                       file_name="rapport_pde_lab.md", mime="text/markdown", icon=":material/download:", on_click="ignore")
    next_step("export", "export et reproductibilité")


def _solution_csv(res: dict) -> str:
    nd = res["nodes"]
    df = pd.DataFrame({"x": nd[:, 0]})
    if nd.shape[1] > 1 and res["dim"] == 2:
        df["y"] = nd[:, 1]
    df["u_h"] = res["U"]
    if res.get("Uex") is not None:
        df["u_exacte"] = res["Uex"]
        df["erreur"] = np.asarray(res["U"]) - np.asarray(res["Uex"])
    return df.to_csv(index_label="noeud")


def _frames_csv(res: dict) -> str:
    nd = res["nodes"]
    parts = []
    for t, fr in zip(res["times"], res["frames"]):
        d = pd.DataFrame({"t": t, "noeud": np.arange(len(fr)), "x": nd[:, 0]})
        if res["dim"] == 2:
            d["y"] = nd[:, 1]
        d["u_h"] = fr
        parts.append(d)
    return pd.concat(parts).to_csv(index=False)


def page_export() -> None:
    page_header("Module 14", "Export et reproductibilité",
                "Téléchargez solutions, matrices, erreurs, convergence, rapport et figures ; rechargez une configuration "
                "JSON pour reproduire exactement un calcul.")
    prob = current_problem()
    cfg = cfg_norm()
    meta = {"application": APP_TITLE, "version": APP_VERSION, "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sympy": sympy.__version__, "plotly": plotly.__version__, "streamlit": st.__version__,
            "hash": config_hash(cfg)}
    a, b = st.columns([1.2, 1])
    with a:
        with card("exp_dl"):
            card_title("download", "Téléchargements")
            res = get_result("solve")
            st.download_button("Configuration du problème (JSON)",
                               data=lambda: json.dumps({"config": cfg, "metadata": meta}, indent=2, ensure_ascii=False),
                               file_name=f"pde_lab_config_{meta['hash']}.json", mime="application/json",
                               icon=":material/settings:", on_click="ignore", width="stretch")
            if res and res.get("ok"):
                st.download_button("Solution finale (CSV)", data=lambda: _solution_csv(res), file_name="solution.csv",
                                   mime="text/csv", icon=":material/table_view:", on_click="ignore", width="stretch")
                if res.get("frames") is not None:
                    st.download_button("Évolution temporelle — images stockées (CSV)", data=lambda: _frames_csv(res),
                                       file_name="evolution.csv", mime="text/csv", icon=":material/movie:",
                                       on_click="ignore", width="stretch")
                if res.get("errors"):
                    def _err_csv() -> str:
                        df = pd.DataFrame([res["errors"]])
                        if res.get("hist", {}).get("err") is not None:
                            E = np.asarray(res["hist"]["err"])
                            df2 = pd.DataFrame(E, columns=["t", "erreur_L2_discrete", "erreur_max"])
                            return df.to_csv(index=False) + "\n" + df2.to_csv(index=False)
                        return df.to_csv(index=False)
                    st.download_button("Données d'erreur (CSV)", data=_err_csv, file_name="erreurs.csv", mime="text/csv",
                                       icon=":material/query_stats:", on_click="ignore", width="stretch")
                fig = (fig_solution_1d(res["nodes"][:, 0], res["U"], res.get("Uex"), theme_mode(), "Solution")
                       if res["dim"] == 1 else fig_field_2d(res, res["U"], theme_mode(), "surface", "Solution u_h"))
                st.download_button("Figure de la solution (HTML interactif)", data=lambda: fig.to_html(include_plotlyjs="cdn"),
                                   file_name="solution.html", mime="text/html", icon=":material/image:", on_click="ignore",
                                   width="stretch")
                try:
                    import kaleido  # noqa: F401  (détection de disponibilité)
                    st.download_button("Figure de la solution (PNG, kaleido)", data=lambda: fig.to_image(format="png", scale=2),
                                       file_name="solution.png", mime="image/png", icon=":material/image:",
                                       on_click="ignore", width="stretch")
                except Exception:  # noqa: BLE001
                    ui_caption("Export PNG côté serveur indisponible (paquet facultatif « kaleido » absent) : utilisez "
                               "l'icône appareil photo de chaque graphique, ou l'export HTML.")
                stab = stability_report(prob, res) if prob.ok and prob.time_dependent else None
                conv = get_result("convergence")
                rep = build_report(prob, res, stab, conv if conv and conv.get("ok") else None,
                                   spectral=get_result("spectral"))
                st.download_button("Rapport scientifique (Markdown)", data=lambda: report_markdown(rep, cfg),
                                   file_name="rapport.md", mime="text/markdown", icon=":material/description:",
                                   on_click="ignore", width="stretch")
            else:
                ui_caption("Résolvez le problème pour exporter les solutions et les erreurs.")
            conv = get_result("convergence")
            if conv and conv.get("ok"):
                st.download_button("Résultats de convergence (CSV)", data=lambda: _conv_dataframe(conv).to_csv(index=False),
                                   file_name="convergence.csv", mime="text/csv", icon=":material/trending_down:",
                                   on_click="ignore", width="stretch")
            if prob.ok and prob.cls != "transport":
                asm = cached_assembly(cfg_json_cur())
                if asm.get("ok"):
                    st.download_button("Matrice K (Matrix Market)", data=lambda: _mtx_bytes(asm["K"]), file_name="K.mtx",
                                       mime="text/plain", icon=":material/data_array:", on_click="ignore", width="stretch")
                    st.download_button("Matrice M (Matrix Market)", data=lambda: _mtx_bytes(asm["M"]), file_name="M.mtx",
                                       mime="text/plain", icon=":material/data_array:", on_click="ignore", width="stretch")
    with b:
        with card("exp_import"):
            card_title("upload", "Reproduire un calcul")
            up = st.file_uploader("Configuration JSON exportée", type=["json"], key="exp_upload")
            if up is not None:
                try:
                    data = json.loads(up.getvalue().decode("utf-8"))
                    cfg_in = data.get("config", data) if isinstance(data, dict) else None
                    if not isinstance(cfg_in, dict) or cfg_in.get("schema") != "pdelab-config":
                        raise ValueError("ce fichier n'est pas une configuration PDE Numerical Laboratory")
                    newcfg, notes = normalize_config(cfg_in)
                    pr = Problem(newcfg)
                    ui_markdown(f"Configuration lue : **{newcfg.get('name', '')}** — {pr.describe()}")
                    if pr.errors:
                        ui_error("; ".join(pr.errors), icon=":material/error:")
                    for n_ in notes:
                        ui_caption(n_)
                    if st.button("Charger cette configuration", type="primary", key="exp_load", disabled=bool(pr.errors)):
                        set_config(newcfg, "Configuration importée")
                        st.rerun()
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    ui_error(f"Fichier invalide : {exc}", icon=":material/error:")
            ui_caption("Les expressions importées sont à nouveau analysées par l'interpréteur sécurisé ; aucune donnée "
                       "n'est exécutée.")
        with card("exp_env"):
            card_title("memory", "Environnement d'exécution")
            kv_table([(k, v) for k, v in meta.items() if k not in ("date",)])
            ui_caption("Calcul déterministe : maillages perturbés générés avec une graine fixe, solveurs directs, "
                       "vecteurs de départ ARPACK fixés.")
    with st.expander("Configuration courante (JSON)", icon=":material/data_object:"):
        st.json(cfg, expanded=False)


def page_library() -> None:
    page_header("Module 11", "Bibliothèque d'exemples et validation",
                "Problèmes de référence à solution connue et exemples illustrant les phénomènes numériques ; suite de "
                "tests automatiques contrôlant calculs, matrices et résultats.")
    tabs = st.tabs(["Exemples", "Suite de validation automatique"])
    with tabs[0]:
        groups = ["Elliptique", "Parabolique", "Hyperbolique", "Transport"]
        g = st.segmented_control("Catégorie", ["Tous"] + groups, default="Tous", key="lib_group", required=True) or "Tous"
        exs = [e for e in EXAMPLES if g in (None, "Tous") or e["group"] == g]
        cols = st.columns(3)
        for i, ex in enumerate(exs):
            with cols[i % 3]:
                with card(f"lib_{ex['id']}"):
                    ui_markdown(f"**{ex['title']}**")
                    cfg_ex = ex["cfg"]
                    meth = cfg_ex.get("method", "fdm")
                    ui_markdown(badge(ex["group"], "blue") + " " + badge(f"{cfg_ex.get('dim', 1)}D", "gray") + " "
                                + badge("EF" if meth == "fem" and ex["group"] != "Transport" else "DF", "gray") + " "
                                + (badge("solution exacte", "green") if cfg_ex.get("exact") or ex["group"] == "Transport" else ""),
                                unsafe_allow_html=True)
                    ui_caption(ex["desc"])
                    if st.button("Charger", key=f"lib_load_{ex['id']}", icon=":material/download_for_offline:"):
                        load_example(ex["id"])
                        pages = st.session_state.get("_pages", {})
                        if "dashboard" in pages:
                            st.switch_page(pages["dashboard"])
                        st.rerun()
    with tabs[1]:
        ui_markdown("La suite exécute les problèmes de référence (Tests 1 à 5) et des contrôles unitaires : sécurité de "
                    "l'analyseur, fonctions de forme, quadratures, matrices, conditions aux limites, ordres de "
                    "convergence, critères de stabilité, Lax–Milgram, solveurs. Les valeurs attendues sont théoriques ; "
                    "elles ne sont jamais ajustées aux résultats.")
        st.code("python main.py --selftest", language="bash")
        if st.button("Exécuter la suite de validation", type="primary", icon=":material/fact_check:", key="val_run"):
            st.session_state["val_stamp"] = st.session_state.get("val_stamp", 0) + 1
            with st.spinner("Validation en cours (quelques secondes)…"):
                cached_validation(st.session_state["val_stamp"])
        stamp = st.session_state.get("val_stamp")
        if stamp:
            res = cached_validation(stamp)
            npass = sum(r["passed"] for r in res)
            c = st.columns(3)
            c[0].metric("Tests réussis", f"{npass} / {len(res)}", border=True)
            c[1].metric("Durée totale", f"{sum(r['time'] for r in res):.2f} s", border=True)
            c[2].metric("Statut", "VALIDÉ" if npass == len(res) else "ÉCHECS", border=True)
            df = pd.DataFrame([{"Catégorie": r["category"], "Test": r["test"], "Statut": "réussi" if r["passed"] else "ÉCHEC",
                                "Mesuré": r["measured"], "Attendu": r["expected"], "Durée (s)": f"{r['time']:.2f}"}
                               for r in res])
            st.dataframe(df, hide_index=True, width="stretch", height=min(60 + 36 * len(df), 900))


def page_docs() -> None:
    page_header("Documentation", "Guide, périmètre et limites",
                "Ce que fait l'application, avec quelles garanties, et ce qu'elle ne fait pas.")
    a, b = st.columns(2)
    with a:
        with card("doc_classes"):
            card_title("category", "Classes de problèmes prises en charge")
            st.dataframe(pd.DataFrame([{"Famille": m["label"], "Classe": CLASS_LABELS[m["class"]],
                                        "Dimensions": ", ".join(f"{d}D" for d in m["dims"]),
                                        "Méthodes": ", ".join({"fdm": "DF", "fem": "EF P1"}[x] for x in m["methods"]),
                                        "Conditions aux limites": ", ".join(BC_LABELS[t] for t in m["bc_types"]) or "périodique / entrée"}
                                       for m in FAMILIES.values()]), hide_index=True, width="stretch")
            ui_caption("Domaines : intervalle, rectangle, domaine en L (EF). Coefficients k(x, y), c(x, y) variables ; "
                       "vitesses b, a, κ et c constantes. Équations linéaires uniquement.")
        with card("doc_methods"):
            card_title("build", "Méthodes numériques implémentées")
            ui_markdown(
                "- **Différences finies** : schémas 3/5 points d'ordre 2, coefficients aux demi-points, Neumann/Robin "
                "par demi-maille (point fantôme), advection centrée ou décentrée.\n"
                "- **Éléments finis P1** conformes : maillages 1D (uniforme, perturbé, Tchebychev), triangulations 2D "
                "(4 motifs, perturbation, raffinement rouge emboîté), quadratures de degré 4 (assemblage) et 5 (erreurs).\n"
                "- **Temps** : θ-schéma (explicite, Crank–Nicolson, implicite, ϑ quelconque), masse cohérente ou "
                "condensée ; ondes : saute-mouton et Newmark β = 1/4 ; transport : décentré, FTCS, Lax–Friedrichs, "
                "Lax–Wendroff, centré implicite.\n"
                "- **Solveurs** : SuperLU (COLAMD), CG, GMRES(50), BiCGSTAB, préconditionneurs Jacobi / ILU ; Neumann pur "
                "par multiplicateur de Lagrange.\n"
                "- **Analyse** : von Neumann (symboles exacts), spectre généralisé K z = μ M z (dense ou ARPACK), "
                "perturbations, convergence espace/temps, normes L², H¹, ∞.")
        with card("doc_guide"):
            card_title("route", "Parcours recommandé")
            ui_markdown("1. **Définition** : famille, géométrie, coefficients, CL, solution exacte (ou manufacturée).\n"
                        "2. **Formulations** forte puis faible, **espaces** et **existence** (Lax–Milgram ou cadre d'évolution).\n"
                        "3. **Discrétisation**, **maillage**, **matrices**.\n4. **Résolution**, **stabilité**.\n"
                        "5. **Convergence**, **visualisation**, **comparaison**, **synthèse**, **export**.\n\n"
                        "Le bouton « EXÉCUTER LA RÉSOLUTION COMPLÈTE » du tableau de bord enchaîne toutes ces étapes.")
    with b:
        with card("doc_limits"):
            card_title("report", "Limites — ce qui n'est PAS traité")
            ui_markdown(
                "- EDP non linéaires, systèmes d'EDP, problèmes 3D.\n"
                "- Advection–diffusion stationnaire, opérateurs à coefficients principaux anisotropes ou de signe variable "
                "(k ≤ 0 est refusé : problème non elliptique).\n"
                "- Stabilisation SUPG / Galerkin discontinu : non implémentées ; en régime advectif (Pe_h > 1) le Galerkin "
                "standard peut osciller — l'application le signale sans le corriger.\n"
                "- Maillages non structurés quelconques (import), adaptation de maillage, éléments d'ordre élevé.\n"
                "- Différences finies sur grille non uniforme ou domaine non rectangulaire.\n"
                "- Conditions de Robin pour l'équation des ondes ; conditions périodiques hors transport.\n"
                "- Les bornes de coefficients variables sont obtenues par échantillonnage fin : elles sont signalées comme "
                "« vérifiées numériquement », jamais comme démontrées.")
        with card("doc_notation"):
            card_title("translate", "Notations")
            st.dataframe(pd.DataFrame([
                ("Ω, ∂Ω, n", "domaine, bord, normale unitaire sortante"), ("Γ₁…Γ₄", "bords gauche, droite, bas, haut"),
                ("Γ_D, Γ_N, Γ_R", "parties Dirichlet, Neumann, Robin du bord"), ("k, c, b, β", "diffusion, réaction, vitesse, coefficient de Robin"),
                ("h, Δt", "pas d'espace, pas de temps"), ("r, σ, λ", "k Δt/h², c Δt/h, a Δt/h"),
                ("ϑ", "paramètre du θ-schéma"), ("θ = ξh", "angle de phase (von Neumann)"),
                ("G(θ)", "facteur d'amplification"), ("K, M, F", "rigidité, masse, second membre"),
                ("α, M", "constantes de coercivité et de continuité"), ("u, u_h, U", "solution exacte, approchée, vecteur nodal")],
                columns=["Symbole", "Signification"]), hide_index=True, width="stretch")
        with card("doc_security"):
            card_title("shield", "Sécurité des expressions")
            ui_markdown("Les saisies sont découpées par un analyseur lexical puis interprétées par un analyseur récursif "
                        "qui ne reconnaît qu'une liste blanche (x, y, t, pi, e, fonctions usuelles). L'arbre SymPy obtenu "
                        "est évalué par un compilateur maison en fonctions NumPy : aucun appel à eval, exec, sympify ou "
                        "parse_expr sur une chaîne utilisateur.")
        with card("doc_refs"):
            card_title("menu_book", "Références")
            ui_markdown("- G. Allaire, *Analyse numérique et optimisation*, Éditions de l'École polytechnique.\n"
                        "- H. Brezis, *Analyse fonctionnelle*, Dunod.\n- A. Ern, J.-L. Guermond, *Theory and Practice of "
                        "Finite Elements*, Springer.\n- R. J. LeVeque, *Finite Difference Methods for ODEs and PDEs*, SIAM.\n"
                        "- J. C. Strikwerda, *Finite Difference Schemes and PDEs*, SIAM.\n- A. Quarteroni, A. Valli, "
                        "*Numerical Approximation of PDEs*, Springer.\n- P. Grisvard, *Elliptic Problems in Nonsmooth "
                        "Domains* (singularités des coins rentrants).")


# =============================================================================
# 12g. NAVIGATION, EN-TÊTE, BARRE LATÉRALE, POINT D'ENTRÉE
# =============================================================================
PAGE_SPECS: list[tuple[str, str, str, str, int | None, Callable[[], None]]] = [
    ("dashboard", "Tableau de bord", ":material/space_dashboard:", "Accueil", None, page_dashboard),
    ("definition", "Définition de l'EDP", ":material/edit_note:", "Problème", 1, page_definition),
    ("strong", "Formulation forte", ":material/functions:", "Problème", 1, page_strong),
    ("weak", "Formulation faible", ":material/integration_instructions:", "Analyse mathématique", 1, page_weak),
    ("spaces", "Espaces fonctionnels", ":material/hub:", "Analyse mathématique", 1, page_spaces),
    ("existence", "Existence et unicité", ":material/verified:", "Analyse mathématique", 1, page_existence),
    ("discretization", "Discrétisation", ":material/grid_on:", "Discrétisation", 2, page_discretization),
    ("mesh", "Maillage", ":material/change_history:", "Discrétisation", 3, page_mesh),
    ("matrices", "Matrices et assemblage", ":material/data_array:", "Discrétisation", 3, page_matrices),
    ("solve", "Résolution numérique", ":material/play_circle:", "Calcul", 4, page_solve),
    ("stability", "Stabilité et CFL", ":material/monitor_heart:", "Calcul", 4, page_stability),
    ("convergence", "Convergence et erreurs", ":material/trending_down:", "Calcul", 5, page_convergence),
    ("visualization", "Visualisation", ":material/insights:", "Résultats", 5, page_visualization),
    ("comparison", "Comparaison des méthodes", ":material/compare_arrows:", "Résultats", 5, page_comparison),
    ("report", "Synthèse et interprétation", ":material/summarize:", "Résultats", 5, page_report),
    ("export", "Export et reproductibilité", ":material/download:", "Résultats", 5, page_export),
    ("library", "Bibliothèque d'exemples", ":material/library_books:", "Ressources", None, page_library),
    ("docs", "Documentation", ":material/menu_book:", "Ressources", None, page_docs),
]
STAGES = [(1, "Problème"), (2, "Méthode"), (3, "Maillage"), (4, "Résolution"), (5, "Résultats")]


def stages_done(prob: Problem) -> set[int]:
    done: set[int] = set()
    if not prob.ok:
        return done
    done.add(1)
    try:
        nn, nt = estimate_cost(prob, prob.cfg)
        if nt <= LIMITS["nt_max"] and nn * max(nt, 1) <= LIMITS["work_max"]:
            done.add(2)
    except Exception:  # noqa: BLE001
        pass
    r = get_result("solve")
    solved = bool(r and r.get("ok"))
    if get_result("assembly") or solved:
        done.add(3)
    if solved:
        done.add(4)
        if get_result("report") or get_result("convergence"):
            done.add(5)
    return done


def render_header(prob: Problem) -> None:
    r = get_result("solve")
    if not prob.ok:
        dot, txt = "bad", "Définition invalide"
    elif r and r.get("ok") and r.get("blowup"):
        dot, txt = "warn", "Instabilité détectée"
    elif r and r.get("ok"):
        dot, txt = "ok", "Solution à jour"
    elif result_is_stale("solve"):
        dot, txt = "warn", "Résultats obsolètes"
    else:
        dot, txt = "idle", "Système prêt"
    ui_markdown(f'<div class="pdl-header"><div class="pdl-brand">{LOGO_SVG}<div><div class="pdl-title">{APP_TITLE}</div>'
                f'<div class="pdl-subtitle">{APP_SUBTITLE}</div></div></div>'
                f'<div class="pdl-pill"><span class="pdl-dot {dot}"></span>{txt}</div></div>', unsafe_allow_html=True)


def render_stepper(active_stage: int | None, done: set[int]) -> None:
    parts = []
    for num, label in STAGES:
        cls = "pdl-step"
        if num == active_stage:
            cls += " active"
        elif num in done:
            cls += " done"
        mark = CHECK_SVG if (num in done and num != active_stage) else str(num)
        parts.append(f'<div class="{cls}"><span class="num">{mark}</span><span class="lbl">{label}</span></div>')
    ui_markdown('<div class="pdl-stepper">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def _on_theme_change() -> None:
    choice = st.session_state.get("ui_theme")
    if choice:
        st.session_state.theme_mode = "dark" if choice == "Sombre" else "light"


def render_sidebar(prob: Problem, done: set[int]) -> None:
    with st.sidebar:
        ui_markdown("**Progression de la résolution**")
        rows = "".join(f'<div class="row{" done" if n in done else ""}"><span class="mk">'
                       f'{CHECK_SVG if n in done else ""}</span>{n}. {lab}</div>' for n, lab in STAGES)
        ui_markdown(f'<div class="pdl-prog">{rows}</div>', unsafe_allow_html=True)
        st.progress(len(done) / len(STAGES))
        st.segmented_control("Thème", ["Sombre", "Clair"], default="Sombre" if theme_mode() == "dark" else "Clair",
                             key="ui_theme", on_change=_on_theme_change, required=True)
        cfg = cfg_norm()
        ui_markdown(f'<div class="pdl-side-card"><b>{APP_TITLE}</b><br>Version {APP_VERSION}<br>'
                    f'Configuration #{config_hash(cfg)}<br>{prob.meta["label"]} — {cfg["dim"]}D — '
                    f'{"EF P1" if cfg["method"] == "fem" and prob.cls != "transport" else "DF"}</div>',
                    unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/change_history:", layout="wide",
                       initial_sidebar_state="expanded", menu_items={"About": f"{APP_TITLE} {APP_VERSION} — {APP_SUBTITLE}"})
    init_state()
    if "ui_theme" in st.session_state and st.session_state.ui_theme in ("Sombre", "Clair"):
        st.session_state.theme_mode = "dark" if st.session_state.ui_theme == "Sombre" else "light"
    try:
        st.set_option("client.toolbarMode", "viewer")
    except Exception:  # noqa: BLE001
        pass
    if apply_runtime_theme(theme_mode()):
        st.rerun()
    inject_css(theme_mode())
    pages: dict[str, Any] = {}
    groups: dict[str, list] = {}
    for key, title, icon, group, _, fn in PAGE_SPECS:
        page = st.Page(fn, title=title, icon=icon, url_path=key, default=(key == "dashboard"))
        pages[key] = page
        groups.setdefault(group, []).append(page)
    st.session_state["_pages"] = pages
    nav = st.navigation(groups, position="sidebar", expanded=True)
    active = next((k for k, p in pages.items() if p.url_path == nav.url_path or p.title == nav.title), "dashboard")
    stage = next((s for k, *_, s, _ in PAGE_SPECS if k == active), None)
    prob = current_problem()
    done = stages_done(prob)
    render_header(prob)
    render_stepper(stage, done)
    render_sidebar(prob, done)
    flash_messages()
    try:
        nav.run()
    except (ConfigError, ExpressionError) as exc:
        ui_error(f"Configuration invalide : {exc}", icon=":material/error:")


def _cli_selftest() -> int:
    print(f"{APP_TITLE} {APP_VERSION} — suite de validation automatique")
    print(f"Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__} | "
          f"SymPy {sympy.__version__} | Plotly {plotly.__version__} | Streamlit {st.__version__}\n")
    t0 = time.perf_counter()
    results = run_validation_suite()
    for r in results:
        print(f"[{'OK ' if r['passed'] else 'ÉCHEC'}] {r['time']:6.2f}s  {r['category']:<24} {r['test']}")
        print(f"         mesuré : {r['measured']}   |   attendu : {r['expected']}")
    npass = sum(r["passed"] for r in results)
    print(f"\n{npass}/{len(results)} tests réussis en {time.perf_counter() - t0:.1f} s")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_cli_selftest())
    try:
        from streamlit.runtime import exists as _st_runtime_exists
        _RUNNING_IN_STREAMLIT = _st_runtime_exists()
    except Exception:  # noqa: BLE001
        _RUNNING_IN_STREAMLIT = False
    if _RUNNING_IN_STREAMLIT:
        main()
    else:
        print(f"{APP_TITLE} — lancez l'application avec :  streamlit run main.py\n"
              "Auto-tests numériques :                   python main.py --selftest")
