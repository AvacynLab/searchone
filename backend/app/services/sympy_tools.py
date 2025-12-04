from typing import List, Dict

import sympy as sp


def simplify_expression(expr: str) -> str:
    try:
        return str(sp.simplify(expr))
    except Exception:
        return expr


def solve_equation(lhs: str, rhs: str = "0", symbol: str = "x") -> Dict[str, List[str]]:
    try:
        sym = sp.symbols(symbol)
        sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), sym)
        return {"solutions": [str(s) for s in sol]}
    except Exception:
        return {"solutions": []}
