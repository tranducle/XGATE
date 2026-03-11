"""
SymPy Tools for Local Mathematical Verification

This module provides tools using the SymPy library for local, offline, and free
mathematical verification. It serves as a complement to Wolfram Alpha tools.

Prerequisites:
    pip install sympy
"""

import sys
from typing import Dict, Any, Optional, List, Union

# Try to import sympy
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympy = None


def _check_sympy():
    """Check if SymPy is available and raise error if not."""
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is not installed. Please run: pip install sympy")


def _parse_expression(expr_str: str):
    """Parse a string into a SymPy expression with loose syntax."""
    _check_sympy()
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        # Handle simple equations "a = b" -> "a - b"
        if "=" in expr_str:
            lhs, rhs = expr_str.split("=", 1)
            # Remove == if present
            if rhs.startswith("="):
                rhs = rhs[1:]
            return parse_expr(lhs, transformations=transformations) - parse_expr(
                rhs, transformations=transformations
            )
        return parse_expr(expr_str, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Could not parse expression '{expr_str}': {e}")


def verify_equation_sympy(lhs: str, rhs: str) -> Dict[str, Any]:
    """
    Verify if two expressions are mathematically equivalent.
    Checks if simplify(lhs - rhs) == 0.

    Args:
        lhs: Left-hand side expression (e.g. "(a+b)^2")
        rhs: Right-hand side expression (e.g. "a^2 + 2ab + b^2")

    Returns:
        Dict result with 'verified' boolean.
    """
    _check_sympy()
    try:
        expr_lhs = _parse_expression(lhs)
        expr_rhs = _parse_expression(rhs)

        diff = sympy.simplify(expr_lhs - expr_rhs)
        is_zero = diff == 0

        return {
            "verified": is_zero,
            "lhs": str(expr_lhs),
            "rhs": str(expr_rhs),
            "difference": str(diff),
            "simplified_diff": str(diff),
        }
    except Exception as e:
        return {"verified": False, "error": str(e)}


def compute_derivative_sympy(expression: str, variable: str = "x") -> Dict[str, Any]:
    """
    Compute symbolic derivative.

    Args:
        expression: Math expression
        variable: Variable to differentiate with respect to
    """
    _check_sympy()
    try:
        expr = _parse_expression(expression)
        var = sympy.Symbol(variable)
        derivative = sympy.diff(expr, var)
        return {
            "success": True,
            "expression": str(expr),
            "derivative": str(derivative),
            "variable": variable,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_integral_sympy(expression: str, variable: str = "x") -> Dict[str, Any]:
    """Compute symbolic indefinite integral."""
    _check_sympy()
    try:
        expr = _parse_expression(expression)
        var = sympy.Symbol(variable)
        integral = sympy.integrate(expr, var)
        return {
            "success": True,
            "expression": str(expr),
            "integral": str(integral),
            "variable": variable,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def simplify_expression_sympy(expression: str) -> Dict[str, Any]:
    """Simplify a mathematical expression."""
    _check_sympy()
    try:
        expr = _parse_expression(expression)
        simplified = sympy.simplify(expr)
        expanded = sympy.expand(expr)
        factored = sympy.factor(expr)

        return {
            "success": True,
            "original": str(expr),
            "simplified": str(simplified),
            "expanded": str(expanded),
            "factored": str(factored),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


SYMPY_TOOLS = [
    {
        "name": "verify_equation_sympy",
        "description": "Verify math equality locally",
        "function": verify_equation_sympy,
    },
    {
        "name": "compute_derivative_sympy",
        "description": "Symbolic derivative locally",
        "function": compute_derivative_sympy,
    },
    {
        "name": "compute_integral_sympy",
        "description": "Symbolic indefinite integral locally",
        "function": compute_integral_sympy,
    },
    {
        "name": "simplify_expression_sympy",
        "description": "Simplify expression locally",
        "function": simplify_expression_sympy,
    },
]

if __name__ == "__main__":
    if not SYMPY_AVAILABLE:
        print("SymPy not installed. Skipping tests.")
    else:
        print("Running SymPy Tests...")
        print(
            "1. Verify (a+b)^2:", verify_equation_sympy("(a+b)^2", "a^2 + 2*a*b + b^2")
        )
        print("2. Derivative x^3:", compute_derivative_sympy("x^3", "x"))
        print(
            "3. Simplify sin^2 + cos^2:",
            simplify_expression_sympy("sin(x)**2 + cos(x)**2"),
        )
