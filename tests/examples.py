
"""Collection of objective functions for testing the optimisation algorithms.

Each function follows the interface:
    f(x, hess=False) -> (f, g, H)
where:
    x    : numpy (n,)
    hess : bool - if True, compute Hessian
Returns:
    f : float               - function value
    g : numpy (n,)          - gradient
    H : numpy (n, n) | None - Hessian if requested, else None
"""

import numpy as np
from math import sqrt, exp

TWO_D_EYE = 2.0 * np.eye(2)

# ---------------------------------------------------------------------
def _quad_factory(Q):
    """Return a quadratic objective f(x) = xᵀ Q x with given positive-definite Q."""
    Q = np.asarray(Q, dtype=float)

    def quad(x, hess=False):
        x = np.asarray(x, dtype=float).ravel()
        f = float(x @ Q @ x)
        g = 2.0 * (Q @ x)
        H = 2.0 * Q if hess else None
        return f, g, H
    return quad

# (i) isotropic – circles
quad_iso = _quad_factory(np.eye(2))

# (ii) axis‑aligned ellipse
quad_axis = _quad_factory(np.diag([1.0, 100.0]))

# (iii) rotated ellipse
theta = np.radians(30.0)  # 30°
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
D = np.diag([100.0, 1.0])
Q_rot = R.T @ D @ R
quad_rot = _quad_factory(Q_rot)

# ---------------------------------------------------------------------
# Rosenbrock
def rosenbrock(x, hess=False):
    x = np.asarray(x, dtype=float).ravel()
    x1, x2 = x
    f = 100.0 * (x2 - x1 ** 2) ** 2 + (1.0 - x1) ** 2
    g = np.array([
        -400.0 * x1 * (x2 - x1 ** 2) - 2.0 * (1.0 - x1),
        200.0 * (x2 - x1 ** 2),
    ])
    if not hess:
        return f, g, None
    H = np.array([
        [1200.0 * x1 ** 2 - 400.0 * x2 + 2.0, -400.0 * x1],
        [-400.0 * x1, 200.0],
    ])
    return f, g, H

# ---------------------------------------------------------------------
# Linear
a_vec = np.array([3.0, -4.0])

def linear(x, hess=False):
    x = np.asarray(x, dtype=float).ravel()
    f = float(a_vec @ x)
    g = a_vec.copy()
    H = np.zeros((2, 2)) if hess else None
    return f, g, H

# ---------------------------------------------------------------------
# Exponential "triangle" – Boyd example 9.20
def exp_triangle(x, hess=False):
    x = np.asarray(x, dtype=float).ravel()
    x1, x2 = x
    e1 = exp(x1 + 3 * x2 - 0.1)
    e2 = exp(x1 - 3 * x2 - 0.1)
    e3 = exp(-x1 - 0.1)
    f = e1 + e2 + e3
    g = np.array([
        e1 + e2 - e3,
        3 * e1 - 3 * e2,
    ])
    if not hess:
        return f, g, None
    H = np.array([
        [e1 + e2 + e3, 3 * e1 - 3 * e2],
        [3 * e1 - 3 * e2, 9 * e1 + 9 * e2],
    ])
    return f, g, H

# ---------------------------------------------------------------------
# Dictionary for convenient iteration in tests
FUNCTIONS_2D = {
    "Quad-Isotropic": quad_iso,
    "Quad-Axis": quad_axis,
    "Quad-Rotated": quad_rot,
    "Rosenbrock": rosenbrock,
    "Linear": linear,
    "Exp-Triangle": exp_triangle,
}
