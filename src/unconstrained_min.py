"""
Numerical Optimisation - unconstrained line-search solver.

Supports:
* Gradient-Descent  (method="gd")
* Newton            (method="newton")

Both directions use a back-tracking line-search that satisfies the Armijo
(decrease) part of the weak Wolfe conditions.

The solver reports `success = False` in every failure mode required by the
specification:

1. **Max-iteration reached** - loop exit.
2. **Line-search gave up** - step length under-flow.
3. **Non-finite numbers encountered** - NaN / Inf in f or ∇f.
4. **User-supplied `terminate()` callback returns** - early abort.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------
# Back-tracking line-search (Armijo decrease)
# ---------------------------------------------------------------------
def _backtracking_wolfe(
    f,
    x: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-2,
    rho: float = 0.5,
    min_alpha: float = 1e-16,
) -> tuple[float, bool]:
    """
    Parameters
    ----------
    f: objective returning (f, g, H)
    x:  current iterate
    p: search direction
    g: gradient at x
    alpha0: initial trial step
    c1, rho: Armijo constant and back-tracking factor
    min_alpha: lower bound under which the search fails

    Returns
    -------
    alpha: accepted step length
    success : True if Armijo condition met before hitting min_alpha
    """
    alpha = float(alpha0)
    f_x, _, _ = f(x, hess=False)
    slope = c1 * np.dot(g, p)

    # Guard: non-descent direction  →  switch to steepest descent
    if slope >= 0.0:
        p = -g
        slope = c1 * (-np.dot(g, g))

    while alpha >= min_alpha:
        x_new = x + alpha * p
        f_new, _, _ = f(x_new, hess=False)
        if f_new <= f_x + alpha * slope:      
            return alpha, True
        alpha *= rho

    # Failed to find a sufficient step
    return alpha, False

# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------
def minimize(
    f,
    x0,
    *,
    obj_tol: float = 1e-12,
    param_tol: float = 1e-8,
    max_iter: int = 100,
    method: str = "gd",
    alpha0: float = 1.0,
    c1: float = 1e-2,
    rho: float = 0.5,
    min_alpha: float = 1e-16,
    terminate=None,
):
    """
    Unconstrained minimisation with robust failure detection.

    Parameters
    ----------
    f: objective with signature  f(x, hess=False) → (f, g, H)
    x0: starting point
    obj_tol: |f_{k+1} - f_k|   stopping threshold
    param_tol: ||x_{k+1} - x_k|| stopping threshold
    max_iter:  iteration limit
    method: {"gd", "newton"}
    alpha0,c1,rho,min_alpha : line-search parameters
    terminate  : callable(it, x, f, g) → bool | None
                 Early-abort hook supplied by the user. When it returns True
                 the run stops and reports success=False.

    Returns
    -------
    x_best  : ndarray
    f_best  : float
    success : bool
    path    : (m, n) ndarray- iterates (including x0, x_best)
    fvals   : (m,)  ndarray- objective values along the path
    """
    x = np.asarray(x0, dtype=float).copy()
    f_val, g, H = f(x, hess=(method == "newton"))
    path  = [x.copy()]
    fvals = [float(f_val)]

    for it in range(1, max_iter + 1):

        # ---------------- search direction ----------------
        if method == "gd":
            p = -g
        elif method == "newton":
            try:
                p = -np.linalg.solve(H, g)           # H p = −g  (no explicit inverse!)
            except np.linalg.LinAlgError:
                print(f"[{it:3d}] Hessian singular - falling back to gradient direction.")
                p = -g
        else:
            raise ValueError("method must be 'gd' or 'newton'")

        # tiny step → convergence
        if np.linalg.norm(p) < param_tol:
            stop_reason = "|p| < param_tol - convergence."
            print(f"[{it:3d}] {stop_reason}")
            return x, f_val, True, it, stop_reason, np.vstack(path), np.array(fvals)

        # ---------------- line-search ----------------
        alpha, ls_ok = _backtracking_wolfe(
            f, x, p, g,
            alpha0=alpha0, c1=c1, rho=rho, min_alpha=min_alpha
        )
        if not ls_ok:
            stop_reason = f"line-search failed (alpha dropped < {min_alpha:g})."
            print(f"[{it:3d}] {stop_reason}")
            return x, f_val, False, it, stop_reason, np.vstack(path), np.array(fvals)
        
        s = alpha * p
        x_next = x + s
        f_next, g_next, H_next = f(x_next, hess=(method == "newton"))

        # ---------------- non-finite guard ----------------
        if (not np.isfinite(f_next)) or (not np.all(np.isfinite(g_next))):
            stop_reason = "non-finite value encountered - aborting."
            print(f"[{it:3d}] {stop_reason}")
            return x, f_val, False, it, stop_reason, np.vstack(path), np.array(fvals)

        # ---------------- bookkeeping ----------------
        path.append(x_next.copy())
        fvals.append(float(f_next))

        # optional user early-abort
        if terminate is not None and terminate(it, x_next, f_next, g_next):
            stop_reason = "terminate() returned True - aborting."
            print(f"[{it:3d}] {stop_reason}")
            return x_next, f_next, False, it, stop_reason, np.vstack(path), np.array(fvals)

        # ------------- convergence tests -------------
        if abs(f_next - f_val) < obj_tol:
            stop_reason = "|f_next - f| < obj_tol - convergence."
            print(f"[{it:3d}] {stop_reason}")
            return x_next, f_next, True, it, stop_reason, np.vstack(path), np.array(fvals)
          
        if np.linalg.norm(s) < param_tol:
            stop_reason = "|s| < param_tol - convergence."
            print(f"[{it:3d}] {stop_reason}")
            return x_next, f_next, True, it, stop_reason, np.vstack(path), np.array(fvals)
        

        # prep for next iteration
        x, f_val, g, H = x_next, f_next, g_next, H_next

    # -------- max_iter exhausted --------
    stop_reason = "max_iter exceeded"
    print(stop_reason)
    return x, f_val, False, max_iter, stop_reason, np.vstack(path), np.array(fvals)
# ---------------------------------------------------------------------
