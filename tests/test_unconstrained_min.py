
"""Automated tests & visualisations for the optimization routines."""
import os
import unittest
import numpy as np
import csv
import pathlib

results_summary = []

from src.unconstrained_min import minimize
from src.utils import plot_contours, plot_fvals
from tests.examples import FUNCTIONS_2D

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(_PLOTS_DIR, exist_ok=True)

class TestUnconstrainedMin(unittest.TestCase):
    def _run_case(self, name, func):
        if name == 'Rosenbrock':
            x0 = np.array([-1.0, 2.0])
            gd_max_iter = 10_000
        else:
            x0 = np.array([1.0, 1.0])
            gd_max_iter = 100

        # Gradient‑descent
        x_gd, f_gd, ok_gd, n_iter_gd, stop_reason_gd, path_gd, fvals_gd = minimize(
            func, x0,
            obj_tol=1e-12, param_tol=1e-8, max_iter=gd_max_iter,
            method='gd'
        )
        # Newton
        x_nt, f_nt, ok_nt, n_iter_nt, stop_reason_nt, path_nt, fvals_nt = minimize(
            func, x0,
            obj_tol=1e-12, param_tol=1e-8, max_iter=100,
            method='newton'
        )

        # # Assertions
        # self.assertTrue(ok_gd, f'GD failed on {name}')
        # self.assertTrue(ok_nt, f'Newton failed on {name}')

        # Record results for summary
        results_summary.append({
            "case": name,
            "method": "gd",
            "success": ok_gd,
            "n_iter": n_iter_gd,
            "stop_reason": stop_reason_gd,
            "final_x": x_gd.tolist(),
            "final_f": f_gd,
        })
        results_summary.append({
            "case": name,
            "method": "newton",
            "success": ok_nt,
            "n_iter": n_iter_nt,
            "stop_reason": stop_reason_nt,
            "final_x": x_nt.tolist(),
            "final_f": f_nt,
        })

        # Plotting

        # # Contours + paths
        pth = os.path.join(_PLOTS_DIR, f'{name}_contours.png')
        plot_contours(
            func,
            xlim=(-2, 2), ylim=(-2, 2),
            paths=[path_gd, path_nt],
            labels=['GD', 'Newton'],
            title=f'{name} - Trajectories',
            save=pth
        )

        # fvals
        pth = os.path.join(_PLOTS_DIR, f'{name}_fvals.png')
        plot_fvals(
            [fvals_gd, fvals_nt],
            labels=['GD', 'Newton'],
            save=pth
        )

        # Last iteration report
        print(f"\n{name}: GD final x={x_gd}, f={f_gd:.3e}, {stop_reason_gd}; "
              f"NT final x={x_nt}, f={f_nt:.3e}, {stop_reason_nt}\n")
        
        # Assertions
        self.assertTrue(ok_gd, f'GD failed on {name}')
        self.assertTrue(ok_nt, f'Newton failed on {name}')

    # Dynamically generate a test for each example
    def test_examples(self):
        for name, func in FUNCTIONS_2D.items():
            with self.subTest(case=name):
                self._run_case(name, func)

    @classmethod
    def tearDownClass(cls):
        if results_summary:
            out = pathlib.Path("results_summary.csv")
            with out.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=results_summary[0].keys())
                writer.writeheader()
                writer.writerows(results_summary)
            print(f"\nWrote summary for {len(results_summary)} runs → {out.resolve()}")

if __name__ == '__main__':
    unittest.main()
