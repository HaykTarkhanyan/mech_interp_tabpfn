"""Ground-truth checks (transfer_pilot_v2, frontier_probe): each shortcut must equal exact enumeration or a closed form.

Run from synthetic_data/ (~30 s):  .venv/Scripts/python.exe -m unittest tests.test_truth -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import frontier_probe as fp  # noqa: E402
import transfer_pilot_v2 as tp  # noqa: E402


class TestTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(509)
        cov = 0.5 * np.ones((5, 5)) + 0.5 * np.eye(5)
        cls.x = tp.snap(rng.multivariate_normal(np.zeros(5), cov, 600))
        cls.y = np.sin(cls.x[:, 0]) + cls.x[:, 1] ** 2 + cls.x[:, 2] * cls.x[:, 3] + rng.normal(0, 0.1, 600)
        cls.x_e, cls.bg = cls.x[:20], cls.x[100:300]

    def test_exact_matches_linear_closed_form(self):
        model = LinearRegression().fit(self.x, self.y)
        closed = model.coef_ * (self.x_e - self.bg.mean(0))
        np.testing.assert_allclose(tp.exact_truth(model, self.x_e, self.bg), closed, atol=1e-10)

    def test_additive_identity_matches_exact(self):
        model = tp.additive(509).fit(self.x, self.y)
        np.testing.assert_allclose(tp.additive_truth(model, self.x_e, self.bg),
                                   tp.exact_truth(model, self.x_e, self.bg), atol=1e-8)

    def test_additive_identity_detects_non_additive_model(self):
        model = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=509).fit(self.x, self.y)
        diff = np.abs(tp.additive_truth(model, self.x_e, self.bg) - tp.exact_truth(model, self.x_e, self.bg)).max()
        self.assertGreater(diff, 1e-3, "the additive identity should not hold for a model with interactions")

    def test_tree_truth_matches_exact(self):
        model = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=509).fit(self.x, self.y)
        np.testing.assert_allclose(tp.tree_truth(model, self.x_e, self.bg),
                                   tp.exact_truth(model, self.x_e, self.bg), atol=1e-6)


class TestPairwiseTruth(unittest.TestCase):
    """frontier_probe.pairwise_truth: exact for models whose terms use at most two features, wrong otherwise."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(509)
        cov = 0.5 * np.ones((5, 5)) + 0.5 * np.eye(5)
        x = rng.multivariate_normal(np.zeros(5), cov, 300)
        cls.x_e, cls.bg = x[:10], x[100:300]

    def exact(self, f):
        return tp.exact_truth(type("F", (), {"predict": staticmethod(f)})(), self.x_e, self.bg)

    def test_matches_exact_for_order_two(self):
        f = lambda z: z[:, 0] * z[:, 1] + np.sin(z[:, 2]) + z[:, 3] * np.abs(z[:, 4]) + z[:, 1] ** 2  # noqa: E731
        np.testing.assert_allclose(fp.pairwise_truth(f, self.x_e, self.bg), self.exact(f), atol=1e-10)

    def test_detects_third_order_term(self):
        f = lambda z: z[:, 0] * z[:, 1] * z[:, 2] + z[:, 3]  # noqa: E731
        diff = np.abs(fp.pairwise_truth(f, self.x_e, self.bg) - self.exact(f)).max()
        self.assertGreater(diff, 1e-2, "a three-way interaction must break the order-2 identity")
        self.assertGreater(fp.max_third_order(f, self.x_e, self.bg, [(0, 1, 2)]), 1e-2)


if __name__ == "__main__":
    unittest.main()
