"""Ground-truth checks for transfer_pilot_v2: each shortcut must equal exact enumeration or a closed form.

Run from synthetic_data/ (~30 s):  .venv/Scripts/python.exe -m unittest tests.test_truth -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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


if __name__ == "__main__":
    unittest.main()
