"""Ground-truth checks (transfer pilots v2 and v3, frontier_probe): shortcuts vs exact enumeration or closed forms.

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
import transfer_pilot_v3_conditional as t3  # noqa: E402


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


class TestConditionalShap(unittest.TestCase):
    """transfer_pilot_v3_conditional: exact conditionals of mixtures and the Monte Carlo conditional SHAP."""

    def test_gaussian_linear_matches_closed_form(self):
        d, rho = 4, 0.6
        cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
        dist = t3.Mixture(np.ones(1), np.zeros((1, d)), cov[None])
        beta = np.array([1.0, -2.0, 0.5, 3.0])
        f = lambda z: z @ beta  # noqa: E731
        x_e = np.random.default_rng(509).multivariate_normal(np.zeros(d), cov, 5)

        def v(mask, x):   # E[beta.X | X_S = x_S] for a zero-mean Gaussian
            s = [j for j in range(d) if mask >> j & 1]
            r = [j for j in range(d) if not mask >> j & 1]
            if not s:
                return 0.0
            if not r:
                return x @ beta
            cond_r = cov[np.ix_(r, s)] @ np.linalg.solve(cov[np.ix_(s, s)], x[s])
            return x[s] @ beta[s] + cond_r @ beta[r]
        w = [1 / 4, 1 / 12, 1 / 12, 1 / 4]   # |S|! (d - |S| - 1)! / d! for d = 4
        exact = np.zeros((len(x_e), d))
        for n, x in enumerate(x_e):
            for mask in range(1 << d):
                for i in range(d):
                    if not mask >> i & 1:
                        exact[n, i] += w[bin(mask).count("1")] * (v(mask | 1 << i, x) - v(mask, x))
        est = t3.conditional_shap(f, x_e, dist, 20000, np.random.default_rng(1))
        np.testing.assert_allclose(est, exact, atol=0.08)
        np.testing.assert_allclose(est.sum(1), f(x_e) - 0.0, atol=0.05)   # efficiency: phi sums to f(x) - E f

    def test_mixture_conditional_mean(self):
        means = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 1.0]])
        covs = np.array([np.eye(3) + 0.5 * (np.ones((3, 3)) - np.eye(3)), np.diag([0.5, 1.0, 2.0])])
        dist = t3.Mixture(np.array([0.3, 0.7]), means, covs)
        x0 = np.array([[1.0]])
        draws = dist.cond_sample((0,), x0, 50000, np.random.default_rng(509))[0][:, 1:]
        dens = [w * np.exp(-0.5 * (1.0 - mu[0]) ** 2 / c[0, 0]) / np.sqrt(c[0, 0])
                for w, mu, c in zip(dist.w, means, covs)]
        post = np.array(dens) / np.sum(dens)
        cond = [mu[1:] + c[1:, 0] / c[0, 0] * (1.0 - mu[0]) for mu, c in zip(means, covs)]
        np.testing.assert_allclose(draws.mean(0), post @ np.array(cond), atol=0.03)

    def test_independent_world_gives_additive_identity(self):
        rng = np.random.default_rng(509)
        pool = rng.multivariate_normal(np.zeros(3), 0.8 * np.ones((3, 3)) + 0.2 * np.eye(3), 2000)
        f = lambda z: np.sin(z[:, 0]) + z[:, 1] ** 2 + 0.5 * z[:, 2]  # noqa: E731
        x_e = pool[:5]
        # under independence, conditional = interventional, and for an additive f phi_j = f_j(x_j) - E f_j
        exact = np.column_stack([np.sin(x_e[:, 0]) - np.sin(pool[:, 0]).mean(), x_e[:, 1] ** 2 - (pool[:, 1] ** 2).mean(),
                                 0.5 * (x_e[:, 2] - pool[:, 2].mean())])
        est = t3.conditional_shap(f, x_e, t3.Independent(pool), 20000, np.random.default_rng(1))
        np.testing.assert_allclose(est, exact, atol=0.05)

    def _skewed_pool(self):
        rng = np.random.default_rng(509)
        z = rng.multivariate_normal(np.zeros(3), [[1, 0.7, 0.3], [0.7, 1, 0.5], [0.3, 0.5, 1]], 5000)
        return np.column_stack([np.exp(z[:, 0]), z[:, 1] ** 3, np.round(4 * z[:, 2])])   # skewed, heavy, discrete

    def test_copula_reproduces_marginals(self):
        pool = self._skewed_pool()
        cop = t3.GaussianCopula(pool)
        draws = cop.sample(20000, np.random.default_rng(1))
        for j in range(3):   # KS distance to the (dequantized) marginal the copula defines
            grid = cop.sorted[::50, j]
            ecdf_draws = np.searchsorted(np.sort(draws[:, j]), grid, side="right") / len(draws)
            ecdf_pool = np.searchsorted(cop.sorted[:, j], grid, side="right") / len(cop.sorted)
            self.assertLess(np.abs(ecdf_draws - ecdf_pool).max(), 0.02)
        self.assertLess(np.abs(np.sort(pool[:, 2]) - cop.sorted[:, 2]).max(), 0.5 + 1e-9,
                        "dequantization must stay within half the gap between distinct values")

    def test_copula_conditional_is_latent_gaussian_conditional(self):
        pool = self._skewed_pool()
        cop = t3.GaussianCopula(pool)
        x_s = pool[:1, :1]                                                    # condition on feature 0
        draws = cop.cond_sample((0,), x_s, 50000, np.random.default_rng(1))[0]
        z_s = cop._to_z(x_s, (0,))[0, 0]
        z_r = np.column_stack([cop._to_z(draws[:, [j]], (j,))[:, 0] for j in (1, 2)])
        r = cop.latent.cov[0]
        np.testing.assert_allclose(z_r.mean(0), r[1:, 0] * z_s, atol=0.03)


if __name__ == "__main__":
    unittest.main()
