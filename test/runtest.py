import unittest
import sys
import math
import time
import numpy as np
from scipy import stats
import warnings
#
# Linear Model
#
# Current version
sys.path.append('..')
import linear_model as lm
# Old version
import linear_model_v_1_1_0 as lm_1_1_0
#
# OLS
#
# WARN: All the OLS packages in python (statsmodels, sklearn) are
# computationally inefficient on large datasets (N = 200'000, D = 1500)
#
from statsmodels.api import OLS
import sklearn.linear_model as skl




##
## Generative models
##

def causal_model_1(N = 5000, do_x = None):
    """
    Args:
      N (|int|): Number of data points
      do_x (|float|): The value of intervention
    #
    Model 1 from
    A Crash Course in Good and Bad Controls
    Carlos Cinelli, Andrew Forney, Judea Pearl
    """
    Z = np.random.uniform(-3,3, size=N)
    if do_x is None:
        X = 1 + 3*Z + 2*Z**3 + np.random.normal(size=N,scale=1)
    else:
        X = np.full(N, do_x)
        Y = -1 - 2*X + 6*Z**2 + np.random.normal(size=N,scale=1)
    return Z, X, Y


def causal_model_4(N = 5000, do_x = None):
    """
    Args:
      N (|int|): Number of data points
      do_x (|float|): The value of intervention
    #
    Model 1 from
    A Crash Course in Good and Bad Controls
    Carlos Cinelli, Andrew Forney, Judea Pearl
    """
    Z = np.random.normal(size=N, scale=1)
    if do_x is None:
        X = Z**2 + np.random.normal(size=N, scale=1)
    else:
        X = np.full(N, do_x)
    M = 2*Z**2 + 10*X + np.random.normal(size=N, scale=1)
    Y = -1 + 2*M**2 + np.random.normal(size=N, scale=1)
    return Z, X, M, Y

def random_model_0(N, D, seed = 42):
    """
    Args:
    N (|int|): Number of data points
    D (|int|): Number of dimensions
      seed (|int|, *optional*): seed for pseudo-random number
    This function it can be used to create large datasets
        N = 200000
        D = 1500
    or to create datasets with more dimensions than data points
        N = 500
        D = 1500
    """
    np.random.seed(seed)
    X = np.random.randn(N, D)
    true_weights = np.random.randn(D)
    noise = np.random.randn(N) * 0.5
    Y = X.dot(true_weights) + noise
    return X, true_weights, Y

##
## Testing functions
##

def estimates_inside_credible_interval(
        true_weights,
        m_N,                # mean estimates
        S_N,                # dispersion estimates
        conf_level = 0.95,  # credible interval
    ):
    #
    z = stats.norm.ppf((1 + conf_level) / 2)
    # Credible intervals for each parameter
    lower_bounds = m_N - z * np.sqrt(np.diag(S_N))
    upper_bounds = m_N + z * np.sqrt(np.diag(S_N))
    # Check if them are inside the interval
    in_interval = np.logical_and(
        true_weights >= lower_bounds,
        true_weights <= upper_bounds
    )
    # Aggregated output: percentage.
    coverage_pct = 100 * np.mean(in_interval)
    expected_pct = 100 * conf_level
    return coverage_pct, expected_pct


def evidence_old_package(lm_1_1_0_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return lm_1_1_0_model.evidence()

#
# UNIT Test
#

class tests(unittest.TestCase):
    def test_small_data_set(self):
        # Data and true_weights
        N = 5000; D = 10
        X, true_weights, y = random_model_0(N=N, D =D, seed=42)

        # Create models
        lm_model = lm.BayesianLinearModel(basis=lambda x: x)
        lm_1_1_0_model = lm_1_1_0.BayesianLinearModel(basis=lambda x: x)

        # Update models
        lm_model.update(X, y.reshape(N,1))
        lm_1_1_0_model.update(X, y.reshape(N,1))

        # Outputs of models
        # Location
        lm_location = lm_model.location.reshape(1,D)
        lm_1_1_0_location = lm_1_1_0_model.location.reshape(1,D)
        #
        lm_dispersion = lm_model.dispersion
        lm_1_1_0_dispersion = lm_1_1_0_model.dispersion
        # Geometric mean of the evidence
        lm_gm = math.exp(lm_model.evidence()/N)
        lm_1_1_0_gm = math.exp(lm_1_1_0_model.evidence()/N)

        # Difference between models
        self.assertTrue(
        np.all(np.isclose(lm_location, lm_1_1_0_location))
        )
        self.assertTrue(
        np.all(np.isclose(lm_dispersion, lm_1_1_0_dispersion))
        )
        self.assertTrue(
        lm_gm>=lm_1_1_0_gm
        )
        self.assertTrue(
        np.isclose(lm_gm,lm_1_1_0_gm, rtol=1e-2)
        )

        # Difference between estimates and true_weights
        coverage_pct, expected_pct = estimates_inside_credible_interval(
            true_weights,
            lm_location,
            lm_dispersion)
        self.assertTrue(
        coverage_pct >= expected_pct
        )

    def test_large_data_set(self):
        N = 50000; D = 100
        X, true_weights, y = random_model_0(N=N, D =D, seed=42)

        # Create and Update models
        #
        # Current version
        start = time.time()
        lm_model = lm.BayesianLinearModel(basis=lambda x: x)
        lm_model.update(X, y.reshape(N,1))
        lm_elapsed = time.time() -start
        #
        # Old version
        lm_1_1_0_model = lm_1_1_0.BayesianLinearModel(basis=lambda x: x)
        lm_1_1_0_model.update(X, y.reshape(N,1))

        # Outputs of models
        # Location
        lm_location = lm_model.location.reshape(1,D)
        lm_1_1_0_location = lm_1_1_0_model.location.reshape(1,D)
        #
        lm_dispersion = lm_model.dispersion
        lm_1_1_0_dispersion = lm_1_1_0_model.dispersion
        # Geometric mean of the evidence
        lm_gm = math.exp(lm_model.evidence()/N)
        lm_1_1_0_gm = math.exp(
            evidence_old_package(lm_1_1_0_model)/N)

        # Difference between models
        self.assertTrue(
        np.all(np.isclose(lm_location, lm_1_1_0_location))
        )
        self.assertTrue(
        np.all(np.isclose(lm_dispersion, lm_1_1_0_dispersion))
        )
        self.assertTrue(
        lm_gm>=lm_1_1_0_gm
        )
        self.assertTrue(
        lm_1_1_0_gm==0  # Numerical instability in the original package
        )

        # Difference between estimates and true_weights
        coverage_pct, expected_pct = estimates_inside_credible_interval(
            true_weights,
            lm_location,
            lm_dispersion)
        self.assertTrue(
        coverage_pct >= expected_pct
        )

        # OLS
        #
        # statsmodels
        start = time.time()
        ols_model = OLS(y.reshape(N,1), X.reshape(N,D)).fit()
        ols_statmodels_elapsed = time.time() - start
        #
        # sklearn
        start = time.time()
        skl_model = skl.LinearRegression().fit(X,y)
        ols_sklearn_elapsed = time.time() - start

        print(f"""
        Our model is {ols_statmodels_elapsed/lm_elapsed}X faster than OLS statsmodels in a (not so) 'large' dataset (N=50000, D=100)
        """)
        self.assertTrue(
        ols_statmodels_elapsed/lm_elapsed > 34/3
        )

        print(f"""
        Our model is {ols_sklearn_elapsed/lm_elapsed}X faster than OLS sklearn in a (not so) 'large' dataset (N=50000, D=100)
        """)
        self.assertTrue(
        ols_sklearn_elapsed/lm_elapsed > 9.8/3
        )

    def test_LARGE_data_set(self):
        N = 200000; D = 1500
        X, true_weights, y = random_model_0(N=N, D =D, seed=42)

        # Create and Update
        lm_model = lm.BayesianLinearModel(basis=lambda x: x)
        lm_model.update(X, y.reshape(N,1))

        # Output
        lm_location = lm_model.location.reshape(1,D)
        lm_dispersion = lm_model.dispersion
        lm_gm = math.exp(lm_model.evidence()/N)

        # Difference between estimates and true_weights
        coverage_pct, expected_pct = estimates_inside_credible_interval(
            true_weights,
            lm_location,
            lm_dispersion)
        self.assertTrue(
        coverage_pct >= expected_pct
        )


if __name__ == "__main__":
    unittest.main()


