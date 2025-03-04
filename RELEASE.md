# RELEASE NOTES - Bayesian Linear Model Library

## Version 2.0.0 (2025)

We are pleased to announce a major update to the Bayesian Linear Model library, featuring significant improvements in numerical stability, computational efficiency, and new functionality. This release represents a collaboration between Asher Bender and Gustavo Landfried.

### Major Improvements

#### 1. Numerical Stability Enhancements

* **Safe Linear Algebra Operations**: Added `safe_logdet()` and `safe_solve()` functions to handle ill-conditioned matrices gracefully
* **Robust Cholesky Decomposition**: Replaced direct Cholesky decomposition with more stable solvers that handle near-singular matrices
* **Fallback Methods**: Implemented progressive fallback mechanisms when matrix operations fail
* **Early Detection**: Added condition number checks to identify potential numerical issues before they cause failures
* **Warning System**: Integrated a comprehensive warning system to alert users about potential numerical issues

#### 2. Regularization Framework

* **Regularization Parameter**: Added a `reg` parameter to control regularization strength for better numeric stability
* **Weakly Informative Priors**: Changed uninformative priors to weakly informative priors for more reliable Bayesian inference
* **Parameter Bounds**: Enforced positive constraints on scale and shape parameters (beta and alpha) to prevent invalid posterior distributions

#### 3. Error Handling and Diagnostics

* **Enhanced Error Messages**: Improved error messages with more context about what went wrong
* **Exception Handling**: Added comprehensive exception handling throughout the codebase
* **Diagnostic Methods**: Added a new `get_diagnostics()` method to provide insight into model state and potential issues

#### 4. Bayesian Inference Improvements

* **Sampling Methods**: Improved random sampling from posterior with more robust handling of edge cases
* **Evidence Calculation**: Enhanced model evidence calculation with fallback methods for numerical stability
* **Confidence Intervals**: Added confidence interval level parameter to `predict()` method

#### 5. Optimization Improvements

* **Multiple Optimization Methods**: Added support for different optimization methods in `empirical_bayes()`
* **Optimization Options**: Exposed optimizer options to users for fine-tuning
* **Bounded Optimization**: Added bounded optimization by default for more stable solutions

### API Changes

* **New Parameters**:
  * Added `reg` parameter to `__init__` for controlling regularization strength
  * Added `ci_level` parameter to `predict()` for controlling confidence interval width
  * Added `method` and `options` parameters to `empirical_bayes()` for optimization control
  * Added `robust` and `fallback` parameters to evidence calculation methods

* **New Methods**:
  * Added `get_diagnostics()` for examining model internals and numerical stability

* **Modified Default Behavior**:
  * Changed uninformative priors to weakly informative priors
  * Modified initialization to include small regularization by default
  * Changed evidence calculation to use more stable algorithms by default

### Technical Details

#### Numerical Stability Improvements

The primary focus of this release was improving numerical stability. Bayesian linear regression can be prone to numerical issues when:

1. The covariance/precision matrices are ill-conditioned
2. There are collinearities in the input data
3. The number of features approaches or exceeds the number of samples
4. The scale parameters approach zero

We've addressed these issues by:

* Using more stable linear algebra operations with graceful fallbacks
* Adding regularization to prevent perfectly singular matrices
* Implementing checks on parameter values to ensure they remain in valid ranges
* Using pseudo-inverse calculations when appropriate
* Carefully handling eigenvalues and determinants to avoid numeric overflow/underflow

#### Prior Distribution Changes

We've changed the default behavior from using completely uninformative priors to using weakly informative priors, following best practices in the Bayesian statistics literature. Specifically:

* Changed prior shape parameter from `-D/2` to `1.0` to ensure the posterior distribution has finite mean
* Added a small regularization to the precision matrix instead of using zeros
* Enforced constraints to ensure valid Inverse-Gamma parameters

#### Evidence Calculation

The model evidence (marginal likelihood) calculation has been completely rewritten to be more numerically stable, with multiple fallback methods if primary calculations fail. This provides more reliable model comparison and hyperparameter selection.

### Compatibility Notes

* While the core API remains compatible with the previous version, the default behavior has changed to use weakly informative priors instead of uninformative priors.
* Results with the same data and settings may differ slightly due to enhanced numerical stability techniques.
* Custom basis functions should be tested for compatibility with the new version.

### Performance Improvements

* More efficient handling of ill-conditioned matrices
* Better fallback strategies for numerical edge cases
* Improved optimization in empirical Bayes

### References

The Bayesian linear model module was created and improved using the following references:

1. Murphy, K. P., Machine learning: A probabilistic perspective, The MIT Press, 2012
2. Bishop, C. M, Pattern Recognition and Machine Learning (Information Science and Statistics), Jordan, M.; Kleinberg, J. & Scholkopf, B. (Eds.), Springer, 2006
3. Murphy, K. P., Conjugate Bayesian analysis of the Gaussian distribution, Department of Computer Science, The University of British Columbia, 2007

### Acknowledgments


Special thanks to Asher Bender for providing the first version (1.0.0). In this new version (2.0.0) Gustavo Landfried improves the numerical stability.

---

## Future Development

We plan to continue improving the library with:

* Better documentation and examples of commonly used basis functions
* Support for automatic basis function selection
* GPU acceleration for large-scale problems
* Enhanced visualization tools for posterior analysis
