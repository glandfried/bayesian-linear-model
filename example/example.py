import numpy as np
import matplotlib.pyplot as plt
from linear_model import BayesianLinearModel
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(42)

# Create polynomial features in basis function expansion.
polybasis = lambda x, p: PolynomialFeatures(p).fit_transform(x)

# Create random sin() data.
N = 50
noise = 0.25
X = np.sort(np.random.uniform(0, 2*np.pi, N)).reshape((N, 1))
y = np.sin(X) + np.random.normal(scale=noise, size=(N, 1))

# Calculate log marginal likelihood (model evidence) for each model.
lml = list()
for d in range(10):
    blm = BayesianLinearModel(basis=lambda x: polybasis(x, d))
    blm.update(X, y)
    lml.append(blm.evidence())

# Perform model selection by choosing the model with the best fit.
D = np.argmax(lml)
blm = BayesianLinearModel(basis=lambda x: polybasis(x, D))
blm.update(X, y)

# Perform inference in the model.
x_query = np.linspace(0, 2*np.pi, 1000)[:, None]
mu, S2 = blm.predict(x_query, variance=True)

# Plot model selection.
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
f.subplots_adjust(hspace=0.5)
ax1.plot(range(len(lml)), lml)
ax1.set_title('Model selection')
ax1.set_xlabel('number of polynomial features')
ax1.set_ylabel('Log marginal likelihood')
ax1.axvline(D, color='r', linewidth='3')
ax1.grid('on')

# Plot model predictions.
ax2.plot(X, y, 'k.')
ax2.plot(x_query, mu + S2, 'r--', linewidth=2)
ax2.plot(x_query, mu, 'r', linewidth=3)
ax2.plot(x_query, mu - S2, 'r--', linewidth=2)
ax2.set_title('Prediction')
ax2.set_xlabel('input domain (x)')
ax2.set_ylabel('output domain f(x)')
ax2.grid('on')
plt.show()