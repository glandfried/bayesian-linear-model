REPO_DIR = "asherbender-bayesian-linear-model"
import sys
import os
from pathlib import Path
# Obtener directorio actual y añadir el directorio padre al path
splited_dir = str(Path(os.getcwd())).split('/')
uppder_dir = ""; found_upper_dir = False; i = 0
while i < len(splited_dir) and not found_upper_dir:
    uppder_dir += splited_dir[i]+"/"
    if splited_dir[i] == REPO_DIR:
        found_upper_dir = True
    i = i + 1

sys.path.append(uppder_dir)
datos_path = uppder_dir+"datos/"
##########################################
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import software.ModeloLineal.Full as mlf
import software.ModeloLineal.Exacta2 as ml
# Para evaluación de hipótesis y modelos (archivo ModeloLineal.py)
#import ModeloLineal as ml
import software.ModeloLineal.asherbender_claude as ash
import math

# OLS
#
# Todos las regresiones lineales "ordinary" que encontremos
# en python (statsmodels, sklearn) son extremadamente lentas
# en base de datos moderadamente grandes (N = 200'000, D = 1500)
#
#from statsmodels.api import OLS
#import sklearn.linear_model as skl
#
# No parece haber más opciones. Habría que implementar una
# versión propia mediante linalg.lstsq de numpy o scipy




np.random.seed(42)
N = 200000  # Número de observaciones
D = 1500    # Número de características
# Generar matriz X con datos aleatorios
X = np.random.randn(N, D)
# Generar pesos aleatorios "verdaderos" para el modelo
true_weights = np.random.randn(D)
# Generar objetivo y con algo de ruido
noise = np.random.randn(N) * 0.5
y = X.dot(true_weights) + noise
#X = pd.DataFrame(X)

ash_model = ash.BayesianLinearModel(basis=lambda x: x)
#prior_precision = 0.001 * np.eye(D)
#ash_model = ash.BayesianLinearModel(
    #basis=lambda x: x,
    #dispersion=prior_precision,  # Matriz de precisión prior
    #shape=2.0,                   # Parámetro de forma para la distribución inversa gamma
    #scale=1.0                    # Parámetro de escala para la distribución inversa gamma
#)
ash_model.update(X, y.reshape(N,1))
np.sum((true_weights - ash_model.location.reshape(1,D))**2)
math.exp(ash_model.evidence()/N)

#skl_model = skl.LinearRegression()
#skl_model.fit(X,y)

#ols_model = OLS(y.reshape(N,1), X.reshape(N,D)).fit()
#ols_model.params
#np.sum((true_weights - ols_model.params)**2)


m_N = ash_model.location.reshape(1,D)
S_N = ash_model.dispersion
conf_level = 0.99
    # Calcular el valor z para el nivel de confianza
z = stats.norm.ppf((1 + conf_level) / 2)
# Calcular los intervalos de credibilidad para cada parámetro
lower_bounds = m_N - z * np.sqrt(np.diag(S_N))
upper_bounds = m_N + z * np.sqrt(np.diag(S_N))
in_interval = np.logical_and(
    true_weights >= lower_bounds,
    true_weights <= upper_bounds
)
# Calcular el porcentaje de cobertura
coverage_pct = 100 * np.mean(in_interval)
expected_pct = 100 * conf_level
print(coverage_pct,expected_pct )

