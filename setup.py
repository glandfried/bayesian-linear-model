from setuptools import setup

setup(
    name='linear_model',
    version='2.0.0',
    author='Asher Bender, Gustavo Landfried',
    author_email='a.bender.dev@gmail.com, gustavolandfried@gmail.com',
    description=('Implementation of the Bayesian linear model.'),
    py_modules=['linear_model'],
    install_requires=[
        'numpy',
        'scipy',
        'Sphinx',
    ]
)
