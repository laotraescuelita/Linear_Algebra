import matplotlib.pyplot as plt
import numpy as np


# importar de sklearn el modulo para crear de manera artificial matrices para clasificación.
from sklearn.datasets import make_regression
# Generar la matriz con variables numericas y el vector a predecir
A, b, coefficients = make_regression(
n_samples = 100,
n_features = 100,
n_informative = 3,
n_targets = 1,
noise = 0.2,
coef = True,
random_state = 1)


# Solve Ax=b using SVD
# Note that the book uses the Matlab-specific "regress" command
U, S, VT = np.linalg.svd(A,full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b


#Mostrar la forma que tienen las variables 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(b, label='Synthetic regression') # True relationship
axes[0].set_xlabel('X label')
axes[0].set_ylabel('Y label')
axes[0].legend()
axes[1].plot(A@x, '-o',  label='Singular value decomposition')
axes[1].set_xlabel('X label')
axes[1].set_ylabel('Y label')
axes[1].legend()
plt.show()


#Mostrar el resultado pero con datos aleatorios de la variable a predecir
sort_ind = np.argsort(b)
b = b[sort_ind] # sorted values
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(b,  label='Housing Value') # True relationship
axes[0].set_xlabel('x label')
axes[0].legend()
axes[1].plot(A[sort_ind,:]@x, '-x', label='Singular value decomposition')
axes[1].set_xlabel('x label')
axes[1].legend()

plt.show()

#Conclusion: Al ser los datos sinteticos y no tener aoutliers, ni mucho ruido la aproximación es la misma porque si hay resutado, pero 
#si el caso fuera que no hay una solución SVD nos yudaria a encontrar una aproximación.