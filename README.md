# Project 3: Gradient Boosting

Gradient boosting can be used to make a more robust learner and improve predictions. With gradient boosting, a second model is trained on the residuals of the first model, improving the chances that the second model is better than the first. 

In this project, I present code that builds off of the functions for Alex Gramfort's approach to Lowess from before.

Import Statements:

```Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as stats 
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```

The Gramfort Lowess function accepts a choice of kernel as one of its arguments. The four options are defined below:

```Python
# Gaussian Kernel
def Gaussian(w):
  return np.where(w>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*w**2))

# Tricubic Kernel
def Tricubic(w):
  return np.where(w>1,0,70/81*(1-w**3)**3)

# Quartic Kernel
def Quartic(w):
  return np.where(w>1,0,15/16*(1-w**2)**2)

# Epanechnikov Kernel
def Epanechnikov(w):
  return np.where(w>1,0,3/4*(1-w**2)) 
```

Next come the definitions for a distance function, the lowess regressor, and then a class definition for lowess that allows for ScikitLearn compatibility.

```Python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1)
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d
```

```Python
def lowess(x, y, xnew, f = 1/10, iter = 3, intercept=True, kernel=Epanechnikov):

  n = len(x)
  r = int(ceil(f * n))
  yest = np.zeros(n)

  if len(y.shape)==1: # here we make column vectors
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x

  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
  # dist(x,x) is always symmetric
  w = np.clip(dist(x,x) / np.array(h), 0.0, 1.0)
  w = kernel(w)

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(delta).dot(np.diag(w[i,:]))
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)

      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization for solving the system
      beta = linalg.solve(A, b)

      yest[i] = np.dot(x1[i],beta.ravel())

    residuals = y.ravel() - yest
    s = np.median(np.abs(residuals))

    delta = np.clip(residuals / (6.0 * s), -1, 1)

    delta = (1 - delta ** 2) ** 2
    
  # here we are making predictions for xnew by using an interpolation and the predictions we made for the train data
  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ Pp')
      f = LinearNDInterpolator(tri,yest[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) 
      # the output may have NaN's where the data points from xnew are outside the convex hull of X

  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,yest.ravel()) 
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output
```

```Python
class Lowess:
    def __init__(self, f = 1/10, iter = 3, intercept=True, kernel=Epanechnikov):
        self.f = f
        self.iter = iter
        self.intercept = intercept
        self.kernel = kernel
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        kernel = self.kernel
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        kernel = self.kernel
        return lowess(x, y, x_new, f, iter, intercept, kernel) # this is our defined function of Lowess

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter, "intercept": self.intercept, "kernel": self.kernel}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
```

```Python

```

```Python

```

```Python

```
