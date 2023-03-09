# Project 3: Gradient Boosting

Gradient boosting can be used to make a more robust learner and improve predictions. With gradient boosting, a second model is trained on the residuals of the first model, improving the chances that the second model is better than the first. 

In this project, I present code that builds off of the functions for Alex Gramfort's approach to Lowess from before.

### Code Implementation

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

### Boosted Regressor Function

This new function defines the boosted regressor. It allows the user to choose between three different regressors for the second model: Lowess (default), Random Forest Regressor, and Decision Tree Regressor. The user can specify values for the number of estimators and max depth for Random Forest, and the max depth for Decision Tree. This is also where the user can change which kind of kernel is used by the Lowess regression function (the default is Epanechnikov).

```Python
def boosted_lwr(x, y, xnew, mod2 = 'Lowess', f=1/3, iter=2, n_estimators=200, max_depth=5, intercept=True, kernel=Epanechnikov):

  model1 = Lowess(f=f,iter=iter,intercept=intercept,kernel=kernel)

  if mod2 == 'RandomForestRegressor':
    model2 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
  elif mod2 == 'DecisionTreeRegressor':
    model2 = DecisionTreeRegressor(max_depth=max_depth)
  else:
    model2 = Lowess(f=f,iter=iter,intercept=intercept,kernel=kernel)

  # for training the boosted method we use x and y
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)

  model2.fit(x,residuals1)

  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
```

## Testing on Real Datasets

The next step is to test the gradient boosting regressor on some real data. Below I demonstrate a few of the customizations available to the user below, seeing how some choices may impact the mean squared error. We start with a dataset about cars.

```Python
car_data = pd.read_csv('drive/MyDrive/DATA441/data/cars.csv')
x = car_data.loc[:,'CYL':'WGT'].values
y = car_data['MPG'].values
scale = StandardScaler()
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)
```
The next four code blocks can be run sequentially to get an idea about what kernel might be the best choice for this regression. Note that in each case, Lowess is being used for both models with constant f and iter values.

Epanechnikov:
```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, f=1/3, iter=3, intercept=True)
mse(ytest,yhat)
```

Tricubic:
```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, f=1/3, iter=3, intercept=True, kernel=Tricubic)
mse(ytest,yhat)
```

Gaussian:
```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, f=1/3, iter=3, intercept=True, kernel=Gaussian)
mse(ytest,yhat)
```

Quartic:
```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, f=1/3, iter=3, intercept=True, kernel=Quartic)
mse(ytest,yhat)
```

After running these cells, I find that the Tricubic kernel has the lowest mse of 16.3085, narrowly beating the Quartic kernel which had an mse of 16.3261.

Below are two cell blocks that change the second model in the boosted regression. These produce mean squared errors that are comparable to the Tricubic kernel above, but neither is actually lower.

```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, mod2='RandomForestRegressor', max_depth=3, n_estimators=200, f=1/3, iter=3, intercept=True, kernel=Tricubic)
mse(ytest,yhat)
```
MSE of 17.4098

```Python
yhat = boosted_lwr(xtrain, ytrain, xtest, mod2='DecisionTreeRegressor', max_depth=1, f=1/3, iter=3, intercept=True, kernel=Tricubic)
mse(ytest,yhat)
```
MSE of 16.6134

Even with just a few customization options, the number of possible combinations is massive. It's fun and fascinating to play around with the options and see the resulting mean squared error.

Next, we move onto a concrete dataset. After playing around with different options for regressors and kernels, the lowest mse I could achieve was with a Quartic kernel and a Random Forest Regressor as the second model.

```Python
concrete_data = pd.read_csv('drive/MyDrive/DATA441/data/concrete.csv')
x = concrete_data.loc[:,'cement':'age'].values
y = concrete_data['strength'].values
scale = StandardScaler()
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)
yhat = boosted_lwr(xtrain, ytrain, xtest, mod2='RandomForestRegressor', f=25/len(xtrain), iter=1, intercept=True, kernel=Quartic)
mse(ytest,yhat)
```

This model produces an mse of 44.0735. The next step is to compare this method of gradient boosting with a different regressor, such as just Random Forest.

## Complete KFold Crossvalidation

To be confident that the gradient boosting method is both effective on its own and a more optimal choice than a lone regressor, a complete KFold crossvalidation comparing regressors can be implemented. The code below compares the gradient boosting regressor with concrete data from above with Random Forest.

```Python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_lw = boosted_lwr(xtrain, ytrain, xtest, mod2='RandomForestRegressor', f=25/len(xtrain), iter=1, intercept=True, kernel=Quartic)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Gradient Boosting Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```
When I run the code, I get this output: \


```Python

```

```Python

```
