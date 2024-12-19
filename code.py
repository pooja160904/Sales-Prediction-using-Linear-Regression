import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

advertising = pd.DataFrame(pd.read_csv("advertising.csv"))
advertising.head()
advertising.shape
advertising.info()
advertising.describe()
# Checking Null values
advertising.isnull().sum()  /advertising.shape[0]
# There are no NULL values in the dataset, hence it is clean.
# Outlier Analysis - vo points jo dataset se alag jare ho
fig, axs = plt.subplots(3, figsize = (6,6))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()
sns.boxplot(advertising['Sales'])
plt.show()
# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()
# Let's see the correlation between different variables.
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()
X = advertising['TV']
y = advertising['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 70)
X_train.head()
y_train.head()

# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the regression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()
# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params
# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())
plt.scatter(X_train, y_train)
plt.plot(X_train, 6.8974 + 0.0554*X_train, 'r')
plt.show()
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)
fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()
plt.scatter(X_train,res)
plt.show()
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)
y_pred.head()
#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
r_squared
plt.scatter(X_test, y_test)
plt.plot(X_test, 6.8974 + 0.0554 * X_test, 'r')
plt.show()



