#IMPORT LIBRARIES AND RUN OLS
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import os;
path="C:/Users/michaeljgrogan/Documents/a_documents/computing/data science/datasets"
os.chdir(path)
os.getcwd()

variables = pd.read_csv('ols_stock2.csv')
stock_return = variables['stock_return']
dividend = variables['dividend']
earnings = variables['earnings']
debt_to_equity = variables['debt_to_equity']
marketcap = variables['marketcap']
stock_return_scaled = (stock_return/earnings)*1

x = np.column_stack((dividend,earnings,debt_to_equity))
x = sm.add_constant(x, prepend=True)

results = smf.OLS(stock_return,x).fit()
print(results.summary())

#BREUSCH-PAGAN TEST FOR HETEROSCEDASTICITY
import statsmodels

name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
bp = statsmodels.stats.diagnostic.het_breushpagan(results.resid, results.model.exog)
bp
pd.DataFrame(name,bp)

#SCALING
stock_return_scaled = (stock_return/earnings)*1
results = smf.OLS(stock_return_scaled,x).fit()
print(results.summary())

#MULTICOLLINEARITY TESTING WITH SKLEARN
z = np.column_stack((earnings,debt_to_equity))
z = sm.add_constant(z, prepend=True)

from sklearn.metrics import r2_score
from sklearn import linear_model
lm = linear_model.LinearRegression()
model = lm.fit(z,dividend)
predictions = lm.predict(z)
print(predictions)

rsquareddividend=r2_score(dividend, predictions)
rsquareddividend

vifdividend=1/(1-(rsquareddividend))
