#DEFINING VARIABLES MANUALLY

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

y = [984.5410628019, 128.8194444444, 94.495412844, 312.0331950207, 65.1127819549, 168.3289588801, 301.7441860465, 90.1408450704, 249.4573643411, 239.0361445783, 181.1775200714, 327.2440944882, 230.9523809524, 158.9442815249, 30.3759398496, 152.5783619818, 157.5938566553, 150.9933774834, 92.2413793103, 37.5706214689, 161.2958226769, 125.1546391753, 181.0394610202, 423.9678899083, 449.3975903614, 208.9949518128, 151.5695067265, 205.4315027158, 187.1388998813, 121.733615222, 155.1660516605, 196.2901896125, 211.9626168224, 120.3170028818, 104.2812254517, 245.7815565729, 133.5061088486, 135.6054191363, 102.4313372355, 148.6814566764, 125.9274357929, 166.783352781, 104.2042042042, 203.8586703859, 183.3114323259, 251.4944939696, 124.5541022592, 264.9880095923]
#stock_return_scaled

x1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
#dividend

x2 = [28, 48, 45, 43, 46, 32, 27, 42, 36, 31, 30, 26, 29, 40, 49, 38, 35, 37, 41, 47, 33, 39, 34, 12, 10, 4, 16, 15, 3, 20, 9, 2, 5, 23, 22, 11, 13, 17, 24, 14, 19, 8, 18, 7, 21, 1, 25, 6]
#earnings_ranking

x3 = [0.09, 0.12, 0.17, 0.23, 0.31, 0.32, 0.33, 0.34, 0.38, 0.42, 0.43, 0.5, 0.51, 0.52, 0.55, 0.74, 0.78, 0.79, 0.8, 0.83, 0.84, 0.93, 0.95, 1.05, 1.05, 1.09, 1.11, 1.13, 1.2, 1.21, 1.24, 1.27, 1.33, 1.35, 1.38, 1.43, 1.45, 1.45, 1.48, 1.6, 1.62, 1.7, 1.76, 1.79, 1.79, 1.79, 1.94, 2]
#debt_to_equity

x = np.column_stack((x1,x2,x3))
x = sm.add_constant(x, prepend=True)

results = smf.OLS(y,x).fit()
print(results.summary())


#READING VARIABLES FROM CSV

import csv
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

with open('C:/Users/Michael Grogan/Documents/Python/ols_stock2.csv', 'rb') as f:
reader = csv.reader(f)
y = [float(row[5]) for row in reader] #stock_return_scaled
x1 = [float(row[1]) for row in reader] #dividend
x2 = [float(row[2]) for row in reader] #earnings_ranking
x3 = [float(row[3]) for row in reader] #debt_to_equity

x = np.column_stack((x1,x2,x3))
x = sm.add_constant(x, prepend=True)

results = smf.OLS(y,x).fit()
print(results.summary())


#OUTPUT

OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.363
Model:                            OLS   Adj. R-squared:                  0.319
Method:                 Least Squares   F-statistic:                     8.342
Date:                Sun, 10 Jul 2016   Prob (F-statistic):           0.000168
Time:                        20:55:58   Log-Likelihood:                -295.45
No. Observations:                  48   AIC:                             598.9
Df Residuals:                      44   BIC:                             606.4
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const        680.7824    102.383      6.649      0.000       474.443   887.122
x1          -116.6710     76.995     -1.515      0.137      -271.844    38.502
x2           -10.5126      2.459     -4.275      0.000       -15.469    -5.556
x3          -168.7207     52.589     -3.208      0.002      -274.706   -62.735
==============================================================================
Omnibus:                       62.282   Durbin-Watson:                   1.256
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              530.769
Skew:                           3.245   Prob(JB):                    5.56e-116
Kurtosis:                      17.942   Cond. No.                         189.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
