# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:16:27 2022

@author: ankus
"""

#------- 1. Import libarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------2.  Read or import data
dataset=pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\Naresh IT\15_April_Own\Position_Salaries.csv')

# S------------3. plit data into X and Y , Attribute postion not rqeuired
X=dataset.iloc[:,1:2] # only level col
y=dataset.iloc[:,2]


#---------4. First check with linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

           #- Plot for Simplelinear Regression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')  # plot this line for X  and predicted for X test
plt.title('Truth or Bluff Linear Regression')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()
#***************************************************************************
 
#----------5 . Fitting Polynomial Regression with given data set
from sklearn.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures(degree=5, interaction_only=False, include_bias=True, order='C') # degree =0,1,2,3,4,5..
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

         #---. Fit the linear regression also
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

             #----plot the visualization for polynomial Regtression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff Ploynomial Regression')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()

#---------- Predict the result
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
































