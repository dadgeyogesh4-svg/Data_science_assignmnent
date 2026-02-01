# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:40:19 2023


"""
#pip install --upgrade openpyxl==3.0.10

#airlines

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


file_path = 'Airlines+Data.xlsx'
file_path

file_path = 'Airlines+Data.xlsx'


df = pd.read_excel(file_path)


print(df.head())


encoder = OneHotEncoder(sparse=False, drop='first')
dummy_variables = pd.DataFrame(encoder.fit_transform(df[['Month']]))
df = pd.concat([df, dummy_variables], axis=1)


X = df.drop(['Passengers', 'Month'], axis=1)
y = df['Passengers']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
rmse1 = mse1 ** 0.5


model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)
rmse2 = mse2 ** 0.5


model3 = GradientBoostingRegressor()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
mse3 = mean_squared_error(y_test, y_pred3)
rmse3 = mse3 ** 0.5


print(f"Model 1 (Linear Regression) RMSE: {rmse1:.2f}")
print(f"Model 2 (Random Forest Regression) RMSE: {rmse2:.2f}")
print(f"Model 3 (Gradient Boosting Regression) RMSE: {rmse3:.2f}")


best_model = min((rmse1, 'Linear Regression'), (rmse2, 'Random Forest Regression'), (rmse3, 'Gradient Boosting Regression'))
print(f"\nSelected Model for Forecasting: {best_model[1]} with RMSE: {best_model[0]:.2f}")




===============================================================================
#Cocola

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

file_path = 'CocaCola_Sales_Rawdata.xlsx'


df = pd.read_excel(file_path)


print(df.head())



encoder = OneHotEncoder(sparse=False, drop='first')
product_type_dummy = pd.DataFrame(encoder.fit_transform(df[['Sales']]))
quarter_dummy = pd.DataFrame(encoder.fit_transform(df[['Quarter']]))




df = pd.concat([df, product_type_dummy, quarter_dummy], axis=1)




df = df.drop(['product_type', 'quarter'], axis=1)




X = df.drop('Quarter', axis=1)
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model1 = LinearRegression()
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred1))
print(f"Model 1 RMSE: {rmse1:.2f}")

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
print(f"Model 2 RMSE: {rmse2:.2f}")

if rmse1 < rmse2:
    selected_model = "Linear Regression"
else:
    selected_model = "Random Forest Regression"

print(f"\nSelected Model for Forecasting: {selected_model}")









