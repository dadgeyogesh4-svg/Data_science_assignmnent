#LINEAR_REGRESSION

# Import necessary libraries
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
df
df.head()

X = df[['YearsExperience']]
y = df['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn import metrics
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))

# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Years of Experience')
plt.ylabel('Salary Hike')
plt.title('Salary Hike Prediction')
plt.show()

#==============================================================================
import pandas as pd

df = pd.read_csv('delivery_time.csv')
df.head()

X = df[['Sorting Time']]
y = df['Delivery Time']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn import metrics
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred, squared=False))


import matplotlib.pyplot as plt
plt.scatter(df[['Sorting Time']],df['Delivery Time'],color='red')
plt.show()


