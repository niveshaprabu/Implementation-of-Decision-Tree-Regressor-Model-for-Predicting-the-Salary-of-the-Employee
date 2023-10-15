# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
# AIM:

To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
# Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Nivesha P
RegisterNumber:212222040108

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
  
*/
```

# Output:
# data.head()
![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/bb78ec1a-3ac1-42e7-81cb-c6e1dfcd20d7)


# data.info()
![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/3dcc56e4-7a08-45cf-bff7-0832b627edc0)


# isnull() and sum()
![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/7869d0f8-1e73-4576-a758-2465e7fc93f7)

# data.head() for salary

![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/d479393e-f882-46ef-a35d-ad0c2d779a91)

# MSE value

![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/610e0f3f-b80b-424c-ab43-b048b48f3c94)

# r2 value

![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/e68d2798-17ac-4cad-a649-fb201d588d78)


# data prediction


![image](https://github.com/niveshaprabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/122986499/cd233b18-8866-419d-8c2a-227cc80d5667)

# Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
