# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start.

2.Data Preparation.

3.Hypothesis Definition.

4.Cost Function.

5.Parameter Update Rule.

6.Iterative Training.

7.Model Evaluation.

8.End. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

model.fit(X, y)
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.show()

Developed by: Varsha M
RegisterNumber:  25001006
*/
```

## Output:
<img width="1015" height="775" alt="image" src="https://github.com/user-attachments/assets/c092aeca-e7c2-4b29-b60b-125503d27e27" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
