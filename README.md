# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Gayathri m
RegisterNumber:  212223220024
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        #Update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("Desktop/50_Startups.csv")
data.head()
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted valeue: {pre}")

*/
```

## Output:
![362697035-763022ca-616b-4141-942a-109ee33e0446](https://github.com/user-attachments/assets/6b501d9e-dcba-4fad-b41f-135de85e337c)

![362702072-83d89cce-8fa6-491b-9c3f-6c18f2c63c3c](https://github.com/user-attachments/assets/5f88d9eb-fadf-496f-b8e2-a87b1fa7fd88)

![362702263-0920d359-e997-46a2-844c-740ef89f149a](https://github.com/user-attachments/assets/39428105-77df-4102-92ca-8f1c92d8b325)

![362703268-eb585b7f-3b99-40c0-ab3a-9c6072a7ed39](https://github.com/user-attachments/assets/56269964-ad9e-45a8-936d-0c72e987e0fd)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
