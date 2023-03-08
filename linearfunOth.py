import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

points = np.array([[1, 2], [2, 5], [3, 2], [4, 8], [5, 4]])

# Fit the linear regression model
model = LinearRegression().fit(points[:, 0].reshape(-1, 1), points[:, 1])

# Extract the slope and y-intercept of the least squares line
m = model.coef_[0]
b = model.intercept_

# Output the least squares line in the form y = mx + b
print(f"y = {m} x + {b}")


plt.plot(points[:,0],points[:,1],'ro')

x = np.linspace(0, 10, 100)
a =m
b = b
plt.plot(x,a*x+b,'b-')
plt.axis([0,10,0,10])
plt.show()