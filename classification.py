import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
############### data ###############
path = "C:\ML\Classification\First_Project_Classification\data.txt"
data = pd.read_csv(path,header= None,names=["Exam1","Exam2","Admitted"])
# print(data)

data.insert(0,"Bais",1)
# print(data)

# separate featurs and output 
col =data.shape[1] # output number of columns for data 
x = data.iloc[:,0:col - 1]
y = data.iloc[:,col - 1 : col]

# print("x\n",x)
# print("="*50)
# print("y\n",y)

# separate positive and negative data
positive = data[data["Admitted"].isin([1])]
negative = data[data["Admitted"].isin([0])]

# print(positive)
# print(negative)

# draw positive and negative data
# fig,ax = plt.subplots()
# ax.scatter(positive["Exam1"],positive["Exam2"],marker='o',c='b') 

# ax.scatter(negative["Exam1"],negative["Exam2"],marker='x',c='r')
# plt.show()

# convert x , y to matrix
x_matrix = np.matrix(x.values)
y_matrix = np.matrix(y.values)
theta = np.zeros(x_matrix.shape[1])

# print(x_matrix)
# print(y_matrix) 
# print(theta.shape)

# implement Sigmoid Function
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

#implement Cost Function
def Cost(theta,x, y):
    m = len(y)
    prediction = Sigmoid(x @ theta)
    cost = (-1/m) * (np.sum(y @ np.log(prediction) + (1 - y)@(np.log(1 - prediction))))
    return cost

# implement Gradiant Function
def Gradient(theta,x,y):
    m = len(y)
    theta = np.matrix(theta).reshape(-1, 1)
    parameters = x.shape[1]
    grad = np.zeros((parameters,1)) # 
    error =  Sigmoid(x @ theta) - y # 
    grad = (1/m) * (x.T @ error)
    return grad

cost = Cost(theta,x_matrix,y_matrix)
print(f"Cost = {cost}")

result = sc.optimize.fmin_tnc(func=Cost, x0=theta, args=(x_matrix, y_matrix), fprime=Gradient)

# optimize_theta = result[0]
# print(result)
# print(optimize_theta)

# test = Sigmoid(x_matrix @ optimize_theta)
# c = Cost(optimize_theta,x_matrix,y_matrix)   
# print(c)

sa = Gradient(theta,x_matrix,y_matrix)  
print("------------------",sa.shape)
print(x_matrix.shape    )
cost = Cost(sa,x_matrix,y_matrix)