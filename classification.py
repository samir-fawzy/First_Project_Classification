import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
############### data ###############
path = "C:\\ML\\Classification\\First_Project_Classification\\data.txt"
data = pd.read_csv(path,header= None,names=["Exam1","Exam2","Admitted"])
# print(data)

data.insert(0,"Bais",1)
# print(data)

# separate featurs and output 
col =data.shape[1] # output number of columns for data 
x = data.iloc[ : , 0 : col - 1]
y = data.iloc[ : , col - 1 : col]

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
x = np.matrix(x.values)
y= np.matrix(y.values)
theta = np.zeros((x.shape[1],1))

# print(x)
# print(f"y shape : {y.shape}") 
# print(f"x shape : {x.shape}")
# print(f"theta shape : {theta.shape}")

# implement Sigmoid Function
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))
# test = np.log(Sigmoid(x @ theta))
# print (f"test {test.shape}")

#implement Cost Function
def Cost(theta,x, y):
    m = len(x)
    prediction = Sigmoid(x @ theta)
    epsilon = 1e-5
    cost = (-1/m) * (np.sum(np.multiply(y, np.log(prediction + epsilon).T) + np.multiply((1 - y), np.log(1 - prediction + epsilon)).T))
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

result = sc.optimize.fmin_tnc(func=Cost, x0=theta, args=(x, y), fprime=Gradient)


newTheta = result[0]

print(f"newtheta : {newTheta.shape}")
print(f"x : {x.shape}")
print(f"y : {y.shape}")



# print(f"newTheta :\n{newTheta}")
# print("============================")
cost0 = Cost(theta,x,y)

print(f"Cost Before = {cost0}")
print("============================")
cost1 = Cost(newTheta,x,y)

print(f"Cost After: {cost1}")

def predict(x, theta):
    p = Sigmoid(x @ newTheta)
    for i in range(96):
        if p[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    return p

t = predict(x,newTheta)

print(t.reshape(96,1))

# p = Sigmoid(x @ newTheta)

# print(p.shape)

# for i in range(96):
#     if p[0,i] > 0.5:
#         p[0,i] = 1
#     else:
#         p[0,i] = 0

# print(p)

