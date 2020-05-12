from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import sklearn.linear_model as s_lm

def load_data():
    data=load_svmlight_file("File location of your data set")
    return data[0],data[1]

X,y=load_data()

X=X.A

x_train,x_test,y_train,y_test=train_test_split(X,y)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1])

y_train=y_train.reshape((y_train.shape[0],1))

x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]))

y_test=y_test.reshape(y_test.shape[0],1)    

def loss_calc(X,y,theta):
    m=X.shape[0]
    J=sum((y-X@theta)**2)/(2*m)
    return J

def grad_ds(X,y,theta,lr=0.0044):
    m=X.shape[0]   
    theta_grad=2*(y-X@theta)*(-X.T)/m
    theta_grad=theta_grad.reshape(len(theta_grad),1)
    theta=theta-lr*theta_grad
    return theta

def theta_init(X):
    theta=zeros([X.shape[1],1])
    return theta

theta=theta_init(x_train)
loss=loss_calc(x_train,y_train,theta)
print("Loss:")
print(loss)
print("Loss_train:       Loss_val:")
for i in range(10000):
    k=random.randint(0,x_train.shape[0])
    theta=grad_ds(x_train[k],y_train[k],theta)
    loss_train=loss_calc(x_train,y_train,theta)
    loss_val=loss_calc(x_test,y_test,theta)
    plt.scatter(i,loss_train,c='red',label='loss_train')
    plt.scatter(i,loss_val,c='blue',label='loss_val')
    if i%999==0:
        print(str(loss_train)+"  "+str(loss_val))
plt.show()
