from numpy import *
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import sklearn.linear_model as s_lm
def load_data():
    data=load_svmlight_file("C:/Users/QinQS/Desktop/housing_scale.txt")
    return data[0],data[1]

X,y=load_data()

X=X.A

x_train,x_test,y_train,y_test=train_test_split(X,y)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1])

y_train=y_train.reshape((y_train.shape[0],1))

x_test=x_test.reshape((x_test.shape[0],x_test.shape[1]))

y_test=y_test.reshape(y_test.shape[0],1)

def loss_calc(theta,X,y):
    m=X.shape[0]
    J=(1/(2*m))*sum((X@theta-y)**2)
    return J

def normal(X,y):
    theta=linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def theta_init(X):
    theta=zeros([X.shape[1],1])
    return theta

theta=theta_init(x_train)
loss=loss_calc(theta,x_train,y_train)
theta=normal(x_train,y_train)
loss_train=loss_calc(theta,x_train,y_train)
loss_value=loss_calc(theta,x_test,y_test)
print(loss)
print(loss_train)
print(loss_value)
