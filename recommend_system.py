from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

original_path="D:/ml-100k/ml-100k/"

# def matrix_factorization(R,K,Rt,beta=0.004,steps=5000,alpha=0.0002):#使用GD,beta即为惩罚系数
#     N=len(R)
#     M=len(R[0])
#     P=np.random.rand(N,K)
#     Q=np.random.rand(M,K)
#     Q=Q.T
#     result=[]
#     for step in range(steps):
#         for i in range(len(R)):
#             for j in range(len(R[i])):
#                 if R[i][j]>0:
#                     eij=R[i][j]-np.dot(P[i,:],Q[:,j])
#                     for k in range(K):
#                         P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
#                         Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])

#         eR=np.dot(P,Q)
#         loss=0
#         for i in range(len(Rt)):
#             for j in range(len(Rt[i])):
#                 if Rt[i][j]>0:
#                     loss+=pow(Rt[i][j]-np.dot(P[i,:],Q[:,j]),2)
#                     for k in range(K):
#                         loss+=beta*(pow(P[i][k],2)+pow(Q[k][j],2))
#         result.append(loss)
#         if loss<0.0001:
#             break
#     return P,Q.T,result
def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.0002,beta=0.0002):
    Q=Q.T  # .T操作表示矩阵的转置
    result=[]
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j]) # .dot(P,Q) 表示矩阵内积
                    for k in range(K):
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
        result.append(e)
        if e<0.001:
            break
    return P,Q.T,result

def get_data(filename):
    header=['user_id','item_id','rating','timestamp']
    df=pd.read_csv(os.path.join(original_path+filename),names=header,sep='\t')
    return df

if __name__=='__main__':
    df=get_data('u.data')
    del df['timestamp']
    train_data,test_data=train_test_split(df,test_size=0.25)
    print(train_data.head())
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    # N=len(train_data)
    # M=len(train_data[0])
    # K=2
    # P=np.random.rand(N,K)
    # Q=np.random.rand(M,K)
    # nP,nQ,result=matrix_factorization(train_data,P,Q,K)
    # R_p=np.dot(nP,nQ.T)
    # print(R_p)

    
    
    
    
