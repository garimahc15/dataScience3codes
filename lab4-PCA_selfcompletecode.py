import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import  mean_squared_error
import numpy as np

mu= [int(input()) for i in range(2)]
print("enter values: var of x, covar of x&y, covar of x&y, var of y")
covar= [[int(input()) for i in range(2)] for j in range(2)]
print(mu, covar)
x,y= np.random.multivariate_normal(mu,covar,10).T
#print("actual covar matrix based on data : ",np.cov(x,y))

plt.scatter(x,y)
plt.show()

#Prob3/4
D= [[x[i], y[i]] for i in range(len(x))]
meanx= np.mean(x)
meany= np.mean(y)
#data cleaning.. cleaned data in X
X= [[D[i][0]-meanx, D[i][1]-meany] for i in range(len(x))] #centered
matrix
Xcol1= [X[i][0] for i in range(len(x))]
Xcol2= [X[i][1] for i in range(len(x))]
C= np.dot(np.transpose(X), X)
print(C)

eigval, eigvec= np.linalg.eig(C)

#prob 3
srtdeigval= sorted(eigval, reverse=True)
np.transpose(eigvec)
if srtdeigval[0]==eigval[0]:
     srtdeigvec= eigvec #each row contains 1 eigenvector
else:
     temp=[]
     temp=eigvec[0]
     eigvec[0]=eigvec[1]
     eigvec[1]=temp
     srtdeigvec= eigvec
Dcap=[]
for i in range(len(x)):
     Dcap.append([np.dot(srtdeigvec[0],np.transpose([D[i][0], D[i][1]])),
np.dot(srtdeigvec[1],np.transpose([D[i][0], D[i][1]]))])

#reconstructed data X reD
reD= [np.add(np.multiply(srtdeigvec[0],
Dcap[i][0]),np.multiply(srtdeigvec[1], Dcap[i][1])) for i in
range(len(x))] #reconstructed D
mse= mean_squared_error(X, reD)
print("mse : ", mse)

#3-2
plt.scatter(Xcol1, Xcol2)
plt.quiver(srtdeigvec[0], srtdeigvec[1], ['red'])
plt.show()

#3-3
print("eigen values : ", srtdeigval)
Dcapcol1= [Dcap[i][0] for i in range(len(x))]
Dcapcol2= [Dcap[i][1] for i in range(len(x))]
Dcapcol1var= np.var(Dcapcol1)
Dcapcol2var= np.var(Dcapcol2)
print("Dcapcol1var : ", Dcapcol1var)
print("Dcapcol2var : ", Dcapcol2var)

#3-4

print("covar matrix of Dcap : ", np.cov(Dcapcol1,Dcapcol2))
#3-5 the one that has low variance of data along that direction

#prob4
#for 1st eigen value
Dcap1=[]
for i in range(len(x)):
     Dcap1.append((np.dot(srtdeigvec[0],np.transpose([D[i][0],
D[i][1]]))))

#reconstructed data X reD1
reD1= [np.multiply(srtdeigvec[0], Dcap1[i]) for i in range(len(x))]
#reconstructed D
mse1= mean_squared_error(X, reD1)
print("mse1 : ", mse1)
red1col1= [reD1[i][0] for i in range(len(x))]
red1col2= [reD1[i][1] for i in range(len(x))]

plt.scatter(Xcol1,Xcol2)
plt.scatter(red1col1, red1col2, color='green')
plt.quiver(srtdeigvec[0], srtdeigvec[1], ['red'])
plt.show()
#for 2nd eigen value
Dcap2=[]
for i in range(len(x)):
     Dcap2.append((np.dot(srtdeigvec[1],np.transpose([X[i][0],
X[i][1]]))))

#reconstructed data X reD2
reD2= [np.multiply(srtdeigvec[1], Dcap2[i]) for i in range(len(x))]
#reconstructed D
mse2= mean_squared_error(X, reD2)
print("mse2 : ", mse2)
red2col1= [reD2[i][0] for i in range(len(x))]
red2col2= [reD2[i][1] for i in range(len(x))]

plt.scatter(Xcol1,Xcol2)
plt.scatter(red2col1, red2col2, color='green')
plt.quiver(srtdeigvec[0], srtdeigvec[1], ['red'])
plt.show()
