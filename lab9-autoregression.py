import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import scipy.stats
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import AR

data=pd.read_csv("/Users/rupanshirupanshi/Downloads/SoilForce.csv",header=0,index_col=0)
force=data['Force']
data.plot()
plt.xlabel('Date')
plt.ylabel('force in N')
plt.show()
print("plot of date and force")


###################################
#ques1b
#lag=1
values=DataFrame(data.values)
dataframe=concat([values.shift(1),values],axis=1)
dataframe.columns=['x(t-1)','x(t)']
result=dataframe.corr()
print(result)
'''
print("auto correlation plot")
autocorrelation_plot(data)
plt.show()
'''
print("acf plot")
sm.graphics.tsa.plot_acf(data,lags=31)
plt.show()


#######################################
#persistence model
#splitting data into test and train
print("persistence model")
X=dataframe.values
train,test=X[1:71],X[71:]
train_X,train_y=train[:,0],train[:,1]
test_X,test_y=test[:,0],test[:,1]
def model_persistence(x):
     return x
predict=list()
for x in test_X:
     y_t=model_persistence(x)
     predict.append(y_t)
score=mean_squared_error(test_y, predict)
print('rmse',np.sqrt(score))

########################################
#ar model
#train AR
print("auto regression model")
model=AR(train_y)
model_fit=model.fit()
print('lag:',model_fit.k_ar)
predict_ar=model_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=False)
error=mean_squared_error(test_y, predict_ar)
print("rmse",np.sqrt(error))
