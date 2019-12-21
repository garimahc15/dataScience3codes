import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn import mixture
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import scipy.stats as st
# Import or Load the data
def load_dataset(path_to_file):
     data = pd.DataFrame(pd.read_csv(path_to_file))
     return data

def standardization(dataframe):
     names = dataframe.columns
     scaler = preprocessing.StandardScaler()
     dataframe[[names[i] for i in range(len(names)-1)]] =
scaler.fit_transform(dataframe[[names[i] for i in range(len(names)-1)]])
     return dataframe

def train_test_split_(data,y):
     print(data.shape)
     X_train, X_test, y_train, y_test = train_test_split(data,
y,test_size=0.30, random_state=42)
     return X_train, X_test, y_train, y_test

def prediction_accuracy(y_true, y_pred):
     return mean_squared_error(y_true, y_pred)*100
'''
def confusion_matrix_(y_true,y_pred):
     return confusion_matrix(y_true, y_pred)
'''

###################################################################################

data =
load_dataset("/Users/rupanshirupanshi/Downloads/winequality-red.csv")
train = data.drop('quality',axis=1)
test = data['quality']
X_train, X_test, y_train, y_test = train_test_split(train,test)

###############################################################################
#ques2
print("simple linear regression")
pH=data.iloc[:,8:9]
quality=data.iloc[:,11:12]
xpH_train, xpH_test, yQ_train, yQ_test = train_test_split(pH,quality)
reg=LinearRegression()
reg.fit(xpH_train,yQ_train)
y_pred=reg.predict(xpH_train)
y_pred2=reg.predict(xpH_test)
print("Percentage Accuracy on train data ",
prediction_accuracy(y_train,y_pred),"%")
print("prediction accuracy on test
data",prediction_accuracy(yQ_test,y_pred2))
plt.scatter(yQ_test,y_pred2 , color = "blue",  label = 'Test data')
plt.xlabel('original quality')
plt.ylabel('predicted quality')
plt.show()
plt.scatter(xpH_train,yQ_train)
plt.plot(xpH_train,y_pred,color='red')
plt.xlabel('pH')
plt.ylabel('Quality')
plt.show()

###############################################################################
#polynomial curve fitting
#ques3
print("polynomial curve fitting")
rmse_tst=[]
rmse_tr=[]
p=[2,3,4,5]
for i in range(2,6):
     print("degree : ",i)
     polynomial_features= PolynomialFeatures(degree=i)
     x_poly = polynomial_features.fit_transform(pH)
     predict=polynomial_features.fit_transform(xpH_test)
     predict_tr=polynomial_features.fit_transform(xpH_train)
     reg=LinearRegression()
     reg.fit(x_poly,quality)
     y_pred2 = reg.predict(predict)
     y_pred_tr = reg.predict(predict_tr)
     print("prediction accuracy on test
data",prediction_accuracy(yQ_test,y_pred2))
     print("prediction accuracy on train
data",prediction_accuracy(yQ_train,y_pred_tr))
     rmse_tst.append(prediction_accuracy(yQ_test,y_pred2))
     rmse_tr.append(prediction_accuracy(yQ_train,y_pred_tr))
     plt.scatter(yQ_test,y_pred2 , color = "blue",  label = 'Test data')
     plt.xlabel('original quality')
     plt.ylabel('predicted quality')
     plt.show()
     plt.scatter(xpH_train,yQ_train)
     plt.xlabel('pH')
     plt.ylabel('Quality')
     plt.scatter(xpH_test,y_pred2)
     plt.show()
plt.bar(p,rmse_tst)
plt.xlabel('degree')
plt.ylabel('rmse')
plt.show()
plt.bar(p,rmse_tr)
plt.xlabel('degree')
plt.ylabel('rmse')
plt.show()


################################
#multiple linear regression
#ques4
print('multiple linear regression')
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred3=reg.predict(X_test)
y_pred_trd=reg.predict(X_train)
print("prediction accuracy on test
data",prediction_accuracy(y_test,y_pred3),'%')
print("prediction accuracy on train
data",prediction_accuracy(y_train,y_pred_trd),'%')
plt.scatter(y_test,y_pred3 , color = "blue",  label = 'Test data')
plt.xlabel('original quality')
plt.ylabel('predicted quality')
plt.show()

##################################
#ques5
#multivariate polynomial regression
print('multivariate polynomial regression')
rmse_tst=[]
rmse_tr=[]
p=[2,3,4,5]
for i in range(2,6):
     print("degree : ",i)
     polynomial_features= PolynomialFeatures(degree=i)
     x_poly = polynomial_features.fit_transform(train)
     predict=polynomial_features.fit_transform(X_test)
     predict_tr=polynomial_features.fit_transform(X_train)
     reg=LinearRegression()
     reg.fit(x_poly,quality)
     y_pred2 = reg.predict(predict)
     y_pred_tr=reg.predict(predict_tr)
     print("prediction accuracy on test
data",prediction_accuracy(y_test,y_pred2))
     rmse_tst.append(prediction_accuracy(y_test,y_pred2))
     print("prediction accuracy on train
data",prediction_accuracy(y_train,y_pred_tr))
     rmse_tr.append(prediction_accuracy(y_train,y_pred_tr))
     plt.scatter(y_test,y_pred2 , color = "blue",  label = 'Test data')
     plt.xlabel('original quality')
     plt.ylabel('predicted quality')
     plt.show()
plt.bar(p,rmse_tst)
plt.xlabel('degree')
plt.ylabel('rmse')
plt.show()
plt.bar(p,rmse_tr)
plt.xlabel('degree')
plt.ylabel('rmse')
plt.show()


corr=[]
cols=['fixed acidity','volatile acidity','citric acid','residual
sugar','chlorides','free sulfur dioxide','total sulfur
dioxide','density','pH','sulphates','alcohol']
for i in range(0,11):
     x=st.pearsonr(data[cols[i]],data['quality'])
     corr.append(x)
print(sorted(corr))

################################
#multiple linear regression
#ques6a
print("/nmultiple linear regression")
maxcorr=data[['fixed acidity','alcohol']]
quality=data.iloc[:,11:12]
X6_train, X6_test, y6_train, y6_test = train_test_split(maxcorr,quality)
reg=LinearRegression()
reg.fit(X6_train,y6_train)
y_pred6=reg.predict(X6_test)
y_pred6_tr=reg.predict(X6_train)
#print("Percentage Accuracy ", prediction_accuracy(y_train,y_pred2),"%")
print("prediction accuracy on test
data",prediction_accuracy(y6_test,y_pred6))
print("prediction accuracy on train
data",prediction_accuracy(y6_train,y_pred6_tr))
plt.scatter(y6_test,y_pred6 , color = "blue",  label = 'Test data')
plt.xlabel('original quality')
plt.ylabel('predicted quality')
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['fixed
acidity'],data['alcohol'],data['quality'],c='blue', marker='o',
alpha=0.5)
ax.plot_surface(X6_test['fixed acidity'],X6_test['alcohol'],y_pred6)
plt.show()

##################################
#ques6b
#multivariate polynomial regression
print('multivariate polynomial regression')
for i in range(2,6):
     print("degree : ",i)
     polynomial_features= PolynomialFeatures(degree=i)
     x_poly = polynomial_features.fit_transform(maxcorr)
     predict=polynomial_features.fit_transform(X6_test)
     predict_tr=polynomial_features.fit_transform(X6_train)
     reg=LinearRegression()
     reg.fit(x_poly,quality)
     y_pred2 = reg.predict(predict)
     y_pred_tr=reg.predict(predict_tr)
     print("prediction accuracy on test
data",prediction_accuracy(y6_test,y_pred2))
     print("prediction accuracy on train
data",prediction_accuracy(y6_train,y_pred_tr))
     plt.scatter(y6_test,y_pred2 , color = "blue",  label = 'Test data')
     plt.xlabel('original quality')
     plt.ylabel('predicted quality')
     plt.show()
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(data['fixed
acidity'],data['alcohol'],data['quality'],c='blue', marker='o',
alpha=0.5)
     ax.plot_surface(X6_test['fixed acidity'],X6_test['alcohol'],y_pred2)
     plt.show()


