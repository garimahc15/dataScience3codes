rom sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def load_dataset(path_to_file):
     dataframe= pd.read_csv(path_to_file)
     return dataframe

def normalize(dataframe):
     x= dataframe[[i for i in dataframe.columns if i!= 'Z_Scratch']]#do
not normalize class
     normalized= preprocessing.MinMaxScaler().fit_transform(x)
     df= pd.DataFrame(normalized, columns=[i for i in dataframe.columns
if i!= 'Z_Scratch'], index=dataframe.index)
     #passed columns so as to have col names else column index returned
are 0,1,,2,3...
     df= pd.concat([df, dataframe['Z_Scratch']], axis=1)
     return df

def standardize(dataframe):
     x= dataframe[[i for i in dataframe.columns if
i!='Z_Scratch']].values
     x= preprocessing.StandardScaler().fit_transform(x)
     return x

def KNNclassification(dataframe, label):
     dataframe= normalize(dataframe)
     y= dataframe[label]  #where label is dependent attribute i.e. to be
predicted
     x= dataframe.drop('Z_Scratch', axis=1)
#x contains dataframe with independent features(attributes) which will
#be used to predict label (contained in x)
     x_train, x_test, y_train, y_test= train_test_split(x,y,
test_size=0.3, random_state=42)
     classifier= KNeighborsClassifier(n_neighbors=k).fit(x_train,
y_train)
     y_pred= classifier.predict(x_test)
     print("percentage accuracy\n",percentage_accuracy(y_test, y_pred))
     print("confusion matrix\n",confusion_matrix(y_test, y_pred))
     return percentage_accuracy(y_test, y_pred)

def percentage_accuracy(y_test, y_pred):
     return(accuracy_score(y_test, y_pred)*100)

def bayes_classifier(dataframe):
     y= dataframe['Z_Scratch']  #where label is dependent attribute i.e.
to be predicted
     x= dataframe.drop('Z_Scratch', axis=1)
     x_train, x_test, y_train, y_test= train_test_split(x,y,
test_size=0.3, random_state=42)
     train= pd.concat([x_train,y_train], axis=1)
     gk= train.groupby('Z_Scratch')
     dataclass0 = gk.get_group(0)
     dataclass1= gk.get_group(1)
     dataclass0=dataclass0.drop('Z_Scratch', axis=1)
     dataclass1= dataclass1.drop('Z_Scratch', axis=1)
     mucls0= dataclass0.mean(axis=0)
     mucls0= np.array(mucls0)
     covmatcls0= np.array(dataclass0.cov())
     mucls1= dataclass1.mean(axis=0)
     mucls1= np.array(mucls1)
     covmatcls1= np.array(dataclass1.cov())
     d1= multivariate_normal(mean= mucls0, cov= covmatcls0,
allow_singular=True)
     d2= multivariate_normal(mean= mucls1, cov= covmatcls1,
allow_singular=True)

     y_predict=[]
     for i in range(len(x_test)) :
         if
d1.pdf(np.array(x_test.iloc[i]))>d2.pdf(np.array(x_test.iloc[i])):
             y_predict.append(0)
         else:
             y_predict.append(1)
     #print("y_predict from bayes : ", y_predict)
     print("percentage accuracy\n",percentage_accuracy(y_test,
y_predict))
     print("confusion matrix\n",confusion_matrix(y_test, y_predict))
     return percentage_accuracy(y_test, y_predict)

def dimensionality_reduction(dataframe, n):
     standardized= standardize(dataframe)
     pca= PCA(n_components=n)
     data_dr= pca.fit_transform(standardized)
     principaldf= pd.DataFrame(data_dr, columns= [j for j in
range(1,n+1)])
     finaldf= pd.concat([principaldf, dataframe['Z_Scratch']], axis=1)
     return finaldf


df= load_dataset("/local/user/Downloads/SteelPlateFaults-2class.csv")

K_val= [1,3,5,7,9,11,13,15,17,21]
print("for original dataframe")
print("KNN classification")
df_copy= df.copy()
df_normalized= normalize(df_copy)
for k in K_val:
     print("k= ",k)
     y_predicted= KNNclassification(df_normalized, 'Z_Scratch')

print("\n\n bayes classification\n")
bayes_classifier(df)

print("\n\nFOR REDUCED DIMENSIONAL DATA\n")
bayes_accuracy=[]
for i in range(1,len(df.columns)): #from len 1 to len(df.columns)
     reduced_data= dimensionality_reduction(df,i)
     print("\nNo. of dimensions : ", i)
     print("KNN classification")
     knnaccuracy=[]
     for k in K_val:
         print("k= ",k)
         knnacc= KNNclassification(reduced_data, 'Z_Scratch')
         knnaccuracy.append(knnacc)

     plt.plot(K_val, knnaccuracy)
     plt.xlabel("K_values")
     plt.ylabel("KNN Accuracy for dimensions")
     plt.show()
     print("\n\nbayes classification\n")
     bayesprdct= bayes_classifier(reduced_data)
     bayes_accuracy.append(bayesprdct)

plt.plot([i for i in range(1, len(df.columns))], bayes_accuracy)
plt.xlabel("reduced dimensions")
plt.ylabel("accuracy")
