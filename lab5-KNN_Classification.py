#Assignment 5
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
#import numpy as np

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
     standardized= preprocessing.StandardScaler().fit_transform(x)
     df= pd.DataFrame(standardized, columns= [i for i in
dataframe.columns if i!= 'Z_Scratch'], index= dataframe.index)
     df= pd.concat([df, dataframe['Z_Scratch']], axis=1)
     return df

def train_test_split(dataframe, label, path_train, path_test):
     from sklearn.model_selection import train_test_split
     y= dataframe[label]  #where label is dependent attribute i.e. to be
predicted
     x= dataframe.drop('Z_Scratch', axis=1)
#x contains dataframe with independent features(attributes) which will
#be used to predict label (contained in x)
     x_train, x_test, y_train, y_test= train_test_split(x,y,
test_size=0.3, random_state=42)
     train= pd.concat([x_train,y_train], axis=1) #concat 2 dataframes
(train)
     test= pd.concat([x_test, y_test], axis=1) #test dataframe
     train.to_csv(path_train) #save train data in csv file
     test.to_csv(path_test) #save test data in another csv file
     #x_train=x_train.drop(x_train.columns[0], axis=1)
     return x_train, x_test, y_train, y_test

def classification(x_train, y_train, x_test):
     classifier= KNeighborsClassifier(n_neighbors=k).fit(x_train,
y_train)
     y_pred= classifier.predict(x_test)
     return y_pred

def percentage_accuracy(y_test, y_pred):
     return(accuracy_score(y_test, y_pred)*100)


df= load_dataset("/local/user/Downloads/SteelPlateFaults-2class.csv")
x_train, x_test, y_train, y_test=train_test_split(df, 'Z_Scratch',
'/local/user/Downloads/SteelPlateFaults-2class-train.csv',
'/local/user/Downloads/SteelPlateFaults-2class-test.csv')
#dataframe.head returns first 5 rows
K_val= [1,3,5,7,9,11,13,15,17,21]
print("for dataframe passed")
accuracy_original= []
i=0
for k in K_val:
     print("k= ",k)
     y_predicted= classification(x_train, y_train, x_test)
     print(confusion_matrix(y_test, y_predicted))
     accuracy_original.append(percentage_accuracy(y_test, y_predicted))
     print(accuracy_original[i])
     i=i+1

nordf= normalize(df)
nordf.to_csv('/local/user/Downloads/SteelPlateFaults-2class-Normalised.csv')
x_train, x_test, y_train, y_test=train_test_split(nordf, 'Z_Scratch',
'/local/user/Downloads/SteelPlateFaults-2class-train-normalise.csv',
'/local/user/Downloads/SteelPlateFaults-2class-test-normalise.csv')
#dataframe.head returns first 5 rows
print("\n\nFor normalized data")
accuracy_nor= []
i=0
for k in K_val:
     print("k= ",k)
     y_predicted= classification(x_train, y_train, x_test)
     print(confusion_matrix(y_test, y_predicted))
     accuracy_nor.append(percentage_accuracy(y_test, y_predicted))
     print(accuracy_nor[i])
     i=i+1

standf= standardize(df)
standf.to_csv('/local/user/Downloads/SteelPlateFaults-2class-Standardised.csv')
x_train, x_test, y_train, y_test=train_test_split(standf, 'Z_Scratch',
'/local/user/Downloads/SteelPlateFaults-2class-train-standardise.csv',
'/local/user/Downloads/SteelPlateFaults-2class-test-standardise.csv')
#dataframe.head returns first 5 rows
print("\n\nfor standardized data")
accuracy_stan= []
i=0
for k in K_val:
     print("k= ",k)
     y_predicted= classification(x_train, y_train, x_test)
     print(confusion_matrix(y_test, y_predicted))
     accuracy_stan.append(percentage_accuracy(y_test, y_predicted))
     print(accuracy_stan[i], "%")
     i=i+1

plt.plot(K_val, accuracy_original)
plt.plot(K_val, accuracy_nor, color='red')
plt.plot(K_val, accuracy_stan, color= 'green')
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()
