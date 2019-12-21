import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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


def percentage_accuracy(y_test, y_pred):
     return(accuracy_score(y_test, y_pred)*100)

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

def dimensionality_reduction(dataframe, n):
     standardized= standardize(dataframe)
     pca= PCA(n_components=n)
     data_dr= pca.fit_transform(standardized)
     principaldf= pd.DataFrame(data_dr, columns= [j for j in
range(1,n+1)])
     finaldf= pd.concat([principaldf, dataframe['Z_Scratch']], axis=1)
     return finaldf

def multimodal_classifier(dataframe):
     data= pd.DataFrame(standardize(dataframe))
     accuracy=[]
     y_predict=[]
     y= dataframe['Z_Scratch']  #where label is dependent attribute i.e.
to be predicted
     x= data
     x_train, x_test, y_train, y_test= train_test_split(x,y,
test_size=0.3, random_state=42)
     train= pd.concat([x_train,y_train], axis=1)
     gk= train.groupby('Z_Scratch')
     data0= pd.DataFrame(gk.get_group(0))
     data1= pd.DataFrame(gk.get_group(1))
     data0=data0.drop('Z_Scratch', axis=1)
     data1= data1.drop('Z_Scratch', axis=1)
     Q= [1,2,4, 8,16]
     for q in Q:
         print("Q= ",q)
         y_predict=[]
         gmm0= mixture.GaussianMixture(n_components=q)
         g0= gmm0.fit(data0)
         p0= g0.score_samples(x_test)
         gmm1= mixture.GaussianMixture(n_components=q)
         g1=gmm1.fit(data1)
         p1= g1.score_samples(x_test)
         for i in range(len(p0)):
             if p0[i]>p1[i]:
                 y_predict.append(0)
             else:
                 y_predict.append(1)
         print("y_predict : ", y_predict)
         print("percentage accuracy\n",percentage_accuracy(y_test,
y_predict))
         print("confusion matrix\n",confusion_matrix(y_test, y_predict))
         accuracy.append(percentage_accuracy(y_test, y_predict))
     plt.plot(Q, accuracy)
     plt.xlabel("Q avlues")
     plt.ylabel("accuracy")
     plt.show()
     #labels= gmm.predict(df)
     #df['labels']= labels
     #print df
     #d0= df[df['labels']==0]
     #d1= df[df['labels']==1]

df= load_dataset("/local/user/Downloads/SteelPlateFaults-2class.csv")
#PART 1
korigacc=[]
K_val= [1,3,5,7,9,11,13,15,17,21]
print("for original dataframe")
print("KNN classification")
for k in K_val:
     print("k= ",k)
     acc= KNNclassification(df, 'Z_Scratch')
     korigacc.append(acc)
print("max accuracy is for K : ", K_val[korigacc.index(max(korigacc))])
plt.plot(K_val, korigacc)
plt.xlabel("k values")
plt.ylabel("accuracy")
#PART2
multimodal_classifier(df)

#labels were to write cluster no. in this case Q=2 so cluster is 0 or 1
#plot scatter plot for all d0,d1 where d0 is 1st cluster dataframe.. d1
is another cluster dataframe

# print the converged log-likelihood value
#print(gmm.lower_bound_)

# print the number of iterations needed
# for the log-likelihood value to converge
#print(gmm.n_iter_)

#PART 3
print("\n\nFOR REDUCED DIMENSIONAL DATA\n")
for i in range(1,3): #from len 1 to len(df.columns)
     reduced_data= dimensionality_reduction(df,i)
     print("\nNo. of dimensions : ", i)
     print("KNN classification")
     knnaccuracy=[]
     for k in K_val:
         print("k= ",k)
         knnacc= KNNclassification(reduced_data, 'Z_Scratch')
         knnaccuracy.append(knnacc)
     print("max accuracy is for K : ",
K_val[knnaccuracy.index(max(knnaccuracy))])
     #plt.plot(K_val, knnaccuracy)
     #plt.xlabel("K_values")
     #plt.ylabel("KNN Accuracy for dimensions")
     #plt.show()
     print("\n\n multimodal gaussian classification\n")
     multimodal_classifier(df)
     #plt.plot([1,2,4,8,16], gmm_orig_acc)
