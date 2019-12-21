import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
from sklearn import metrics 
from sklearn.cluster import DBSCAN 

def purity_score(actual,predicted): 
     mat = metrics.cluster.contingency_matrix(actual,predicted) 
     a = np.sum(np.amax(mat,axis =0))/np.sum(mat) 
     return a 

def PCAA(dataframe): 
     pca = PCA(n_components=2) 
     newdata = pd.DataFrame(pca.fit_transform(dataframe)) 
     newdata.columns=['X','Y'] 
     return newdata 

def kmen(x,i,df): 
     print("kmeans") 
     kmeans = KMeans(n_clusters = i,random_state = 0).fit(x) 
     pred = kmeans.labels_ 
     centres = kmeans.cluster_centers_ 
     y_kmeans = kmeans.predict(x) 
     plt.scatter(x[:,0],x[:,1],c = y_kmeans,cmap = 'viridis') 
     plt.scatter(centres[:,0],centres[:,1],c="black",alpha = 1) 
     plt.show() 
     print("sum of squared distance") 
     print(kmeans.inertia_) 
     actual = df.Species 
     pty = purity_score(actual,pred) 
     print("the value of purity score",pty) 

def ques5(x,df): 
     K = [ 3] 
     for i in K: 
         print("the value of k ",i) 
         kmen(x,i,df) 

def DBS(x,df): 
     print("DBSCAN") 
     eps=[0.05,0.5,0.95] 
     actual = df.Species 
     for i in eps: 
         model=DBSCAN(eps = i,min_samples = 10) 
         model.fit(x) 
         labels=model.labels_ 
         print(labels) 
         plt.scatter(x[:,0],x[:,1],c=labels,cmap='viridis') 
         plt.show() 
         print(len(set(labels))-(1 if -1 in labels else 0)) 
         pty=purity_score(actual,labels) 
         print("the value of purity score",pty) 

def agglomerative(X,df): 
     print("agglomerative") 
     actual = df.Species 
     ac2 = AgglomerativeClustering(n_clusters = 3) 
     ac2.fit(X) 
     labels=ac2.labels_ 
     plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis') 
     plt.show() 
     #print(labels) 
     pty=purity_score(actual,labels) 
     print("the value of purity score",pty) 
     print(len(set(labels))-(1 if -1 in labels else 0)) 

def main(): 
     df = pd.read_csv("/Users/rupanshirupanshi/Downloads/Iris.csv") 
     red_data = PCAA(df.iloc[:,1:5]) 
     x = red_data.values 
     ques5(x,df) 
     agglomerative(x,df) 
     DBS(x,df) 
main() 
