import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/local/user/Downloads/landslide_data2_miss.csv")
#pd.set_option('display.max_columns', 500)


#print(df1.describe(include = 'all'))

import scipy.stats as s
import statistics as ss
#reading the various columns
#ques1
df1= df.isna().sum(axis=1)
#print(df1)

print("before deleting rows")
print("missing values column wise : ", df.isnull().sum()) #print missing values in each column
print("total no. of missing values : ", df.isnull().sum().sum()) #missing values in full file
    

df1 = df.fillna(df.median(), inplace=True)
df4 = df.interpolate(method = 'linear')
print("\n\n\n\MEdian values: " )
cols = ['dispx', 'dispy', 'dispz', 'temperature', 'humidity', 'rain']
    
df2 = pd.read_csv("/local/user/Downloads/landslide_data2_original.csv")
    
for i in cols:
    fig = plt.figure()
    print(i, ' : \n' )
    print("miss")
    print("mean = ",ss.mean(df[i].values))
    print("median = ",ss.median(df[i].values))
    print("mode = ",ss.mode(df[i].values))
    print("minimum = ",min(df[i].values))
    print("maximum = ",max(df[i].values))
    print("standard deviation = ",ss.stdev(df[i].values))
    print("\n\n\n")
    print("original")
    print("mean = ",ss.mean(df2[i].values))
    print("median = ",ss.median(df2[i].values))
    print("mode = ",ss.mode(df2[i].values))
    print("minimum = ",min(df2[i].values))
    print("maximum = ",max(df2[i].values))
    print("standard deviation = ",ss.stdev(df2[i].values))
    print("\n\n\n")
    
    ax = fig.add_subplot(1,2,1)
    ax.boxplot(df[i])
    ax = fig.add_subplot(1,2,2)
    ax.boxplot(df2[i])   
    plt.show()
    
    
from sklearn.metrics import mean_squared_error as rmse
from math import sqrt
for i in cols:
    r = rmse(df[i].values, df2[i].values)
    print(i, ' :  ' )
    print("the root mean square error (RMSE)between the original and replaced values : ", r)
