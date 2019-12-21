import pandas as pd
import statistics as ss
from sklearn import preprocessing



df= read_data("/home/garimachahar/Downloads/winequality_red_original.csv")
df= replace_outliers(df)
x= df.values #return dataframe in a numpy array form
#normalized= preprocessing.MinMaxScaler().fit_transform(x)
#df= pd.DataFrame(normalized)
standardized= preprocessing.StandardScaler().fit_transform(x)
df= pd.Dataframe(standardized)
print(df)
