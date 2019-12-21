import pandas as pd
import statistics as ss


def read_data(path_to_file):
    file= pd.read_csv(path_to_file)
    return pd.DataFrame(file)

def Range(dataframe,attribute_name):
    low= min(dataframe[attribute_name])
    high= max(dataframe[attribute_name])
    pair= (low,high)
    return pair

def min_max_normalization(dataframe, newrange):
    cols= list(dataframe.columns)
    del cols[len(cols)-1]
    mnmx= [Range(dataframe, col) for col in cols ]
    for i in range(len(cols)):
        for j in range(len(dataframe)):
            dataframe[cols[i]][j]= (((dataframe[cols[i]][j]-mnmx[i][0])/(mnmx[i][1]-mnmx[i][0]))*(newrange[1]-newrange[0])) +newrange[0]
    
    return dataframe

def standardize(dataframe):
    mean= [ss.mean(dataframe[col]) for col in cols]
    stdev= [ss.stdev(dataframe[col]) for col in cols]
    print("mean : ", mean)
    print("stdev : ", stdev)
    
    for i in range(len(cols)):
        for j in range(len(dataframe)):
            dataframe[cols[i]][j]= (dataframe[cols[i]][j]-mean[i])/stdev[i]
    
    return dataframe 

