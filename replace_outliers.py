def replace_outliers(dataframe):
    q1= dataframe.quantile(0.25)
    q2= dataframe.quantile(0.75)
    cols= list(dataframe.columns)
    del cols[len(cols)-1]
    median= [ss.median(dataframe[i]) for i in cols]
    for i in range(len(cols)):
        for j in range(len(dataframe)):
            if (q1[i]-1.5*(q2[i]-q1[i])) <dataframe[cols[i]][j]< (q2[i] + 1.5*(q2[i]-q1[i])):
                continue
            else:
                dataframe[cols[i]][j]= median[i]
    
    return dataframe
