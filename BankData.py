import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def datapreprocessing(df):
    d1=pd.get_dummies(df['default'],drop_first=False).rename(columns=lambda x:'default_'+str(x))
    d2=pd.get_dummies(df['housing'],drop_first=False).rename(columns=lambda x:'housing_'+str(x))
    d3=pd.get_dummies(df['loan'],drop_first=False).rename(columns=lambda x:'loan_'+str(x))
    d4=pd.get_dummies(df['marital'],drop_first=False).rename(columns=lambda x:'marital_' + str(x))
    d5=pd.get_dummies(df['contact'],drop_first=False).rename(columns=lambda x:'contact_' + str(x))
    d6=pd.get_dummies(df['poutcome'],drop_first=False).rename(columns=lambda x:'poutcome_' + str(x))

    df.drop(columns =['marital','contact','poutcome','default','housing','loan'],inplace=True)
    df=pd.concat([df,d1,d2,d3,d4,d5,d6], axis=1)


    job_map       = {'management':1, 'technician':2, 'entrepreneur':3, 'blue-collar':4, 'unknown':5,
                     'retired':6, 'admin.':7 ,'services':8, 'self-employed':9, 'unemployed':10, 'housemaid':11,
                     'student':12}
    education_map = {'tertiary':1 ,'secondary':2,'unknown':4, 'primary':3}
    month_map     = {'may':5, 'jun':6, 'jul':7 ,'aug':8 ,'oct':10 ,'nov':11, 'dec':12, 'jan':1, 'feb':2 ,'mar':3,                      'apr':4 ,'sep':9}
    y_map         = {"no":0,"yes":1}

    df["job_"]       = df.job.map(job_map)
    df["education_"] = df.education.map(education_map)
    df["month_"]     = df.month.map(month_map)
    df["target"]     = df.y.map(y_map)
    df.drop(columns  = ["job","education","month","y"],inplace=True)
    data = df.copy()
    
    return data


def split_data_two_sets(data):
    X_d = data.drop(columns="target")
    X   = preprocessing.scale(X_d)
    y   = data.target
    x_train,  x_test,  y_train,  y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train,  x_test,  y_train,  y_test ,x_train2, x_test2, y_train2, y_test2
    