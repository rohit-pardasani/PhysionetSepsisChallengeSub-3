#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import keras as ks
import keras

def get_sepsis_score(data):
    testModel = ks.models.load_model('SepCNNV4.h5')
    parMeanVarDf = pd.read_csv(r'nonSepStat.csv',sep=',')
    #print(np.shape(data))
    #print(data)
    df = pd.DataFrame(data, columns=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3',
    'FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',
    'Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
    'Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'])
    #print(df)
    df = df.drop(['Unit1','Unit2','HospAdmTime','ICULOS'],axis=1)
    parmList = list(df.columns.values)
    #print(parmList)
    #print(df)
    parmList.remove('Age')
    parmList.remove('Gender')
    #print(parmList)
    for varName in parmList:
        varMean = parMeanVarDf[parMeanVarDf['Parameter']==varName]['Mean'].values[0]
        df[varName] = df[varName].interpolate().fillna(method='bfill')
        df[varName] = df[varName].fillna(varMean)
    #print(df)
    loaded_scaler = joblib.load('my_scaler.pkl')
    X = df.values
    Xn = loaded_scaler.transform(X) 
    #print(Xn)
    CHANNELS = 36
    TOTAL_PTS = np.shape(df)[0]
    THRESH = 0.5
    Xf = np.zeros((1,TOTAL_PTS,CHANNELS), dtype=np.float32)
    Xf[0,:,:] = Xn
    Yp = testModel.predict(Xf).reshape(TOTAL_PTS,)
    scores = Yp.astype(np.float32)
    #print(scores)
    Yp[Yp>=THRESH] = 1
    Yp[Yp<THRESH] = 0
    wh = np.where(Yp==1)
    if(len(wh[0])!=0):
        firstOne = wh[0][0]
        Yp[firstOne:-1] = 1
    labels = Yp.astype(int)
    
    return (scores, labels)

def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

if __name__ == '__main__':
    # read data
    data = read_challenge_data(sys.argv[1])

    # make predictions
    if data.size != 0:
        (scores, labels) = get_sepsis_score(data)

    # write results
    with open(sys.argv[2], 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        if data.size != 0:
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))
