import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_csv(filename):

    df = pd.read_csv(filename)
    
    df.dropna(inplace=True)
    
    df.drop_duplicates(inplace=True)

    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR

        df.loc[df[column] < lowerBound, column] = df[column].median()
        df.loc[df[column] > upperBound, column] = df[column].median()
    
    df = pd.get_dummies(df, drop_first=False)
    #df['Brand'] = pd.factorize(df['Brand'])[0]

    return df

def removeCorrCols(data, matCorr, threshold):
    columnsToRemove = set()
    for i in range(len(matCorr.columns)):
        for j in range(i+1, len(matCorr.columns)):
            if abs(matCorr.iloc[i, j]) > threshold:
                col_name = matCorr.columns[j]
                columnsToRemove.add(col_name)
    data = data.drop(columns=columnsToRemove)
    return data

def showCorrelation(corr):
    
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Тепловая матрица корреляции')
    plt.show()

def showTable(df,count):
    print(df.iloc[0:count])

def getInfo(target):
    vals, counts = np.unique(target, return_counts=True)
    totalСount = np.sum(counts)
    info = -np.sum([(counts[i]/totalСount) * np.log2(counts[i]/totalСount) for i in range(len(vals))])
    return info

def getInfoX(data,atrb,target):
    vals, counts = np.unique(data[atrb], return_counts=True)
    totalСount = np.sum(counts)
    infoX = np.sum([(counts[i]/totalСount) * getInfo(data.where(data[atrb]==vals[i]).dropna()[target]) for i in range(len(vals))])
    return infoX

def getInfoGain(data, atrb, target):

    info = getInfo(data[target])
    infoX = getInfoX(data,atrb,target)

    infoGain = info - infoX
    return infoGain

def getSplitInfo(data, atrb):
    vals, counts = np.unique(data[atrb], return_counts=True)
    totalСount = np.sum(counts)
    splitInfo = -np.sum([(counts[i]/totalСount) * np.log2(counts[i]/totalСount) for i in range(len(vals))])
    return splitInfo

def getGainRatio(data, atrb, target):
    infoGain = getInfoGain(data, atrb, target)
    splitInfo = getSplitInfo(data, atrb)
    if splitInfo == 0:
        return 0
    return infoGain / splitInfo

def getGainRatios(data, target):
    gainRatios = {}
    for atrb in data.columns:
        if atrb != target:
            gainRatios[atrb] = getGainRatio(data, atrb, target)
    sGainRatios = sorted(gainRatios.items(), key=lambda x: x[1], reverse=True)

    #for atrb, gr in sGainRatios:
        #print(f"Gain Ratio for {atrb}: {gr}")
    return gainRatios
#"""

filename = 'data/Laptop_price.csv'
#filename = 'data/data.csv'
data = load_and_clean_csv(filename)
matCorr = data.corr()

#data=removeCorrCols(data,matCorr,0.8)

data['Price'] = data['Price'] // 5000 * 5000

showTable(data,10)
showCorrelation(matCorr)
 
gainRatios = getGainRatios(data, 'Price')

sGainRatios = sorted(gainRatios.items(), key=lambda x: x[1], reverse=True)

for atrb, gr in sGainRatios:
    print(f"Gain Ratio for {atrb}: {gr}")

    #"""