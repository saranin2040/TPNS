import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_csv(filename):
    # Задаём имена столбцов согласно описанию датасета
    column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
        'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    df = pd.read_csv(filename, header=None, names=column_names)
    
    # Замена "?" на NaN для обработки пропущенных значений
    df.replace('?', np.nan, inplace=True)
    
    for column in df.columns:
        if df[column].isnull().any():
            # Получение массива уникальных значений без NaN
            unique_values = df[column].dropna().unique()
            # Вычисление вероятности каждого уникального значения
            probabilities = df[column].value_counts(normalize=True)
            probabilities = probabilities.loc[unique_values].values
            # Заполнение пропущенных значений
            df[column] = df[column].apply(lambda x: np.random.choice(unique_values, p=probabilities) if pd.isnull(x) else x)



    df.drop_duplicates(inplace=True)  # Удаление дубликатов

    # Преобразование категориальных переменных в числовые индикаторы
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]

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
    print(df.iloc[-10:, :13])
    #print(df.iloc[count-10:count])

def save_to_csv(data, filename):
    data.to_csv(filename, index=False)

filename = 'data/mushroom.data'
#filename = 'data/data.csv'
data = load_and_clean_csv(filename)
matCorr = data.corr()



#data['Price'] = data['Price'] // 5000 * 5000

showTable(data,10)
showCorrelation(matCorr)

data=removeCorrCols(data,matCorr,0.8)

showCorrelation(data.corr())

save_to_csv(data, 'data/mushroomClear.data')
 
#gainRatios = getGainRatios(data, 'Price')

#sGainRatios = sorted(gainRatios.items(), key=lambda x: x[1], reverse=True)

#for atrb, gr in sGainRatios:
   # print(f"Gain Ratio for {atrb}: {gr}")

    #"""