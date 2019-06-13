import csv
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

def neural_network_sklearn(x,y,nx):

    neuralNetwork = MLPRegressor(solver='lbfgs', alpha=0.0005)
    model = neuralNetwork.fit(x,y)

    ny = neuralNetwork.predict(nx)

    out = open('test-A\out.tsv', 'w')
    for nyi in ny:
        out.write(str(nyi)+'\n')
        print(nyi)

def normalize(dataset):
    
    dataset = dataset[['Powierzchnia w m2', 'Liczba pokoi', 'Rok budowy', 'Piętro']]
    dataset = dataset.replace({'parter':0, 'poddasze': 0}, regex = True)
    dataset['Piętro'].fillna(round(dataset['Piętro'].median(), 0), inplace = True)
    dataset['Rok budowy'].fillna(round(dataset['Rok budowy'].median(), 0), inplace = True)
    return dataset

def import_data():

    reader = pd.read_csv('train/train.tsv', delimiter ='\t')
    columns = reader.columns[1:]
    test = pd.read_csv('test-A/in.tsv', delimiter ='\t', header = None, names = columns)

    y = reader['cena']
    x = normalize(reader)
    nx = normalize(test)

    return (x,y,nx)

def main():
    
    import_xy = import_data()
    x = import_xy[0]
    y = import_xy[1]
    nx = import_xy[2]

    neural_network_sklearn(x,y,nx)
  

if __name__== "__main__":
    
    main()
