import csv
import numpy as np
from sklearn import linear_model

def linear_regression_sklearn(x,y,new_x):

    # input array
    r_x = np.array(x).reshape(len(x),1)
    r_y = np.array(y)
    r_nx = np.array(new_x).reshape(len(new_x),1)

    # sklearn linear_model predict
    reg = linear_model.LinearRegression().fit(r_x, r_y)
    r_ny = reg.predict(r_nx)

    out = open('test-A\out.tsv', 'w')
    for nyi in r_ny:
        out.write(str(nyi)+'\n')
        print(nyi)

def linear_regression_self(x, y, new_x):

    # input array
    r_x = np.array(x).reshape(len(x),1)
    r_y = np.array(y)
    r_nx = np.array(new_x).reshape(len(new_x),1)
    
    # number of observations/points 
    n = np.size(r_x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(r_x), np.mean(r_y) 

    # mean square
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (r_x[i]-m_x)*(r_y[i]-m_y)
        denominator += (r_x[i]-m_x)**2

    b1 = numerator/denominator
    b0 = m_y-(b1*m_x)

    out = open('test-A\out.tsv', 'w')
    for nxi in r_nx:
        nyi = (b1*nxi)+b0
        out.write(str(nyi)[1:-1]+'\n')
        print(str(nyi)[1:-1])


def import_data():

    reader = open('train/train.tsv', encoding="utf8")
    x = list()
    y = list()
    for line in reader:
        line = line.split()
        if line[0] != 'cena':
            x.append(float(line[1]))
            y.append(float(line[0]))
    
    xe = list()
    examples = open('test-A\in.tsv', encoding="utf8")
    for line in examples:
        line = line.split()
        xe.append(float(line[0]))

    return (x,y,xe)

def main():
    
    import_xy = import_data()
    x = import_xy[0]
    y = import_xy[1]
    nx = import_xy[2]

    linear_regression_self(x,y,nx)
    #linear_regression_sklearn(x,y,nx)
  

if __name__== "__main__":
    
    main()
