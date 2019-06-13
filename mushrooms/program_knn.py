from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
import pandas

def knn_sklearn(x, y, nx):

    # scaling
    scaler = Normalizer().fit(x)
    scaled_x = scaler.fit_transform(x)
    scaled_nx = scaler.fit_transform(nx)

    # sklearn_knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(scaled_x, y)
    ny = knn.predict(scaled_nx)

    output1 = open(r'test-A/out.tsv', 'w')
    for yi in ny:
        print(yi)
        output1.write(yi + '\n')
    output1.close()


def import_data():

    train_data = pandas.read_csv('train/train.tsv', sep='\t')
    train_array = train_data.values
    test_data = pandas.read_csv('test-A/in.tsv', sep='\t')
    nx = test_data.values

    x = train_array[:, 1:23]
    y = train_array[:, 0]

    for i in range(len(x[0])):
        for row in x:
            row[i] = float(ord(row[i]))
    for i in range(len(nx[0])):
        for row in nx:
            row[i] = float(ord(row[i]))

    return (x, y, nx)


def main():
    
    import_xy = import_data()
    x = import_xy[0]
    y = import_xy[1]
    nx = import_xy[2]

    knn_sklearn(x, y, nx)
    

if __name__== "__main__":
    
    main()
