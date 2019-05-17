from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pandas

def naive_bayes_sklearn(x, y, nx):

    # scaling
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    scaled_nx = scaler.fit_transform(nx)

    # sklearn_naive_bayes
    gaussian_nb = GaussianNB()
    gaussian_nb.fit(scaled_x, y)
    GaussianNB(priors=None)

    ny = gaussian_nb.predict(scaled_nx)

    output = open(r'test-A/out1.tsv', 'w')
    for yi in ny:
        print(yi)
        output.write(yi + '\n')
    output.close()


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

    naive_bayes_sklearn(x, y, nx)
    

if __name__== "__main__":
    
    main()
