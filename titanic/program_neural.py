import pandas
from sklearn.neural_network import MLPClassifier

def neural_network_sklearn(x, y, nx):    

    neuralNetwork = MLPClassifier(solver='lbfgs', max_iter=100000000, random_state=3, early_stopping=True, learning_rate='adaptive', alpha=0.000001)
    model = neuralNetwork.fit(x,y)

    ny = neuralNetwork.predict(nx)

    output = open(r'test-A/out.tsv', 'w')
    for yi in ny:
        print(yi)
        output.write(str(yi) + '\n')
    output.close()

def normalize(a):
    a = a.fillna(0)
    a['Sex'] = a['Sex'].replace('male',1)
    a['Sex'] = a['Sex'].replace('female',0)
    a['Embarked'] = a['Embarked'].replace('S',1)
    a['Embarked'] = a['Embarked'].replace('Q',2)
    a['Embarked'] = a['Embarked'].replace('C',3)

    return a

def import_data():

    columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    train_data = pandas.read_csv('train/train.tsv', sep='\t', header=0, usecols=['Survived','PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    x = train_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    x = normalize(x)
    y = train_data['Survived']
    
    test_data = pandas.read_csv('test-A/in.tsv', sep='\t', header=None, names=columns)
    nx = test_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    nx = normalize(nx)
    
    return (x, y, nx)


def main():
    
    import_xy = import_data()
    x = import_xy[0]
    y = import_xy[1]
    nx = import_xy[2]

    neural_network_sklearn(x, y, nx)
    

if __name__== "__main__":
    
    main()
