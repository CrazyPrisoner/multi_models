import time
import numpy
import pandas
import matplotlib.pyplot as plt

def input_data():
    df = pandas.read_csv('./sinos.csv')
    df.drop('date',axis=1,inplace=True)
    df.dropna(inplace=True)
    # Graphics
    plt.plot(df.values[:,-1], 'red');
    plt.title("Sin Value trend");
    plt.show()
    divide = int(len(df)*0.8) # value for divide data on train and test
    df.dropna(inplace=True)
    print(df)
    print(df.head(10),"\n") # print first 10 raws
    print(df.info(),"\n") # print info about dataframe
    print(df.shape,"\n") # print dataframe shape
    print(df.describe(),"\n") # print info about values
    train_X = numpy.array(df.values[:divide,:])
    test_X = numpy.array(df.values[divide:,:])
    x = numpy.array(df.values[:,:])

    return x, train_X, test_X
