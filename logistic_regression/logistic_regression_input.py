import numpy
import pandas
import matplotlib.pyplot as plt
from mlxtend.preprocessing import one_hot

def import_data():
    dataframe = pandas.read_csv('./Dataset1.csv')
    return dataframe

def info_data():
    data = import_data()
    print(data.head(10),"\n") # print first 10 raws
    print(data.info(),"\n") # print info about dataframe
    print(data.shape,"\n") # print dataframe shape
    print(data.describe(),"\n") # print info about values
    print(data.corr(),"\n") # dataframe correlation

    # Convert object type to float type
    data.NGP=pandas.to_numeric(pandas.Series(data.NGP), errors='coerce');
    data.EGT=pandas.to_numeric(pandas.Series(data.EGT), errors='coerce');
    data.WF=pandas.to_numeric(pandas.Series(data.WF), errors='coerce');
    data.dropna(inplace=True) # Drop all NaN values

    # Graphics
    plt.plot(data['NGP'])
    plt.ylabel('Parameter N')
    plt.show()
    plt.plot(data['EGT'],color='red')
    plt.ylabel('Parameter EGT')
    plt.show()
    plt.plot(data['WF'],color='green')
    plt.ylabel('Parameter WF')
    plt.show()

def input_data():
    df = import_data() # import dataframe
    divide = 3584 # value for divide data on train and test
    df.drop('dateandtime',inplace=True,axis=1)
    # Convert object type to float type
    df.NGP=pandas.to_numeric(pandas.Series(df.NGP), errors='coerce');
    df.EGT=pandas.to_numeric(pandas.Series(df.EGT), errors='coerce');
    df.WF=pandas.to_numeric(pandas.Series(df.WF), errors='coerce');
    df.dropna(inplace=True)
    # Divide data on train and test without shuffle
    train_X = numpy.array(df.values[:divide,0:3])
    train_Y_p = numpy.array(df.values[:divide,3:])
    test_X = numpy.array(df.values[divide:,0:3])
    test_Y_p = numpy.array(df.values[divide:,3:])
    # Peapare one hot encode
    train_Y_p = train_Y_p.astype('int') # Convert to int type (train data)
    test_Y_p = test_Y_p.astype('int') # Convert to int type (test data)
    trf = train_Y_p.ravel() # Need be dimension 1, to encode in one hot
    tref = test_Y_p.ravel() # Need be dimension 1, to encode in one hot
    traf_Y = one_hot(trf, num_labels=3) # num_labels need be same as your classes (0,1,2)
    tres_Y = one_hot(tref, num_labels=3) # num_labels need be same as your classes (0,1,2)
    train_Y_en = numpy.array(traf_Y) # one hot numpy array (train data)
    test_Y_en = numpy.array(tres_Y) # one hot numpy array (test data)
    print(train_Y_en)

    # It is one hot:
    #_______________________________
    #|__good__|_anomaly_|_anomaly1_|
    #|____1___|____0____|_____0____|
    #|____0___|____1____|_____0____|
    #|____0___|____0____|_____1____|

    return train_X, test_X, train_Y_en, test_Y_en
