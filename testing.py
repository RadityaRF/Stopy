from tkinter import *
import tkinter
from numpy.lib.polynomial import roots
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime
import math
plt.style.use('fivethirtyeight')

## DEKLARASI VARIABLE
start = datetime.datetime(2011,1,1)
end = datetime.datetime.today()

## DEKLARASI FUNGSI SAHAM
def bca():
    #Get the stock quote
    df= web.DataReader('BBCA.JK', 'yahoo', start=start, end=end)
    df.loc
    #DGet the number of rows and columns in the data set
    df.shape
    #Create new dataframe with only the 'Close column'
    data=df.filter(['Close'])
    #Convert the dataframe into a numpy array
    dataset= data.values
    #Get the number of rows to train the model on
    train_data_len= math.ceil(len(dataset)*.8)
    train_data_len
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)
    scaled_data
    #Create the training data set
    #Create the scaled training data set
    train_data=scaled_data[0:train_data_len, :]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train=[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train into a numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model= Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data test
    # Create new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[train_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    #convert the data into a numpy array
    x_test= np.array(x_test)
    #Reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price values
    predictions= model.predict(x_test)
    predictions= scaler.inverse_transform(predictions)
    #Get the root mean squared error (RMSE)
    rmse= np.sqrt(np.mean(predictions- y_test)**2)
    print('Root Mean Square Error :', rmse)

    #Plot the data
    train= data[:train_data_len]
    valid= data[train_data_len:]
    valid['Predictions']= predictions
    #Visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Prediksi Saham ')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize= 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Prediksi'], loc= 'lower right')
    plt.show()

    #Show the valid and predicted prices
    print(valid)
    return
def bni():
    #Get the stock quote
    df= web.DataReader('BBNI.JK', 'yahoo', start=start, end=end)
    df.loc
    #DGet the number of rows and columns in the data set
    df.shape
    #Create new dataframe with only the 'Close column'
    data=df.filter(['Close'])
    #Convert the dataframe into a numpy array
    dataset= data.values
    #Get the number of rows to train the model on
    train_data_len= math.ceil(len(dataset)*.8)
    train_data_len
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)
    scaled_data
    #Create the training data set
    #Create the scaled training data set
    train_data=scaled_data[0:train_data_len, :]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train=[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train into a numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model= Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data test
    # Create new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[train_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    #convert the data into a numpy array
    x_test= np.array(x_test)
    #Reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price values
    predictions= model.predict(x_test)
    predictions= scaler.inverse_transform(predictions)
    #Get the root mean squared error (RMSE)
    rmse= np.sqrt(np.mean(predictions- y_test)**2)
    print('Root Mean Square Error :', rmse)

    #Plot the data
    train= data[:train_data_len]
    valid= data[train_data_len:]
    valid['Predictions']= predictions
    #Visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Prediksi Saham ')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize= 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Prediksi'], loc= 'lower right')
    plt.show()

    #Show the valid and predicted prices
    print(valid)
    return
def btn():
        #Get the stock quote
    df= web.DataReader('BBTN.JK', 'yahoo', start=start, end=end)
    df.loc
    #DGet the number of rows and columns in the data set
    df.shape
    #Create new dataframe with only the 'Close column'
    data=df.filter(['Close'])
    #Convert the dataframe into a numpy array
    dataset= data.values
    #Get the number of rows to train the model on
    train_data_len= math.ceil(len(dataset)*.8)
    train_data_len
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)
    scaled_data
    #Create the training data set
    #Create the scaled training data set
    train_data=scaled_data[0:train_data_len, :]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train=[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train into a numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model= Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data test
    # Create new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[train_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    #convert the data into a numpy array
    x_test= np.array(x_test)
    #Reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price values
    predictions= model.predict(x_test)
    predictions= scaler.inverse_transform(predictions)
    #Get the root mean squared error (RMSE)
    rmse= np.sqrt(np.mean(predictions- y_test)**2)
    print('Root Mean Square Error :', rmse)

    #Plot the data
    train= data[:train_data_len]
    valid= data[train_data_len:]
    valid['Predictions']= predictions
    #Visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Prediksi Saham ')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize= 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Prediksi'], loc= 'lower right')
    plt.show()

    #Show the valid and predicted prices
    print(valid)
    return
def mandiri():
        #Get the stock quote
    df= web.DataReader('BMRI.JK', 'yahoo', start=start, end=end)
    df.loc
    #DGet the number of rows and columns in the data set
    df.shape
    #Create new dataframe with only the 'Close column'
    data=df.filter(['Close'])
    #Convert the dataframe into a numpy array
    dataset= data.values
    #Get the number of rows to train the model on
    train_data_len= math.ceil(len(dataset)*.8)
    train_data_len
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)
    scaled_data
    #Create the training data set
    #Create the scaled training data set
    train_data=scaled_data[0:train_data_len, :]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train=[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train into a numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model= Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data test
    # Create new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[train_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    #convert the data into a numpy array
    x_test= np.array(x_test)
    #Reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price values
    predictions= model.predict(x_test)
    predictions= scaler.inverse_transform(predictions)
    #Get the root mean squared error (RMSE)
    rmse= np.sqrt(np.mean(predictions- y_test)**2)
    print('Root Mean Square Error :', rmse)

    #Plot the data
    train= data[:train_data_len]
    valid= data[train_data_len:]
    valid['Predictions']= predictions
    #Visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Prediksi Saham ')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize= 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Prediksi'], loc= 'lower right')
    plt.show()

    #Show the valid and predicted prices
    print(valid)
    return
def bri():
    #Get the stock quote
    df= web.DataReader('BBRI.JK', 'yahoo', start=start, end=end)
    df.loc
    #DGet the number of rows and columns in the data set
    df.shape
    #Create new dataframe with only the 'Close column'
    data=df.filter(['Close'])
    #Convert the dataframe into a numpy array
    dataset= data.values
    #Get the number of rows to train the model on
    train_data_len= math.ceil(len(dataset)*.8)
    train_data_len
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data= scaler.fit_transform(dataset)
    scaled_data
    #Create the training data set
    #Create the scaled training data set
    train_data=scaled_data[0:train_data_len, :]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train=[]
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train into a numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    #Build the LSTM model
    model= Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    #Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data test
    # Create new array containing scaled values from index 1543 to 2003
    test_data = scaled_data[train_data_len - 60:, :]
    # create the data sets x_test and y_test
    x_test = []
    y_test = dataset[train_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    #convert the data into a numpy array
    x_test= np.array(x_test)
    #Reshape the data
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price values
    predictions= model.predict(x_test)
    predictions= scaler.inverse_transform(predictions)
    #Get the root mean squared error (RMSE)
    rmse= np.sqrt(np.mean(predictions- y_test)**2)
    print('Root Mean Square Error :', rmse)

    #Plot the data
    train= data[:train_data_len]
    valid= data[train_data_len:]
    valid['Predictions']= predictions
    #Visualize the model
    plt.figure(figsize=(16,8))
    plt.title('Prediksi Saham ')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize= 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Prediksi'], loc= 'lower right')
    plt.show()

    #Show the valid and predicted prices
    print(valid)
    return

## GUI
win = tkinter.Tk()
win.geometry('300x550')
win.title('Stopy')


## Label
l1 = Label(win, text='Prediksi Saham Bank di Indonesia\nDengan Metode LSTM\n(Long - Short Term Memory',
        font=('Times New Roman', 14),
        justify='left',
        bg='#334257',
        fg='white'
        )
l1.pack(pady=20)
l2 = Label(win, text='Daftar Bank :',
        font=('Times New Roman', 14),
        bd=1,
        justify='right',
        bg='#334257',
        fg='white' 
        )
l2.pack(pady=20)

## Button
b1 = Button(win, text='Bank BCA', 
        command=bca, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        bg='#334257',
        fg='white',
        width=10
        )
b1.pack(pady=15)
b2 = Button(win, text='Bank Mandiri', command=mandiri, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        width=10,
        bg='#334257',
        fg='white')
b2.pack(pady=15)
b3 = Button(win, text='Bank BNI', command=bni, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        width=10,
        bg='#334257',
        fg='white')
b3.pack(pady=15)
b4 = Button(win, text='Bank BRI', command=bri, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        width=10,
        bg='#334257',
        fg='white')
b4.pack(pady=15)
b5 = Button(win, text='Bank BTN', command=btn, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        width=10,
        bg='#334257',
        fg='white')
b5.pack(pady=15)
b6 = Button(win, text='Keluar', command=win.quit, 
        font=('Times New Roman', 12),
        bd=1,
        justify='right',
        width=10,
        bg='#334257',
        fg='white')
b6.pack(pady=15)

win.configure(
    background='#476072',)
win.iconbitmap('Stopy.ico')
win.mainloop()

