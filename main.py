#predict stock prices usiong ML models
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import keras.models
import keras.layers
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#get stock data for facebook
company = 'TSLA'
df = web.DataReader(company, data_source='yahoo', start='2016-01-01', end='2021-01-14')
#print first few rows
# print(df.head(5))
# print(df.shape)

#get adj. closed price
df = df[['Adj Close']]
#create a data frame with only the 'Close' column
data = df.filter(['Adj Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.85)
# print(training_data_len)
#print(df.head())

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# print("Scaled Data: ", scaled_data)

#Create the training data set
#scaled first
train_data = scaled_data[:training_data_len , :]
#split the data xtrain and ytrain
x_train = []
y_train = []
for i in range(30, len(train_data)):
    x_train.append(train_data[i-30:i, 0])
    y_train.append(train_data[i, 0])

#convert trains to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data frame
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)

#build LSTM
model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(keras.layers.LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)


#Create testing dataset
#create a new array
test_data = scaled_data[training_data_len-30: , :]
#create x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(30, len(test_data)):
    x_test.append(test_data[i-30:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root means squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print("RMSE: ", rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the model
plt.figure(figsize=(16,8))
plt.title('Predicting Stock Price')
plt.xlabel('Date', fontsize=18)
plt.ylabel('COST', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Training Data', 'Actual Close Price', 'Predicted Close Price'], loc='lower right')
plt.show()

#Get the quote
stock_quote = web.DataReader(company, data_source='yahoo', start='2016-01-01', end='2021-01-21')
new_df = stock_quote.filter(['Adj Close'])
last_30_days = new_df[-30:].values
#scale the data
last_30_days_scaled = scaler.transform(last_30_days)
#create an pty list
x_test = []
x_test.append(last_30_days_scaled)
#convert x_test to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#get the predicted scaled price
pred_price = model.predict(x_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Price USD ($): ", pred_price)

#Actual price
stock_quote2 = web.DataReader(company, data_source='yahoo', start='2021-01-14', end='2021-01-14')
print("Actual Stock Details:\n", stock_quote2['Adj Close'])