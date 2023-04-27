import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
yf.pdr_override()
start = (dt.datetime.today() - dt.timedelta(days=365*10)).strftime('%Y-%m-%d') # calculate start date 10 years ago
end = dt.datetime.today().strftime('%Y-%m-%d')

st.title("Stock Price Prediction App")

user_input = st.text_input("Enter Company name", "apple")

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()
    if len(data['quotes']) == 0:
        return "Sorry couldn't find any ticker for that company"
    company_code = data['quotes'][0]['symbol']
    return company_code

company_code = getTicker(user_input)
if company_code == "Sorry couldn't find any ticker for that company":
    st.write("Sorry couldn't find any ticker for that company")
else:
    df = pdr.get_data_yahoo(company_code, start, end)
    st.subheader("Data from 10 years ")
    st.write(df.describe())

    # Visualization
    st.subheader("Closing Price vs Time Chart")
    fig= plt.figure(figsize=(12,6))
    plt.plot(df['Close'])
    st.pyplot(fig)

    st.subheader("Closing Price vs Time Chart with 100 rolling average")
    ma100 = df.Close.rolling(100).mean()
    fig= plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df['Close'])
    st.pyplot(fig)

    st.subheader("Closing Price vs Time Chart with 200 rolling average")
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig= plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df['Close'])
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    scaler = MinMaxScaler(feature_range=(0,1))

    model = load_model('./keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing,ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test=[]
    y_test = []
    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    x_test,y_test = np.array(x_test),np.array(y_test)

    # Predictions
    y_pred = model.predict(x_test)  

    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    # scale_factor = 1/scaler.scale_[0]
    # y_pred = y_pred * scale_factor
    # y_test = y_test * scale_factor

    # Final Graph
    st.subheader("Predictions vs Original")
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_pred,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    # Todays predicted price and actual price
    st.subheader("Todays predicted price and actual price")
    st.write(f"Todays predicted price is: {y_pred[-1][0]}")
    st.write(f"Todays actual price is: {y_test[-1][0]}")
    # st.write(input_data[-1,0],df['Close'][-1],final_df['Close'][len(final_df)-1])

    latest_price = df['Close'][-1:].values.reshape(-1,1)
    latest_price_scaled = scaler.fit_transform(latest_price)
    next_day_input = np.append(x_test[-1][1:], latest_price_scaled)
    next_day_input = next_day_input.reshape(1, 100, 1)

    next_day_pred = model.predict(next_day_input)
    next_day_pred = scaler.inverse_transform(next_day_pred)

    st.write(f"Next day predicted price is: {next_day_pred[0][0]}")