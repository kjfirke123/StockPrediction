#!/usr/bin/env python
# coding: utf-8

# # Predicting NIFTY 50 USING LSTM MODEL

# ## The Nifty is the flagship benchmark of the National Stock Exchange (NSE), which is a well-diversified index, comprising top 50 companies in terms of free-float market capitalisation that are traded on the bourse. It is supposed to reflect the health of the listed universe of Indian companies, and hence the broader economy, in all market conditions

# In[1]:


import math
import pandas as pd
import numpy as np
import streamlit as st
from plotly import graph_objs as go
import scipy
from nsetools import Nse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('fivethirtyeight')
nse = Nse()


# In[2]:


#!pip install nsepy


# In[3]:

import datetime
from nsepy import get_history
from datetime import date



st.title("Stock Trend Prediction")


st.subheader("Today's data")
col1, col2, col3 = st.columns(3)
col1.write("Index Name")
col2.write("Last Price")
col3.write("Percent Change")
index_quote = nse.get_index_quote("nifty bank") 
nifty50 = nse.get_index_quote("nifty 50")
with col1:
    st.write(index_quote['name'])
    st.write(nifty50['name'])
    
with col2:
    st.write(index_quote['lastPrice'])
    st.write(nifty50['lastPrice'])
with col3:
    st.write(index_quote['pChange'])
    st.write(nifty50['pChange'])



top_gainers = nse.get_top_gainers()


st.subheader("Top  Gainers Today")



col1, col2, col3 = st.columns(3)
col1.write("Stock Name")
col2.write("Last Traded Price")
col3.write("Percent Change")

for i in range(3):
    stock_name=top_gainers[i]['symbol']
    stock_price=top_gainers[i]['ltp']
    last_change=top_gainers[i]['netPrice']

    with col1:
        st.write(stock_name)
    with col2:
        st.write(stock_price)
    with col3:
        st.write(last_change)




top_losers = nse.get_top_losers()



st.subheader("Top Losers today") 
col1, col2, col3 = st.columns(3)
col1.write("Stock Name")
col2.write("Last Traded Price")
col3.write("Percent Change")
for i in range(3):
    stock_name=top_losers[i]['symbol']
    stock_price=top_losers[i]['ltp']
    last_change=top_losers[i]['netPrice']

    with col1:
        st.write(stock_name)
    with col2:
        st.write(stock_price)
    with col3:
        st.write(last_change)










st.markdown("""
<style>
.big-font {
    font-size:150% !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Enter Stock Ticker</p>', unsafe_allow_html=True)


user_input = st.text_input(" ",'SBIN')
start_date = datetime.date.today() - datetime.timedelta(days=1825)

data=get_history(symbol=user_input,start=start_date,end = datetime.date.today(),index=0)
data.reset_index(inplace=True)

end_date = datetime.date.today() + datetime.timedelta(days=1)
start_date = datetime.date.today() - datetime.timedelta(days=120)

nifty_quote = get_history(symbol=user_input, start=start_date, end=end_date,index=0)

stock_start = datetime.date.today()- datetime.timedelta(days=2)
stock_end = datetime.date.today()
stock_data = get_history(symbol=user_input, start=stock_start, end=stock_end,index=0)

st.subheader(f"{user_input} Previous Data")
last_open=stock_data.Open[1:]
last_close=stock_data.Close[1:]
last_high=stock_data.High[1:]
last_low=stock_data.Low[1:]
todays_data = pd.concat([last_open,last_close],axis=1)
todays_data = pd.concat([todays_data,last_low],axis=1)
todays_data = pd.concat([todays_data,last_high],axis=1)
todays_data

stock = nse.get_quote(user_input)
stock_high=stock['high52']
stock_low=stock['low52']
col1, col2 = st.columns(2)
col1.write("52 Week High")
col2.write("52 Week Low")
with col1:
    stock_high=stock['high52']
    st.write(f'\u20B9{stock_high}')
with col2:
    stock_low=stock['low52']
    st.write(f'\u20B9{stock_low}')



# In[4]:

#Visualizations
st.subheader('Closing price vs Time')


#st.write("This graph shows historical closing price")
close_fig = go.Figure()
close_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price',showlegend = True))

close_fig.layout.update(xaxis_rangeslider_visible = True)
close_fig.layout.update(plot_bgcolor="#BBCCC9")
close_fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Closing Price",
    autosize=False,
    width=800,
    height=600,
    font=dict(
        family="Courier New, monospace",
        size=12,        
    )
    )
st.plotly_chart(close_fig)
st.markdown("""
<style>
.big-font {
    font-size:200% !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">This graph shows historical closing price</p>', unsafe_allow_html=True)

#close_fig.update_yaxes(automargin=True)



st.subheader('Closing Price vs Time chart with 50MA & 200MA')
ma_50 = data.Close.rolling(50).mean()
ma_200 = data.Close.rolling(200).mean()

pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
pred_fig.add_trace(go.Scatter(x=data['Date'], y=ma_50, name='50 Moving Average'))
pred_fig.add_trace(go.Scatter(x=data['Date'], y=ma_200, name='200 Moving Average')) 
pred_fig.layout.update(xaxis_rangeslider_visible = True)  

pred_fig.layout.update(plot_bgcolor="#DEDEDE")
pred_fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Stock Closing Price",

    autosize=False,
    width=800,
    height=600,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
        ),
    font=dict(
        family="Courier New, monospace",
        size=12,        
    )
    )
st.plotly_chart(pred_fig)

st.subheader("Inferring information based on Above graph")
st.write("1) If Red Line(50 Day Moving Average) cuts Green Line(200 Day Moving Average) upside then chances are stock price will go up")
st.write("2) If Red Line(50 Day Moving Average) cuts Green Line(200 Day Moving Average) downside then chances are stock price will go down")


# In[5]:
#Training and testing
close=data.filter(["Close"])
dataset = close.values
training_data_len=math.ceil(len(dataset)*0.60)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)



#Load my model
model = load_model('models.h5')
#X_test.append(last_60_days_scaled)
test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    

new_df=nifty_quote.filter(['Close'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)


pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)

st.subheader(f"Predicted Price for {user_input} on {end_date} is \u20B9{int(pred_price)}")

         




