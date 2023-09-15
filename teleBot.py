import time
import pandas as pd
import numpy as np
import pickle
import telebot
import requests
from datetime import datetime, timedelta
from telebot import types
import psutil
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# Data Collection - Parse last 30 days prices of BTC
def get_trades(endtime, symbol = 'BTC', limit=31, interval='1d'):
    r = requests.get("https://api.binance.com/api/v1/klines",
        params = {
            "symbol": symbol + 'USDT',
            "limit": limit,
            "endTime": endtime,
            "interval": interval
            })
    return r.json()

# DataFrame
df = pd.DataFrame(get_trades(int(time.time()*1000)),
    columns = ['OpenTime', 'open', 'high', 'low', 'close', 'volume', 'CloseTime', 'bav', 'not', 'tbv', 'tbBav', 'ignore'])
print('DataFrame is ready', '\n')



# Preprocessing
def create_dateTime(x):
    res = f"{datetime.fromtimestamp(x/1000).year}-{datetime.fromtimestamp(x/1000).month}-{datetime.fromtimestamp(x/1000).day}"
    res = pd.to_datetime(res)
    return res

df['year_month_day'] = df['OpenTime'].apply(lambda x: create_dateTime(x))

# Feature Engeenering
df_for_model = df[['year_month_day', 'close']].iloc[:-1].set_index('year_month_day')
df_for_model['close'] = pd.to_numeric(df_for_model['close'])

# Prediction
def prediction(df_for_model):
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(df_for_model.close).reshape(-1,1))

    closedf = closedf.reshape(1,30,1)

    filename = 'LSTM_BTC_Predict.sav'

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(closedf)

    lm_predict = scaler.inverse_transform(result)[0][0]
    return lm_predict

# Last 30 days trend
def c_chart(data,label):
    candlestick = go.Figure(data = [go.Candlestick(x=data.index,
                                                open = data['open'], 
                                                high = data['high'], 
                                                low = data['low'], 
                                                close = data['close'])])
    candlestick.update_xaxes(title_text = 'Time',
                            rangeslider_visible = True)

    candlestick.update_layout(
    title = {
            'text': '{:} Candelstick Chart'.format(label),
            "y":0.8,
            "x":0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    candlestick.update_yaxes(title_text = 'Price in USD', ticksuffix = '$')
    return candlestick

# Save the graph as an image
# image_path = f'BTC_Candelstick_Chart_{str(df.iloc[-1].year_month_day).split(" ")[0]}.png'
# fig = c_chart(df.loc[:30], label="BTC Price")
# pio.write_image(fig=fig, file='file.png', format="png", engine="auto")

image_path_plt = f'BTC_Chart_{str(df.iloc[-1].year_month_day).split(" ")[0]}_plt.png'
lm_predict = prediction(df_for_model)
plt.plot(range(1, 11), df_for_model['close'].iloc[-10:])
plt.plot([10, 11], [df_for_model['close'].iloc[-1], lm_predict], color = 'red')
plt.savefig(image_path_plt)


# Messages

bot = telebot.TeleBot(token='6140805983:AAH5-3ZoNkd5OkPcUZLgkeLTgbM7vcfglI0')

users = []

@bot.message_handler(commands=['start'])
def start_message(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton('Prediction Price')
    button2 = types.KeyboardButton('Last 30 days\' candle chart')
    button3 = types.KeyboardButton('Last 10 days trend with prediction price')
    button4 = types.KeyboardButton('Actual price')

    markup.add(button1, button2, button3, button4)

    bot.send_message(message.chat.id, "Welcome! You will receive periodic predictions for the BTC price. What do you want to see?",
                     reply_markup=markup)



@bot.message_handler(content_types=['text'])
def get_text(message):
    if message.text == 'Prediction Price':
        #prct_change = round(lm_predict/float(df.iloc[-1].close), 4) * 100 - 100
        bot.send_message(message.chat.id, 
                         f'Prediction close price of BTC for {str(df["year_month_day"].iloc[-1]).split(" ")[0]} is {round(lm_predict, 2)}.')
    
    elif message.text == 'Actual price':
        bot.send_message(message.chat.id, 
                         f'Actual price equal to {df.iloc[-1].close}.Difference between next close price and actual price is {lm_predict - float(df.iloc[-1].close)} USDT')

    elif message.text == 'Last 30 days\' candle chart':
        with open('BTC_Candles.png', 'rb') as photo:
            bot.send_photo(chat_id=message.chat.id, photo=photo)
        pass

    elif message.text == 'Last 10 days trend with prediction price':
        with open(image_path_plt, 'rb') as photo:
            bot.send_photo(chat_id=message.chat.id, photo=photo)


    else:
        bot.send_message(message.chat.id, 'Sorry, I don\'t understand')


bot.polling(none_stop=True)


