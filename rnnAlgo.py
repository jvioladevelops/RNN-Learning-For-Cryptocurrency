import pandas as pd
from sklearn import preprocessing
import os
from collections import deque
import random
import numpy as np

#df = pd.read_csv("crypto_data/LTC-USD.csv", names=["time", "low", "high", "open", "close", "volume"])

'''
We need sequence and targets for RNN
'''

#define Sequence ie: constant variables

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1 #1 is a good thing
    else:
        return 0 

def preprocess_df(df):
    df = df.drop('future', 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) #scale data to between 0 and 1
            
    df.dropna(inplace = True)
    
    sequential_data = []
    
    prev_days = deque(maxLen = SEQ_LEN) #
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            
    random.shuffle(sequential_data)
        
            
    
main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD" ]
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"] )

    df.rename(columns = {"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace = True)
    
    df.set_index("time", inplace = True)
    
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)    
        
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
    
#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))
    #print(df.head())
times = sorted(main_df.index.values)

last_5pct = times[-int(0.05)*len(times)] #thresshold last 5 percent

#print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)] #anywhere that timestamp s greater tahn 5 % value

main_df = main_df[(main_df.index < last_5pct)]

#train_x, train_y = preprocess_df(main_df)
#train_x, train_y = preprocess_df(main_df)
