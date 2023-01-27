import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, List
import tensorflow.keras.backend as K



class Preprocesing:
    
    def __init__(self, conf):
        self.batch_size=conf["batch_size"]
        self.seq_len=conf["seq_len"]
        self.start_timestamp=conf["start_timestamp"]
        self.end_timestamp=conf["end_timestamp"]
        self.val_ptc=conf["val_ptc"]
        self.test_ptc=conf["test_ptc"]
        self.file_1=conf["file_1"]
        self.scaler_ohlc=conf["scaler_ohlc"]
        self.cols = conf["cols"]
        self.one_hot=conf["one_hot"]
        self.nan_rows = []

    def scale_and_save(self, data: pd.DataFrame, cols_stan: List[str], cols_norm: List[str], scaler_filename: str):
        if cols_stan:
            scaler_stan = StandardScaler()
            data[cols_stan] = scaler_stan.fit_transform(data[cols_stan])
            joblib.dump(scaler_stan, scaler_filename+'_stan') 
        if cols_norm:
            scaler_norm = MinMaxScaler(feature_range=(0, 1))
            data[cols_norm] = scaler_norm.fit_transform(data[cols_norm])
            joblib.dump(scaler_norm, scaler_filename+'_norm')
        return data


    def update_new_position(self, df, current_index, value):
        for i in range(1, 24):
            if df.loc[current_index-i, 'position'] != -value:
                df.loc[current_index-i, 'new_position'] = value
            else:
                break


    def load_data(self, path: str, cols: List[str]) -> pd.DataFrame:
        data = pd.read_csv(path, index_col='timestamp')
        data["timestamp"]=data.index
        col1_counts = data['position'].value_counts()
        col1_percent = col1_counts / data.shape[0] * 100
        print(col1_percent)
        mask = (data.index >= self.start_timestamp) & (data.index < self.end_timestamp)
        data = data[mask][cols]
        if not self.one_hot:
            data = data[data.position != 0]
            data.loc[data.position == -1, :] = 0
        return data

    def data_oversample(self,df):
        ts = df.index
        df=df.reset_index(drop=True)
        df['new_position'] = df['position']
        for i in range(len(df)):
            if df.loc[i, 'position'] == 1:
                self.update_new_position(df, i, 1)
            elif df.loc[i, 'position'] == -1:
                self.update_new_position(df, i, -1)
        df=df.drop(["position"],axis=1)
        df = df.rename(columns={'new_position': 'position'})#we should print the oversample percentages
        df.index=ts
        df = df.iloc[1:]
        col1_counts = df['position'].value_counts()
        col1_percent = col1_counts / df.shape[0] * 100
        print(col1_percent)
        return df


    def transform_data(self, data: pd.DataFrame, cols_stan: List[str], cols_norm: List[str], scaler_filename: str) -> pd.DataFrame:
        self.nan_rows = data[data.isna().any(axis=1)]
        print(len(self.nan_rows))
        if not self.one_hot:
            data = self.scale_and_save(data, cols_stan, cols_norm, scaler_filename)
        else:
            data = pd.get_dummies(data, columns=['position'])
            data = self.scale_and_save(data, cols_stan, cols_norm, scaler_filename)
        return data

    def data_splitting(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        times = data.index.values
        last_10pct = times[-int(self.test_ptc*len(times))] 
        last_20pct = times[-int(self.val_ptc*len(times))]
        last_10pct_plus_seqlen = times[-int(self.test_ptc*len(times))-self.seq_len]
        last_20pct_plus_seqlen = times[-int(self.val_ptc*len(times))-self.seq_len] 

        train_num = data[data.index < last_20pct]
        val_num = data[(data.index >= last_20pct_plus_seqlen) & (data.index < last_10pct)]
        test_num = data[data.index >= last_10pct_plus_seqlen]

        train_num_df = train_num.copy()
        val_num_df = val_num.copy()
        test_num_df = test_num.copy()

        # Remove print statements
        self.ts_forcallback = test_num_df.index[0]

        # Convert pandas columns into arrays
        train_num = train_num.values
        val_num = val_num.values
        test_num = test_num.values
        if self.one_hot:
            train_num_labels = train_num[:, -4:]
            val_num_labels = val_num[:, -4:]
            test_num_labels = test_num[:, -4:]
            return train_num, val_num, test_num, train_num_labels, val_num_labels, test_num_labels, train_num_df, val_num_df, test_num_df
        else:
            return train_num, val_num, test_num, train_num_df, val_num_df, test_num_df

    def sequence_creator(self, data):
        X_0, y1 = [], []
        for i in range(self.seq_len, len(data)):
            X_0.append(data[i-self.seq_len:i])
            if not self.one_hot:
                y1.append(data[i, -1])
            else:
                labels = data[i, -4:]
                y1.append(labels)
        X_0 = np.array(X_0)
        y1 = np.array(y1)
        return X_0, y1

    def dataset_gen(self, df, targets):
        """Generates dataset (X, Y) for training"""
        
        x_0 = tf.constant([x for x in df["x_0"].values])
        X = {"input_0": x_0}
        Y = {"output_1": targets}
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset
        
