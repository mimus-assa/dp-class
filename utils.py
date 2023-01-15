class Preprocesing:
    '''esta clase contiene las herramientas para crear 
    el dataset a partir de las configuraciones'''
    
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
         
    def loading_data(self, path, cols):
        """Reads csv file and filters, sets timestamp as index, and selects relevant columns."""
        data = pd.read_csv(path, index_col='timestamp')
        data = data[(data.index >= self.start_timestamp) & (data.index < self.end_timestamp)]
        data = data[cols]
        #data = data[data.position != 0]
        #data.loc[data.position == -1, :] = 0
        return data

    def transform_data(self, data, cols_stan, cols_norm, scaler_filename):
        '''Transforms the columns that are desired to be transformed.'''
        # Standardize columns
        scaler_stan = StandardScaler()
        if len(cols_stan)!=0:
            data[cols_stan] = scaler_stan.fit_transform(data[cols_stan])
            joblib.dump(scaler_stan, scaler_filename+'_stan') 
        # Normalize columns
        scaler_norm = MinMaxScaler(feature_range=(0, 1))
        if len(cols_norm)!=0:
            data[cols_norm] = scaler_norm.fit_transform(data[cols_norm])
            joblib.dump(scaler_norm, scaler_filename+'_norm') 
        return data

    def data_splitting(self, data):
        """Splits data into train, validation, and test sets, also returns copies of pandas dataframes."""
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
        
        print("train dataset start on timeseries: ", train_num_df.index[0])
        print("train dataset ends on timeseries: ", train_num_df.index[-1])
        print("val dataset start on timeseries: ", val_num_df.index[0])
        print("val dataset ends on timeseries: ", val_num_df.index[-1])
        print("test dataset start on timeseries: ", test_num_df.index[0])
        print("test dataset ends on timeseries: ", test_num_df.index[-1])
        self.ts_forcallback = test_num_df.index[0]
        print("train dataset shape: ", train_num_df.shape)
        print("val dataset shape: ", val_num_df.shape)
        print("test dataset shape: ", test_num_df.shape)
        
        # Convert pandas columns into arrays
        train_num = train_num.values
        val_num = val_num.values
        test_num = test_num.values
        return train_num, val_num, test_num, train_num_df, val_num_df, test_num_df


    def sequence_creator(self, datas):
        '''cambiamos cada entrada por una secuencia de 
        las ultimas seq_len filas y despues incluimos 
        las salidas con indice seq_len+1 '''
        data1=pd.DataFrame(datas[0])
        X_0, y1= [],[]
        for i in range(self.seq_len, len(data1)):
            X_0.append(data1.reset_index(drop=True).iloc[i-self.seq_len:i])
            y1.append(data1.iloc[:, -1].reset_index(drop=True).iloc[i])
        X_0   = np.array(X_0)
        y1  = np.array(y1).reshape((-1,1))
        return X_0, y1
  
    #crear un dataset exploration
    def dataset_gen(self, df, targets):
        """Generates dataset (X, Y) for training"""
        targets = tf.one_hot(targets, 3) # one-hot encode the targets
        x_0 = tf.constant([x for x in df["x_0"].values])
        X = {"input_0": x_0}
        Y = {"dense_1": targets}
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset


    def create_dataset(self):
        X_0 = self.loading_data(self.file_1, self.cols)
        X_0 = self.normalize_data(X_0, self.cols, self.scaler_ohlc)
        train_0, val_0, test_0, _, _, test_num2 = self.data_splitting(X_0)

        # train dataset
        X0_train, y1_train = self.sequence_creator([train_0])
        df = pd.DataFrame({"x_0": X0_train})
        train_data = self.dataset_gen(df, [y1_train])

        # val dataset
        X0_val, y1_val = self.sequence_creator([val_0])
        df = pd.DataFrame({"x_0": X0_val})
        val_data = self.dataset_gen(df, [y1_val])

        # test dataset
        X0_test, y1_test = self.sequence_creator([test_0])
        df = pd.DataFrame({"x_0": X0_test})
        test_data = self.dataset_gen(df, [y1_test])

        return train_data, val_data, test_data
    def dataset_exploration(self, train_data, val_data, test_data):
        '''Function to explore the datasets generated by the create_dataset function'''
        train_size = 0
        val_size = 0
        test_size = 0
        for data in train_data:
            train_size += data[0]['input_0'].numpy().nbytes
            shape = data[0]['input_0'].numpy().shape
            dtypes = data[0]['input_0'].dtype
            target_shape = data[1]['dense_1'].numpy().shape
            target_dtypes = data[1]['dense_1'].dtype
            mean = tf.reduce_mean(data[0]['input_0']).numpy()
            std = tf.math.reduce_std(data[0]['input_0']).numpy()
            min_val = tf.math.reduce_min(data[0]['input_0']).numpy()
            max_val = tf.math.reduce_max(data[0]['input_0']).numpy()
        for data in val_data:
            val_size += data[0]['input_0'].numpy().nbytes
        for data in test_data:
            test_size += data[0]['input_0'].numpy().nbytes
        total_mem = train_size + val_size + test_size
        print(f"Total memory usage: {total_mem / (1024 ** 2):.2f}MB")
        print(f"Train data shape: {shape}")
        print(f"Validation data shape: {shape}")
        print(f"Test data shape: {shape}")
        print(f"Data types: {dtypes}")
        print(f"Target shape: {target_shape}")
        print(f"Target dtypes: {target_dtypes}")
        print(f"Data range: Min: {min_val}, Max: {max_val}")
        print(f"Data mean: {mean}")
        print(f"Data standard deviation: {std}")
