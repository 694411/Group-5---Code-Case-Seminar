import numpy as np
from model import create_lstm_model
from keras.callbacks import EarlyStopping
import datetime 

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")


#### DEFINE HYPERPARAMETERS
dropout_rate = 0.1
l2_reg = 0.0001
learning_rate = 0.001
batch_size = 32

input_shape = (X_train.shape[1], X_train.shape[2]) 
model = create_lstm_model(input_shape, dropout_rate, l2_reg, learning_rate)

model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=[EarlyStopping(monitor='val_f1_score', patience=15, restore_best_weights=True, mode = 'max')])

today = datetime.date.today().strftime("%Y-%m-%d")
model.save(f"{today}-lstm-model-trained-256-128-{dropout_rate}-{l2_reg}-{learning_rate}-{batch_size}.keras")
