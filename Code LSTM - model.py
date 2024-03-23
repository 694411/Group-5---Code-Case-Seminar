from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Masking
from keras.regularizers import l2
from keras.optimizers import Adam
from metrics import f1_score

def create_lstm_model(input_shape, dropout_rate, l2_reg, learning_rate):
    model = Sequential()
    model.add(Masking(mask_value=-9999, input_shape=input_shape))
    model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_dropout = dropout_rate))
    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(l2_reg), recurrent_dropout = dropout_rate))
    model.add(TimeDistributed(Dense(units=1, activation='sigmoid', kernel_regularizer=l2(l2_reg))))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1_score])

    return model