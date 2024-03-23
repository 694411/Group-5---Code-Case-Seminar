import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

path = '/Users/char/Downloads/df_lstm.csv'
df_lstm = pd.read_csv(path)

df_lstm['period'] = pd.to_datetime(df_lstm['period'])
averaged_df = df_lstm.groupby(['unique_id', 'period'], as_index=False).mean()
averaged_df = averaged_df.drop(columns=['id_electricity', 'id_gas'])

groups = averaged_df.groupby('unique_id')
max_length = max(groups.apply(lambda x: len(x)))

feature_columns = averaged_df.columns.difference(['unique_id', 'period', 'final_churn'])
target_column = 'final_churn'

X_seq = []
y_seq = []

for name,group in groups: 
    sequence_x = group[feature_columns].values
    padded_x = pad_sequences([sequence_x], maxlen=max_length, padding='pre', dtype='float32', value=-9999)

    X_seq.append(padded_x.squeeze())

    sequence_y = group[target_column].values
    padded_y = pad_sequences([sequence_y], maxlen=max_length, padding='pre', dtype='float32', value=-9999)

    y_seq.append(padded_y.reshape(-1,1))

X = np.stack(X_seq)
y = np.stack(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("data/X_train", X_train)
np.save("data/X_test", X_test)
np.save("data/y_train", y_train)
np.save("data/y_test", y_test)