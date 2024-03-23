import pandas as pd 
import numpy as np
from model import create_lstm_model
from sklearn.model_selection import KFold
from metrics import f1_score
from keras.callbacks import EarlyStopping


X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

# sample_indices = np.random.choice(X_train.shape[0], 100, replace=False)

hyperparameters = {
    'dropout_rate': [0, 0.1, 0.2, 0.3],
    'l2_reg': [0.0001, 0.001],
    'learning_rate': [0.001, 0.01],
    'batch_size':[32, 64, 128]
}

n_splits = 3
kf = KFold(n_splits=3)

best_score =  -np.inf
best_params = None


for dropout_rate in hyperparameters['dropout_rate']:
    for l2_reg in hyperparameters['l2_reg']:
        for learning_rate in hyperparameters['learning_rate']:
            for batch_size in hyperparameters['batch_size']:
                thresholds = np.linspace(0.01, 0.99, 99)
                f1_scores = {k: {threshold: [] for threshold in thresholds} for k in range(1, n_splits+1)}
                scores = []

                print(f"dropout_rate: {dropout_rate}, l2_reg: {l2_reg}, learning_rate: {learning_rate}, batch_size: {batch_size}")
                
                k = 1
                
                for train_index, val_index in kf.split(X_train): 
                    print(f"kfold: {k}/{n_splits}")
                    X_t, X_v = X_train[train_index], X_train[val_index]
                    y_t, y_v = y_train[train_index], y_train[val_index]
                    
                    input_shape = (X_t.shape[1], X_t.shape[2])
                    model = create_lstm_model(input_shape, dropout_rate, l2_reg, learning_rate)
                    print("Fitting model ...")
                    history = model.fit(X_t, y_t, epochs=50, batch_size=batch_size, validation_data=(X_v, y_v), verbose=1, callbacks=[EarlyStopping(monitor='val_f1_score', patience=7, restore_best_weights=True, mode ='max')])

                    y_pred = model.predict(X_v)

                    for threshold in thresholds:
                        f1 = f1_score(y_v.squeeze(), y_pred.squeeze(), threshold)
                        f1_scores[k][threshold].append(f1.numpy())
                    
                    k = k + 1

                threshold_sums = {threshold: 0 for threshold in f1_scores[1].keys()}
                total_ks = len(f1_scores)
                for k, thresholds in f1_scores.items():
                    for threshold, scores in thresholds.items():
                        threshold_sums[threshold] += sum(scores) / len(scores) # Averaging scores per threshold per k
                average_f1_scores = {threshold: sum_ / total_ks for threshold, sum_ in threshold_sums.items()}

                max_f1_threshold = max(average_f1_scores, key=average_f1_scores.get)
                max_f1_score = average_f1_scores[max_f1_threshold]

                scores.append(max_f1_score)
                params = {'dropout_rate': dropout_rate, 'l2_reg': l2_reg, 'learning_rate': learning_rate, 'batch_size': batch_size} 
                
                print("End of k-fold loop")
                print(f"Parameter setting: {params}")
                print(f"F1-score: {max_f1_score} with optimal threshold {max_f1_threshold}")

                if max_f1_score > best_score:
                    best_score = max_f1_score
                    best_params = {'dropout_rate': dropout_rate, 'l2_reg': l2_reg, 'learning_rate': learning_rate, 'batch_size': batch_size} 

print(f"\nEnd of hyperparameter training. \nBest f1_score: {best_score}. \nBest parameter setting: {best_params}")