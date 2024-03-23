import numpy as np 
from metrics import f1_score, accuracy_score, calculate_auc_roc, calculate_auc_pr
import keras
import pandas as pd
from sklearn.metrics import confusion_matrix


X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

path_model = "/Users/char/Repositories/EUR/MasterThesis/seminar/models/lstm-model-trained_256-128-0.2-0.0001-0.001-32.keras"
threshold = 0.25

model = keras.models.load_model(path_model, custom_objects={'f1_score': f1_score, 'accuracy_score': accuracy_score})

y_pred = model.predict(X_test)

y_test_flattened = y_test.flatten()
y_pred_flattened = y_pred.flatten()
y_pred_binary = (y_pred_flattened > threshold).astype(int)


print(f"Accuracy: {accuracy_score(y_test_flattened,y_pred_flattened, threshold)}")
print(f"F1 Score: {f1_score(y_test_flattened,y_pred_flattened, threshold)}")
print(f"AUC-ROC: {calculate_auc_roc(y_test_flattened,y_pred_flattened)}")
print(f"AUC-PR: {calculate_auc_pr(y_test_flattened,y_pred_flattened)}")

mask = y_test_flattened != -9999
y_test_filtered = y_test_flattened[mask]
y_pred_filtered = y_pred_flattened[mask]
y_pred_binary_filtered = y_pred_binary[mask]

conf_matrix = confusion_matrix(y_test_filtered.astype(int), y_pred_binary_filtered)

print("Confusion Matrix:")
print(conf_matrix)

data = {'y_test': y_test_filtered, 'y_test_pred_proba': y_pred_filtered}
df = pd.DataFrame(data)
df.to_csv('probabilities_lstm.csv', index=False)
