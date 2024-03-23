from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from keras import backend as K

def accuracy_score(y_true, y_pred, threshold = 0.5): 
    mask_padding_value = -9999  
    mask = K.cast(K.not_equal(y_true, mask_padding_value), K.floatx())  

    y_pred_thresholded = K.cast(K.greater_equal(y_pred, threshold), K.floatx())
    
    correct_predictions = K.sum(K.cast(K.equal(y_true, y_pred_thresholded), K.floatx()) * mask)
    num_non_masked = K.sum(mask)

    accuracy = K.switch(K.greater(num_non_masked, 0), correct_predictions / num_non_masked, K.zeros_like(correct_predictions))

    return accuracy

def f1_score(y_true, y_pred, threshold = 0.5):
    mask_padding_value = -9999
    mask = K.cast(K.not_equal(y_true, mask_padding_value), K.floatx())
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    y_pred_thresholded = K.cast(K.greater_equal(y_pred, threshold), K.floatx())

    def recall(y_true, y_pred_thresholded):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred_thresholded, 0, 1)) * mask)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)) * mask)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred_thresholded):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred_thresholded, 0, 1)) * mask)
        predicted_positives = K.sum(K.round(K.clip(y_pred_thresholded, 0, 1)) * mask)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred_thresholded)
    recall_value = recall(y_true, y_pred_thresholded)
    f1 = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))

    num_non_masked = K.sum(mask)
    return K.switch(K.greater(num_non_masked, 0), f1, K.zeros_like(f1))

def preprocess_for_auc(y_true, y_pred):
    mask_value = -9999
    mask = y_true != mask_value

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    return y_true_filtered, y_pred_filtered

def calculate_auc_roc(y_true, y_pred):
    y_true_filtered, y_pred_filtered = preprocess_for_auc(y_true, y_pred)
    y_true_filtered = y_true_filtered.astype("int")

    return roc_auc_score(y_true_filtered, y_pred_filtered)

def calculate_auc_pr(y_true, y_pred):
    y_true_filtered, y_pred_filtered = preprocess_for_auc(y_true, y_pred)
    y_true_filtered = y_true_filtered.astype("int")

    precision, recall, _ = precision_recall_curve(y_true_filtered, y_pred_filtered)
    return auc(recall, precision)