from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)
from sklearn.base import clone

def evaluate_model(model, datasets, threshold=0.5):
    """
    Evaluate a trained binary classifier across internal validation,
    external validation, and test datasets.

    Computes Average Precision, ROC-AUC, Precision, Recall, F1,
    Balanced Accuracy, and positive-class prevalence.

    Parameters
    ----------
    model : classifier
        Trained model supporting predict_proba().
    datasets : dict
        Dictionary containing X and y for each dataset split.
    threshold : float, default=0.5
        Probability cutoff for predicting the positive class.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for each dataset split.
    """
    results = []

    X_train, y_train = datasets["train"]["X"], datasets["train"]["y"]
    cloned_model = clone(model)
    cloned_model.fit(X_train, y_train)
    
    for data_split in ["int_val", "ext_val", "test"]:
        X = datasets[data_split]["X"]
        y = datasets[data_split]["y"]

        # Predicted probabilities
        y_probs = cloned_model.predict_proba(X)[:, 1]
        # Convert probabilities to binary predictions
        y_pred = (y_probs >= threshold).astype(int)

        # Calculate metrics
        results.append({
            "Dataset": data_split,
            "Prevalence": y.mean(),
            "Average_Precision": average_precision_score(y, y_probs),
            "ROC_AUC": roc_auc_score(y, y_probs),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y, y_pred)
        })
    return pd.DataFrame(results)
