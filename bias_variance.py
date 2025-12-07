import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from randomForest import RandomForest
from decisionTree import DecisionTree

"""
this is the comparison between the Decision Trees and Random Forest Performance
the output is:
Decision Tree: Bias^2: 0.0542, Variance: 0.0458, Accuracy: 0.9186
Random Forest: Bias^2: 0.0739, Variance: 0.0232, Accuracy: 0.8953

Bias:
The random forest exhibits slightly higher bias compared to the single decision tree. 
This is expected because averaging predictions across multiple trees “smoothens” the model,
making it less flexible and slightly less able to perfectly fit the training data.
The single decision tree, being more flexible, 
has lower bias and can fit the training data closely.


Variance:
The variance of the random forest is significantly lower than that of the single tree.
Single decision trees are sensitive to the specific training data they see, 
causing their predictions to fluctuate widely if the training set changes. 
Random forests reduce this instability by averaging over many trees trained on different 
bootstrap samples, resulting in more consistent predictions.

Bias² = error of average prediction
bias squared is average of square of all errors basically

Variance = instability across different models
Variance is mean of all Squared differences between predictions and their mean

"""

def estimate_bias_variance(model_class, X_train, y_train, X_test, y_test, n_models=10, **kwargs):
    all_preds = []

    n_samples = len(X_train)

    # Train multiple models on bootstrap samples
    for _ in range(n_models):
        print(f"{_+1}/{n_models}")
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample = X_train[idxs]
        y_sample = y_train[idxs]

        model = model_class(**kwargs)
        model.fit(X_sample, y_sample)
        preds = model.predict(X_test)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # shape = (n_models, n_test_samples)

    # Compute mean prediction per sample
    # For binary labels 0/1, mean is fraction of models predicting 1
    mean_preds = np.mean(all_preds, axis=0)

    # Bias^2 = (true - mean_pred)^2
    bias2 = np.mean((y_test - mean_preds) ** 2)

    # Variance = mean of variance across models for each sample
    variance = np.mean(np.var(all_preds, axis=0))

    # Overall accuracy of average model (optional)
    avg_pred_labels = (mean_preds >= 0.5).astype(int)
    acc = np.mean(avg_pred_labels == y_test)

    return bias2, variance, acc



if __name__ == "__main__":
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    bias2_dt, var_dt, acc_dt = estimate_bias_variance(
        model_class=DecisionTree,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_models=20,
        max_depth=4,
        min_samples_split=2
    )

    # Random Forest
    bias2_rf, var_rf, acc_rf = estimate_bias_variance(
        model_class=RandomForest,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_models=20,
        n_trees=30,
        max_depth=4,
        min_samples_split=2,
        max_features=int(np.sqrt(X_train.shape[1])),
    )


    print(f"Decision Tree: Bias^2: {bias2_dt:.4f}, Variance: {var_dt:.4f}, Accuracy: {acc_dt:.4f}")
    print(f"Random Forest: Bias^2: {bias2_rf:.4f}, Variance: {var_rf:.4f}, Accuracy: {acc_rf:.4f}")
