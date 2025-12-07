import numpy as np
from decisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timeit

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None, random_state=None):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []             # list of DecisionTree instances
        self.feature_indices = []   # list of arrays: which feature columns each tree saw
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        n_samples, n_features = X.shape

        for _ in range(self.n_trees):

            row_idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[row_idxs]
            y_sample = y[row_idxs]

            # Choose feature subset for this tree
            if self.max_features is None:
                feat_idxs = np.arange(n_features)
            else:
                feat_idxs = np.random.choice(n_features, self.max_features, replace=False)

            # Train a DecisionTree on the reduced feature set (no changes to DecisionTree)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feat_idxs], y_sample)

            # Save tree and its feature mapping
            self.trees.append(tree)
            self.feature_indices.append(feat_idxs)

    def predict(self, X):
        if not self.trees:
            raise ValueError("The RandomForest instance is not fitted yet.")

        # Collect predictions from each tree (shape: n_trees x n_samples)
        all_preds = []
        for tree, feat_idxs in zip(self.trees, self.feature_indices):
            preds = tree.predict(X[:, feat_idxs])
            all_preds.append(preds)

        all_preds = np.vstack(all_preds)  # shape (n_trees, n_samples)

        # Majority vote per sample
        final_preds = []
        for col in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, col], return_counts=True)
            final_preds.append(vals[np.argmax(counts)])

        return np.array(final_preds)

if __name__ == "__main__":
    timer = timeit.default_timer
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)
    n_trees_values = [5, 10, 30, 50]
    d = X_train.shape[1]

    max_features_values = [
        int(np.sqrt(d)),
        d // 2
    ]

    best_rf_acc = -1
    best_rf_params = None

    for n_trees in n_trees_values:
        for mf in max_features_values:
            rf = RandomForest(
                n_trees=n_trees,
                max_depth=4,
                min_samples_split=2,
                max_features=mf
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            acc = np.mean(preds == y_val)

            print(f"Trees={n_trees}, max_features={mf}, val_acc={acc:.4f}")

            if acc > best_rf_acc:
                best_rf_acc = acc
                best_rf_params = (n_trees, mf)


    best_n_trees, best_mf = best_rf_params

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    final_rf = RandomForest(
        n_trees=best_n_trees,
        max_depth=4,
        min_samples_split=2,
        max_features=best_mf
    )

    final_rf.fit(X_trainval, y_trainval)
    rf_preds = final_rf.predict(X_test)

    print(f"Random Forest Test Accuracy:{np.mean(rf_preds == y_test):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))