import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_two_features(X, y, feat1, feat2):

    f1 = X[:, feat1]
    f2 = X[:, feat2]

    classes = np.unique(y)

    plt.figure(figsize=(8, 6), dpi=200)

    alpha = 1
    linewidths = 1
    for c in classes:
        idx = (y == c)
        plt.scatter(
            f1[idx], 
            f2[idx],
            alpha=alpha,
            s=30,
            label=f"Class {c}",
            linewidths=linewidths

        )
        alpha-=0.1
        linewidths +=1

    plt.xlabel(f"Feature {feat1}")
    plt.ylabel(f"Feature {feat2}")
    plt.title(f"Feature {feat1} vs Feature {feat2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
