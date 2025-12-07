import numpy as np
import math

class GaussianDigitClassifier:

    def __init__(self, num_of_features = 64, num_of_classes = 10):

        self.num_of_features = num_of_features
        self.num_of_classes = num_of_classes
        
        self.MeanMatrix = np.zeros((num_of_classes, num_of_classes))
        self.CovarianceMatrix = np.zeros((num_of_features, num_of_features))
        self.priors = np.zeros(num_of_classes)

    def calculateClassPriors(self, y, verbose = True):
        total_num_of_smaples = np.size(y)
        
        for i in range(len(self.priors)):
            class_labels = np.where(y==i)
            num_of_samples = np.size(class_labels)
            self.priors[i] = num_of_samples/total_num_of_smaples
            if verbose:
                print(f"number of {i} samples: {num_of_samples} with prior probability: {self.priors[i]}")
        # print(self.priors.sum())
        if verbose:
            print("\n\n=========================\n\n")

    def calculateMeans(self, x, verbose = True):
        



if __name__ == '__main__':
    print(np.size([True, False]))
