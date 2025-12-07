import numpy as np
import math

class GaussianDigitClassifier:

    def __init__(self, num_of_features = 64, num_of_classes = 10):

        self.num_of_features = num_of_features
        self.num_of_classes = num_of_classes
        
        self.MeanMatrix = np.zeros((num_of_classes, num_of_features))
        self.CovarianceMatrix = np.zeros((num_of_features, num_of_features))
        self.priors = np.zeros(num_of_classes)
        self.total_num_of_samples = 0


    def calculateClassPriors(self, y, verbose = True):
        self.total_num_of_samples = np.size(y)
        
        for i in range(len(self.priors)):
            class_labels = np.where(y==i)
            num_of_samples = np.size(class_labels)
            self.priors[i] = num_of_samples/self.total_num_of_samples
            if verbose:
                print(f"number of {i} samples: {num_of_samples} with prior probability: {self.priors[i]}")
        # print(self.priors.sum())
        if verbose:
            print("\n\n=========================\n\n")


    def calculateMeans(self, X, y, verbose = True):

        for i in range((len(self.MeanMatrix))):
            class_samples = X[y == i]
            features_summation = class_samples.sum(axis=0)
            self.MeanMatrix[i] = features_summation/np.size(class_samples)
            # print(len(self.MeanMatrix[0]))

        if verbose:
            print("Mean Matrix is filled successfully")
            print("\n\n=========================\n\n")


    def calculateCovarianceMatrix(self, X, y, verbose = True):

        for i, sample in enumerate(X):

            sample_label = y[i]
            feature_minus_mean = sample - self.MeanMatrix[sample_label]
            self.CovarianceMatrix += np.outer(feature_minus_mean, feature_minus_mean)
            # print(feature_minus_mean.shape) 
        self.CovarianceMatrix /= self.total_num_of_samples
        
        if verbose:
            print(f"Covariance Matrix calculated successfully: {self.CovarianceMatrix}")

    def regularizeCovariance(self, lamda = 2, verbose = True):
        
        for i in range(self.num_of_features):
            self.CovarianceMatrix[i,i] += lamda

        if verbose:
            print(f"Covariance Matrix after regularization: {self.CovarianceMatrix}")


if __name__ == '__main__':
    print(np.size([True, False]))
