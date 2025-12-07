import numpy as np
from math import *

class GaussianDigitClassifier:

    def __init__(self, num_of_features = 64, num_of_classes = 10):

        self.num_of_features = num_of_features
        self.num_of_classes = num_of_classes
        
        self.MeanMatrix = np.zeros((num_of_classes, num_of_features))
        self.CovarianceMatrix = np.zeros((num_of_features, num_of_features))
        self.CovarianceMatrixRegularized = self.CovarianceMatrix.copy()
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
            print(f"Covariance Matrix calculated successfully:\n {self.CovarianceMatrix}")

    def regularizeCovariance(self, lamda = 2, verbose = True):
        
        self.CovarianceMatrixRegularized = self.CovarianceMatrix.copy()
        for i in range(self.num_of_features):
            self.CovarianceMatrixRegularized[i,i] += lamda

        if verbose:
            print(f"Covariance Matrix after regularization:\n {self.CovarianceMatrixRegularized}")

    def predict(self, X, verbose = True):
        
        y_predicted = np.zeros(len(X))

        j = 1
        # print(y_predicted.shape)
        for i, sample in enumerate(X):
            y_predicted[i] = self.predictSample(sample)
            
            if verbose and (j%100 == 0):
                print(f"{i+1} samples done")
            j+=1

        return y_predicted
    
    def predictSample(self, x):
                
        prediction = -1
        maxPosterior = -1000
        for i, prior in enumerate(self.priors):
            logPos = self.calculateLogPosterior(x, i, verbose=False)
            if logPos > maxPosterior:
                maxPosterior = logPos
                prediction = i
        return prediction


    def calculateLogPosterior(self, x, c, verbose = True):
        a = -(self.num_of_features/2) * log1p(2*pi)
        # print(a)
        b = -0.5*log1p(np.linalg.det(self.CovarianceMatrixRegularized))
        # print(b)
        x_sample_minus_mean = x - self.MeanMatrix[c]
        h_temp = -0.5*np.matmul(x_sample_minus_mean, np.linalg.inv(self.CovarianceMatrixRegularized))
        # print(h.shape)        
        i = np.dot(h_temp, x_sample_minus_mean)
        # print(i)
        j = log1p(self.priors[c])

        ans = a+b+i+j
        if verbose:
            print(f"The Log of the posterior log(p(t|x))\
 is calculated successfully without including the p(x) term:\n {ans}")

        return ans

    def getMeanMatrix(self):
        return self.MeanMatrix
    
    def getCovarianceMatrix(self):
        return self.CovarianceMatrix
    
    def getRegularizedCovariancMatrix(self):
        return self.CovarianceMatrixRegularized


if __name__ == '__main__':
    print(np.size([True, False]))
