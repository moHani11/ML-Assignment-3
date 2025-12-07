import numpy as np
import math

class GaussianDigitClassifier:

    def __init__(self, num_of_features = 64, num_of_classes = 10):

        self.num_of_features = num_of_features
        self.num_of_classes = num_of_classes
        
        self.MeanMatrix = np.zeros((num_of_classes, num_of_classes))
        self.CovarianceMatrix = np.zeros((num_of_features, num_of_features))
        self.priors = np.zeros(num_of_classes)

    def calculateClassPriors(self, y):
        total_num_of_smaples = np.size(y)







