import numpy as np

class CustomNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.priors = {}
        self.likelihoods = {}
        self.unseen_log_probs = {} # To store log prob for unseen feature values
        self.feature_counts = [] # stores number of unique values per feature

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.classes = np.unique(y)
        
        # calculating priors P(c_k)
        for c in self.classes:
            # (count of samples in class k + alpha) / (total samples + alpha * num_classes)
            num_samples_in_class = np.sum(y == c)
            self.priors[c] = np.log((num_samples_in_class + self.alpha) / (num_samples + self.alpha * len(self.classes)))
            
            
            self.likelihoods[c] = []
            self.unseen_log_probs[c] = []
            x_c = x[y == c]
            # calculating likelihoods P(x_ i | c_k)
            for feature in range(num_features):
                # calculate the values for "feature"
                num_feature_values = len(np.unique(x[:, feature]))

                # calculate the number of unique values of the feature
                if len(self.feature_counts) <= feature:
                    self.feature_counts.append(num_feature_values)
                else:
                    num_feature_values = self.feature_counts[feature]

                # count occurrences of each value i in class k
                occurrences_count = np.bincount(x_c[:, feature], minlength=num_feature_values)
                
                # (count feature val i in class k + alpha) / (count class k + alpha * num_feature_values)
                prob = (occurrences_count + self.alpha) / (num_samples_in_class + (self.alpha * num_feature_values))
                
                self.likelihoods[c].append(np.log(prob))
                # logs are used for stability


                # Calculate and store the log probability for an unseen feature value
                unseen_prob = self.alpha / (num_samples_in_class + (self.alpha * num_feature_values))
                self.unseen_log_probs[c].append(np.log(unseen_prob))

    
    def calculate_log_probabilities(self, x):
        n_samples, n_features = x.shape

        # 2D matrix to hold the score for each sample and each class
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            prior = self.priors[c]
            
            # addinf likelihoods (adding logs is the same as multiplying the probabilities)
            class_likelihood = np.zeros(n_samples)
            
            for feature_index in range(n_features):
                # get values for the current feature from all the input samples.
                sample_feature_values = x[:, feature_index]

                # for handling new feature categories
                max_val = len(self.likelihoods[c][feature_index]) - 1
                
                # Create an array for the likelihoods of the samples' feature values
                feature_likelihoods = np.zeros(n_samples)

                # Identify seen and unseen feature values
                unseen_mask = sample_feature_values > max_val
                seen_mask = ~unseen_mask

                # Get likelihoods for seen values
                if np.any(seen_mask):
                    seen_values = sample_feature_values[seen_mask]
                    feature_likelihoods[seen_mask] = self.likelihoods[c][feature_index][seen_values]

                # Get likelihoods for unseen values
                if np.any(unseen_mask):
                    feature_likelihoods[unseen_mask] = self.unseen_log_probs[c][feature_index]

                class_likelihood += feature_likelihoods
            
            log_probs[:, idx] = prior + class_likelihood
            
        return log_probs
    

    def predict(self, x) :
        log_probs = self.calculate_log_probabilities(x)
        return self.classes[np.argmax(log_probs, axis=1)]

    def calculate_class_probabilities(self, x):
        log_probs = self.calculate_log_probabilities(x)
        probs = np.exp(log_probs)
        return probs / np.sum(probs, axis=1, keepdims=True)