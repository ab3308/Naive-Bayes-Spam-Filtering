import numpy as np

class MyClassifier:
        
    def train(self, train_data, train_labels):
        
        #numpy array of length 2 to store log class priors
        self.log_class_priors = np.zeros(2) 
        self.log_class_priors[0]=np.log(np.count_nonzero(train_labels == 0)/len(train_labels))
        self.log_class_priors[1]=np.log(np.count_nonzero(train_labels == 1)/len(train_labels))   
    
        #getting number of words and number of messages in training data
        n_features = len(train_data[0])
        n_samples = len(train_data)

        #getting totals of number of spam and ham messages
        n_spam_messages = np.count_nonzero(train_labels==1)
        n_ham_messages = np.count_nonzero(train_labels==0)

        #saving theta values to self, allows access in other functions
        #pre-defined in numpy array of format: 2 x number of features, allowing us to track theta value for
        #each class-word combination
        self.theta = np.zeros(((2),(n_features)))

        hamArray = []
        spamArray = []

        for i in range(n_samples):
            #loops through training set, separating training data into spam and ham messages
            if(train_labels[i]==0):
                hamArray.append(train_data[i])
            else:
                spamArray.append(train_data[i])

        #getting total number of words in ham and spam messages
        #corresponds to nc in formula
        n_ham_words = np.count_nonzero(hamArray == 1)    
        n_spam_words = np.count_nonzero(spamArray == 1)
        
        #gets total wordcount for each word, given the class
        #corresponds to n c,w in formula
        wordcount_given_ham = np.sum(hamArray, axis = 0)
        wordcount_given_spam = np.sum(spamArray, axis = 0)

        for i in range(n_features):
            #loops through all words, calculating and saving theta values
            self.theta[0][i] = np.log((wordcount_given_ham[i] + 1)/(n_ham_words + n_features*1))
            self.theta[1][i] = np.log((wordcount_given_spam[i] + 1)/(n_spam_words + n_features*1))


    def predict(self, test_data):
        
        #define numpy array of zeros of length (number of messages)
        class_predictions = np.zeros(len(test_data))
    
        for j in range(len(test_data)):
            #loops through all messages
            
            #sets initial p(ham or spam) to corresponding log class prior - means we don't have to add it later
            p_ham = self.log_class_priors[0]
            p_spam = self.log_class_priors[1]
            
            for i in range(len(test_data[0])):
                #loops through all features in message of test data
                #adds probability of word being in class to total probability
                p_ham += test_data[j][i]*self.theta[0][i]
                p_spam += test_data[j][i]*self.theta[1][i]

            if(p_ham<p_spam):
                #if more likely to be spam, change corresponding 0 to 1
                class_predictions[j] = 1
            
            #if p(ham)=p(spam), I chose to leave it as ham as this is more functional in real-world use
            
        #returns numpy array of class predictions (0 or 1) for each message in test data
        return class_predictions

spam_classifier = MyClassifier()
spam_classifier.train(training_spam[:, 1:], training_spam[:, 0])