For this project, I was provided with supplementary code which formatted data for us. 
The task was as follows: 
- Your code must provide a variable with the name `spam_classifier`. 
- This object must have a method called `predict` which takes input data and returns class predictions. 
- The input will be a single n x 54 numpy array, your classifier should return a numpy array of length n with classifications.

## The problem

 - The problem is spam, and being able to distinguish it from normal emails. 
 - In this coursework, we refer to normal emails as ham (represented by 0's), while spam is unwanted (1's).
 - We are to take a n x 55 numpy array input to train our classifier. The first column here identifies whether it is spam or ham (using 0/1). This allows us to train our model. 
 - This classifier will then take a n x 54 numpy array, where each row represents a message. 
 - Each column represents a word in the message, represented by a 1 if present in the message. Otherwise, a 0. 
 - The classifier should try to best distinguish whether the row (message) is spam or ham, based on data from the training set.

## My Classifier
- For my classifier, I chose to implement Naive Bayes. 
- To summise, this is a supervised learning technique which gets it's name from the assumption that events are conditionally independent given a class.
- This means that the words themselves are independent of each other given the class (spam/ham)
- Furthermore, it utilises Bayes' law to calculate probabilities.
- Naive Bayes is a relatively simple algorithm, which executes fast - making it ideal for message classification.

## How the Naive Bayes classifier works

- As previously mentioned, Naive Bayes assumes the conditional independence of events.

- The equation for the classifier is shown below (source: classification.ipynb):

\begin{equation}
= \underset{c \in \{0,1\}}{\operatorname{argmax}} \ \left[ \log( {p(C=c)}) + \sum_{i = 1}^k w_i \ \log \left({\theta_{c, w_i}} \right) \right]
\end{equation}
	
	- As this equation shows, for each class, you calculate the log of the probability of the class occurring (log class priors). 
	- Then, for all words multiply the appearance of the word (0 or 1) by the log of the parameter theta. Sum these up.
	- Add these two values (log of probability + sum of all words by log parameters). 
	- It then takes the largest value out of all classes, and returns the corresponding class.
	
- To calculate the log of the probability of the class occurring, simply input the fraction of 
  (no. specific class occurrences) / (no. all class occurrences) into a log function for each class.

- wi represents the number 0 or 1, 1 if the word appears in the message

- Theta is calculated using the following formula (source: classification.ipynb): 
	
	\begin{equation}
{\theta_{c, w}} = \frac{n_{c, w} + \alpha}{n_{c} + k \alpha}
\end{equation}
	
	- Where nc,w represents the number of times the word appears in messages belonging to the given class
	- Alpha corresponds to the value one, for Laplace smoothing, used to nullify potential problems caused by zero probability.
	- nc represents the count of all words belonging to a given class
	- k represents the number of unique words (features)
    
## How I implemented Naive Bayes

### Calculating log class priors:
- For this, I began by defining a numpy array of length (number of classes) filled with zeros.
- Then, I used numpy log to find the logarithm of the probability of the class occurring.
- P(C=c) = no. specific class occurrences / no. all class occurrences
- Since we have two classes, spam and ham, I decided to assign the array indexes accordingly so that spam, represented by 1s, would be in index 1. Ham would be in index 0. 
	- This helped with referencing later.  
    
### Separating spam and ham messages in training data:
- I decided to separate spam and ham to make training easier and more efficient, since I could use numpy axis methods.
- To separate, I iterated through the labels array. If it was spam, I took the index and used it to move the corresponding row from the training data array into a new spam array.
- I did the same for ham messages, leaving me with two arrays: spam and ham.  

### Calculating theta:
- For this, I used a for-loop
- In the range of the number of unique words (features), it loops through each word (feature) to calculate the likelihood of it being in spam/ham.
- Once again, I used numpy log to calculate log values
- For each word, I used the previously discussed theta formula to calculate the theta value for the word's index, storing these values in a 2D numpy array.  

### Classifier function (predict):
- To implement the previously shown classification method, I used nested for-loops
- I also used a numpy array, initialised with numpy zeros of length (no. messages) to store predictions for each message
- The outer for-loop iterates through the indexes of all messages in the range (no. of messages in the data set).
- Within each iteration, the probabilities of the message being ham or spam are reset to equal the log class priors. This removes the need to add it afterwards.
- In the nested for-loop, each iteration adds the value for:
- Presence of the word (1 if present, 0 if not) multiplied by the theta value for this word given the class 
- To the probability of the message belonging to the class (ham/spam). I used separate variables for ham and spam probabilities to enable comparison later.
- Within the same nested for-loop there is a condition at the end to identify if the message is spam (this occurs when p(ham) < p(spam))
- In this case, the corresponding message index in the predictions array is set to equal 1 to indicate that the message is spam
   - In this case, the corresponding message index in the predictions array is set to equal 1 to indicate that the message is spam  
   

For the classifier, I chose to leave cases where p(ham) = p(spam) as ham as this is more realistic for real-world applications. For example, if I were to identify it as spam without it being more likely, I may accidentally hide an important email. Instead, a better approach is to leave it as ham. If the user individually identifies it as spam, then we can use that email to further train our classifier.

        
### My accuracy estimate = 0.90 (90%)

Through testing on the two notebooks (classification.ipynb and naivebayes.ipynb), my classifier method achieved accuracy of around 90% in both. This gives me confidence that my accuracy will consistently be roughly 90%. This accuracy is very good, since the large majority of messages will be correctly classified.

To increase accuracy of my classification model, simply add more data to the training set. This will make the classifier much more reliable since there is more data supporting the predictions, giving a user more confidence and peace-of-mind. Furthermore, for situations previously discussed, where p(ham)=p(spam), you could take user input as to whether these messages are spam. In doing so, more data is provided to train the model, and it will better predict messages lying in this region of uncertainty.

However, extreme cases exist which may affect accuracy. For example, if the training set consists of no spam messages, or no ham. This would hinder the ability to identify these messages, since my classifier would have no data to base the classification on.
