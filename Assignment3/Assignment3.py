# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
import math 
import random
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# %%
#importing the data from the csv files

def import_data():
    emails = []
    labels = []
    with open('emails.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) #ignoring the header
        for row in csvreader:
            emails.append(row[0])
            labels.append(row[1])
    return emails, labels

emails, labels = import_data()
labels = np.array(labels).astype(int)

#text preprocessing

def preprocess_text(emails):
    for i in range(len(emails)):
        emails[i] = emails[i].lower()
        emails[i] = emails[i].replace('subject', '')
        emails[i] = emails[i].replace('re', '')
        emails[i] = emails[i].replace('fw', '')
        emails[i] = re.sub(r'\d+', ' ', emails[i])
        emails[i] = re.sub(r'\W', ' ', emails[i])
        emails[i] = re.sub(r'\s+', ' ', emails[i])
    return emails

emails = preprocess_text(emails)

#CountVectorizer - Frequency of words
Count_Vectorizer = CountVectorizer()
X = Count_Vectorizer.fit_transform(emails)
emails_data_count = X.toarray()
features = Count_Vectorizer.get_feature_names_out()

#CountVectorizer - Binary (i.e. presence or absence of words)
NaiveBayes_Vectorizer = CountVectorizer(binary=True)
X = NaiveBayes_Vectorizer.fit_transform(emails)
emails_data_count_naivebayes = X.toarray()


#Term Frequency-Inverse Document Frequency

TFIDF_Vectorizer = TfidfVectorizer()
X = TFIDF_Vectorizer.fit_transform(emails)
emails_data_tfidf = X.toarray()


# %%
#Splitting the data into training and testing sets
#Such that the test data has 20% of the data
emails_data_count_train, emails_data_count_test, labels_count_train, labels_count_test = train_test_split(emails_data_count, labels, test_size=0.2)
emails_data_count_naivebayes_train, emails_data_count_naivebayes_test, labels_count_naivebayes_train, labels_count_naivebayes_test = train_test_split(emails_data_count_naivebayes, labels, test_size=0.2)
emails_data_tfidf_train, emails_data_tfidf_test, labels_tfidf_train, labels_tfidf_test = train_test_split(emails_data_tfidf, labels, test_size=0.2)

# %%
# Naive Bayes Classifier
def NaiveBayesClassifier(emails_data_count_naivebayes_train, labels_count_naivebayes_train, emails_data_count_naivebayes_test, labels_count_naivebayes_test):
    
    def TrainNaiveBayes(emails_data_count_naivebayes_train, labels_count_naivebayes_train):    
        num_of_features = emails_data_count_naivebayes_train.shape[1]
        probability_spam = np.sum(labels_count_naivebayes_train)/(labels_count_naivebayes_train.shape[0])
        probability_non_spam = 1 - probability_spam

        probability_words_spam = np.zeros(num_of_features)
        probability_words_non_spam = np.zeros(num_of_features)

        #smoothening that is adding 1 to the numerator and 2 to the denominator
        # 1 is added because of one email that has all words
        # 2 is added because of two emails in which one has all words whereas other has no words
        probability_words_spam = (np.sum(emails_data_count_naivebayes_train[labels_count_naivebayes_train == 1], axis=0) + 1)  / (np.sum(labels_count_naivebayes_train) + 2) 
        probability_words_non_spam = (np.sum(emails_data_count_naivebayes_train[labels_count_naivebayes_train == 0], axis=0) + 1)  / ((labels_count_naivebayes_train.shape[0] - np.sum(labels_count_naivebayes_train)) + 2)

        line_slope_w = np.zeros(num_of_features)
        line_constant_b = 0
        for i in range(num_of_features):
            line_slope_w[i] = math.log((probability_words_spam[i]*(1-probability_words_non_spam[i]) )/((1-probability_words_spam[i])*probability_words_non_spam[i]))
            line_constant_b += math.log((1-probability_words_spam[i])/(1-probability_words_non_spam[i]))
        line_constant_b += math.log(probability_spam/probability_non_spam)
        return line_slope_w, line_constant_b
    
    def TestNaiveBayes(emails_data_count_naivebayes_test, labels_count_naivebayes_test, line_slope_w, line_constant_b):
        num_test_emails , num_features = emails_data_count_naivebayes_test.shape
        labels_predicted = np.zeros(num_test_emails)
        for i in range(num_test_emails):
            sum = line_constant_b
            sum += np.sum(emails_data_count_naivebayes_test[i]*line_slope_w)
            if sum >= 0:
                labels_predicted[i] = 1
            else:
                labels_predicted[i] = 0
        accuracy = np.sum(labels_predicted == labels_count_naivebayes_test)/num_test_emails
        print("Number of wrong predictions in test data: ", num_test_emails - np.sum(labels_predicted == labels_count_naivebayes_test))
        print("Accuracy of Naive Bayes Classifier test data is: ", accuracy)
    
    def TestNaiveBayes_Train_data(emails_data_count_naivebayes_train , labels_count_naivebayes_train, line_slope_w, line_constant_b):
        num_features = emails_data_count_naivebayes_train.shape[1]
        num_train_emails = emails_data_count_naivebayes_train.shape[0]
        labels_predicted = np.zeros(num_train_emails)
        for i in range(num_train_emails):
            sum = line_constant_b
            sum += np.sum(emails_data_count_naivebayes_train[i]*line_slope_w)
            if sum >= 0:
                labels_predicted[i] = 1
            else:
                labels_predicted[i] = 0
        accuracy = np.sum(labels_predicted == labels_count_naivebayes_train)/num_train_emails
        print("Number of wrong predictions in train data: ", num_train_emails - np.sum(labels_predicted == labels_count_naivebayes_train))
        print("Accuracy of Naive Bayes Classifier train data is: ", accuracy)
    
    line_slope_w, line_constant_b = TrainNaiveBayes(emails_data_count_naivebayes_train, labels_count_naivebayes_train)
    TestNaiveBayes(emails_data_count_naivebayes_test, labels_count_naivebayes_test, line_slope_w, line_constant_b)
    TestNaiveBayes_Train_data(emails_data_count_naivebayes_train, labels_count_naivebayes_train, line_slope_w, line_constant_b)
    return line_slope_w , line_constant_b

line_slope_w_naivebayes, line_constant_b_naivebayes = NaiveBayesClassifier(emails_data_count_naivebayes_train, labels_count_naivebayes_train, emails_data_count_naivebayes_test, labels_count_naivebayes_test)
print("The whole data is train and test data for Naive Bayes Classifier")
line_slope_w_naivebayes, line_constant_b_naivebayes = NaiveBayesClassifier(emails_data_count_naivebayes, labels, emails_data_count_naivebayes, labels)

#End of Naive Bayes Classifier


# %%
# Logistic Regression Classifier

def LogisticRegressionClassifier(emails_data_count_train, labels_count_train, emails_data_count_test, labels_count_test , step_size , num_iterations):
    def TrainLogisticRegression(emails_data_count_train, labels_count_train):
        num_points ,  num_features = emails_data_count_train.shape
        def maximizing_function(w):
            result = 0
            temp = np.dot(emails_data_count_train, w)
            result = np.sum((labels_count_train - 1) * temp - np.log(1 + np.exp(-temp)))
            # for i in range(num_points):
            #     temp = np.dot(w , emails_data_count_train[i])
            #     result += (labels_count_train[i] - 1)*temp - math.log(1 + math.exp(-temp))
            #     if(temp < -100):
            #         result -= (-temp)
            #     elif(temp > 100):
            #         result = result
            #     else:
            #         result -= math.log(1 + math.exp(-temp))
            return result

        def sigmoid(w_tx):
            w_tx = np.clip(w_tx, -100, 100)
            return 1/(1 + np.exp((-1)*w_tx))

        def gradient_ascent(w):
            Error = []
            Error_on_test = []
            Error_on_train = []
            for dummy in range(num_iterations):
                gradient = np.full(num_features , 0 , dtype = 'float64')
                #for j in range(num_points):
                #    gradient += emails_data_count_train[j]*(labels_count_train[j] - sigmoid(np.dot(w , emails_data_count_train[j])))
                gradient = np.dot((labels_count_train - sigmoid(np.dot(emails_data_count_train , w ))) , emails_data_count_train)
                w += ((step_size*gradient)/( 1))
                # Error_on_test.append(TestLogisticRegression(emails_data_count_test, labels_count_test, w))
                # Error_on_train.append(TestLogisticRegression_Train_data(emails_data_count_train, labels_count_train, w))
                # Error.append(maximizing_function(w))
            return w , Error , Error_on_test , Error_on_train


        w = np.full((num_features) , 0 , dtype= 'float64') 
        w , Error_Array , Error_on_test , Error_on_train = gradient_ascent(w)
        return w , Error_Array , Error_on_test , Error_on_train
        
    
    def TestLogisticRegression(emails_data_count_test, labels_count_test, line_slope_w):
        num_test_emails = emails_data_count_test.shape[0]
        sums = np.dot(emails_data_count_test, line_slope_w)
        labels_predicted = np.where(sums >= 0, 1, 0)
        accuracy = np.sum(labels_predicted == labels_count_test) / num_test_emails
        return 1 - accuracy

    def TestLogisticRegression_Train_data(emails_data_count_train, labels_count_train, line_slope_w):
        num_train_emails = emails_data_count_train.shape[0]
        sums = np.dot(emails_data_count_train, line_slope_w)
        labels_predicted = np.where(sums >= 0, 1, 0)
        accuracy = np.sum(labels_predicted == labels_count_train) / num_train_emails
        return 1 - accuracy


    W , Error , Error_on_test , Error_on_train = TrainLogisticRegression(emails_data_count_train, labels_count_train)
    # ax = plt.axes()
    # ax.set_facecolor('lightblue')
    # ax.plot(Error_on_train, label='train')
    # ax.plot(Error_on_test, label='test')
    # ax.set_title(f'Error over train and test dataset for stepsize {step_size}')
    # ax.set_xlabel('Number of Iterations')
    # ax.set_ylabel('Error over train and test dataset')
    # ax.legend(loc=1)
    # plt.show()
    # print(Error_on_train[-1] , Error_on_test[-1])
    num_wrong_test = TestLogisticRegression(emails_data_count_test, labels_count_test, W)
    num_wrong_train = TestLogisticRegression_Train_data(emails_data_count_train, labels_count_train, W)
    print("Accuracy of Logistic Regression Classifier test data is: ", 1 - num_wrong_test)
    print("Accuracy of Logistic Regression Classifier train data is: ", 1 - num_wrong_train)
    return W 
num_iterations = 100
#fixing the number of iterations
step_size = 0.1
#Found step size by cross validation
line_slope_w_logistic_regression_tfidf = LogisticRegressionClassifier(emails_data_tfidf_train,labels_tfidf_train , emails_data_tfidf_test,labels_tfidf_test , step_size , num_iterations)
#line_slope_w_logistic_regression = LogisticRegressionClassifier(emails_data_count_train, labels_count_train, emails_data_count_test, labels_count_test)
print("The whole data is train and test data for Logistic Regression Classifier")
line_slope_w_logistic_regression_tfidf = LogisticRegressionClassifier(emails_data_tfidf, labels, emails_data_tfidf, labels, step_size, num_iterations)


#End of Logistic Regression Classifier

# %%
#SVM Classifier
Accuaracy_train = []
Accuaracy_test = []
C = 1
# Best C found through cross validation
SVClassifier = LinearSVC(dual=True, C = C , max_iter=10000)
SVClassifier.fit( emails_data_count_train, labels_count_train)
print("Accuracy of SVM Classifier train data is: ", SVClassifier.score(emails_data_count_train, labels_count_train))
print("Accuracy of SVM Classifier test data is: ", SVClassifier.score(emails_data_count_test, labels_count_test))

print("The whole data is train and test data for SVM Classifier")
SVClassifier = LinearSVC(dual=True, C = C , max_iter=10000)
SVClassifier.fit(emails_data_count, labels)
print("Accuracy of SVM Classifier whole data is: ", SVClassifier.score(emails_data_count, labels))
#End of SVM Classifier

# %%
import os
n = 1

NaiveBayes_Predictions = []
LogisticRegression_Predictions = []
SVM_Predictions = []
email_count_features = []
email_count_naivebayes_features = []
email_tfidf_features = []
while True:
    file_path = f'test/email{n}.txt'
    if not os.path.exists(file_path):
        break
    with open(file_path, 'r') as file:
        email = [file.read()]


        
        email = preprocess_text(email)
        email_count_features_temp = Count_Vectorizer.transform(email)
        email_count_features.append(email_count_features_temp.toarray()[0].tolist())

        email_count_naivebayes_features_temp = NaiveBayes_Vectorizer.transform(email)
        email_count_naivebayes_features.append( email_count_naivebayes_features_temp.toarray()[0].tolist())

        email_tfidf_features_temp = TFIDF_Vectorizer.transform(email)
        email_tfidf_features .append(email_tfidf_features_temp.toarray()[0].tolist())

        n += 1

email_count_features = np.array(email_count_features)
email_count_naivebayes_features = np.array(email_count_naivebayes_features)
email_tfidf_features = np.array(email_tfidf_features)


# Testing process for Naive Bayes classifier
def TestNaiveBayes(email_count_naivebayes_features, line_slope_w_naivebayes, line_constant_b_naivebayes):
    num_test_emails = email_count_naivebayes_features.shape[0]
    sums = line_constant_b_naivebayes + np.sum(email_count_naivebayes_features * line_slope_w_naivebayes, axis=1)
    labels_predicted = np.where(sums >= 0, 1, 0)
    return labels_predicted

# Testing process for Logistic Regression classifier
def TestLogisticRegression(email_tfidf_features, line_slope_w_logistic_regression_tfidf):
    num_test_emails = email_tfidf_features.shape[0]
    sums = np.dot(email_tfidf_features, line_slope_w_logistic_regression_tfidf)
    labels_predicted = np.where(sums >= 0, 1, 0)
    return labels_predicted

# Testing process for SVM classifier
def TestSVM(email_count_features):
    labels_predicted = SVClassifier.predict(email_count_features)
    return labels_predicted

NaiveBayes_Predictions = TestNaiveBayes(email_count_naivebayes_features, line_slope_w_naivebayes, line_constant_b_naivebayes)
LogisticRegression_Predictions = TestLogisticRegression(email_tfidf_features, line_slope_w_logistic_regression_tfidf)
SVM_Predictions = TestSVM(email_count_features)
Final_Predictions = []

# Write the testing process details to a text file
file_path = 'Test Predictions.txt'
with open(file_path, 'w') as file:
    for i in range(len(NaiveBayes_Predictions)):
        file.write(f'Email {i+1}:\n')
        file.write(f'Naive Bayes Classifier Prediction: {NaiveBayes_Predictions[i]}\n')
        file.write(f'Logistic Regression Classifier Prediction: {LogisticRegression_Predictions[i]}\n')
        file.write(f'SVM Classifier Prediction: {SVM_Predictions[i]}\n')

        Final_Predictions.append(1 if NaiveBayes_Predictions[i] + LogisticRegression_Predictions[i] + SVM_Predictions[i] >= 2 else 0)
        file.write(f'Final Prediction: {Final_Predictions[i]}\n')
        file.write('\n')    

file_path = 'Final Predictions.txt'
with open(file_path, 'w') as file:
    for i in range(len(NaiveBayes_Predictions)):
        file.write(f'Email {i+1}:  {Final_Predictions[i]}')
        file.write('\n')

# Print a confirmation message





