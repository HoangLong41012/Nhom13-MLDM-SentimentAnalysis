import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from collections import Counter

# Đọc file
data = pd.read_csv('data\\5000-Test1.csv')
data['label'] = 0

def transform(text):
    if text == 'positive':
        return 1
    if text == 'negative':
        return 0

# Chia tập dữ liệu thành Train và Test theo tỉ lệ 8:2
data['label'] = data['sentiment'].apply(transform)
X = data["review"]
Y = data["label"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Tiền xử lý dữ liệu
stopword = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def standardize(text):
    data = []
    for word in text.split():
        if word not in stopword:
            word = lemmatizer.lemmatize(word, "n")
            word = lemmatizer.lemmatize(word, "v")
            word = lemmatizer.lemmatize(word, "a")
            data.append(word)
    return ' '.join(data)

vocabulary = []
for text in X_train:
    for word in text.split():
        vocabulary.append(word)

dictionary = Counter(vocabulary)
vocabulary = []
for item in dictionary:
    if dictionary[item] >= 6:
        vocabulary.append(item)

def filter(text):
    data = []
    for word in text.split(): 
        if word in vocabulary:
            data.append(word)  
    return ' '.join(data)

def preprocess(text):
    text = text.str.replace('http\S+', ' ')
    text = text.str.replace('\W', ' ')
    text = text.str.replace('\d', ' ')
    text = text.str.lower()
    text = text.apply(standardize)
    text = text.apply(filter)
    return text

data['review'] = preprocess(data['review'])

# Tạo feature vector
cv = CountVectorizer()
features_CV = cv.fit_transform(X_train)
features_test_cv = cv.transform(X_test)

# Tạo model SVM
model_svm = svm.SVC( C = 100, gamma = 0.0001)
model_svm.fit(features_CV, Y_train)

    # test accuracy
Y_preds_svm = model_svm.predict(features_test_cv)
print('SVM:')
print('Accuracy: ', accuracy_score(Y_test, Y_preds_svm)*100)
print('Precision: ', precision_score(Y_test, Y_preds_svm)*100)
print('Recall: ', recall_score(Y_test, Y_preds_svm)*100)
print('Confusion Matrix: ', confusion_matrix(Y_test, Y_preds_svm))
print('********************************************************')

# Tạo model NB
model_NB = MultinomialNB()
model_NB.fit(features_CV, Y_train)

    # test accuracy
Y_preds_NB = model_NB.predict(features_test_cv)
print('MultinomialNB:')
print('Accuracy: ', accuracy_score(Y_test, Y_preds_NB)*100)
print('Precision: ', precision_score(Y_test, Y_preds_NB)*100)
print('Recall: ', recall_score(Y_test, Y_preds_NB)*100)
print('Confusion Matrix: ', confusion_matrix(Y_test, Y_preds_NB))
print('********************************************************')

# Tạo model LR
model_LR = LogisticRegression(C=0.1 )
model_LR.fit(features_CV, Y_train)

    # test accuracy
Y_preds_LR = model_LR.predict(features_test_cv)
print('Logistic Regression:')
print('Accuracy: ', accuracy_score(Y_test, Y_preds_LR)*100)
print('Precision: ', precision_score(Y_test, Y_preds_LR)*100)
print('Recall: ', recall_score(Y_test, Y_preds_LR)*100)
print('Confusion Matrix: ', confusion_matrix(Y_test, Y_preds_LR))
print('********************************************************')

model_NR = MLPClassifier(hidden_layer_sizes=100, activation="logistic")
model_NR.fit(features_CV, Y_train)

Y_preds_NR = model_NR.predict(features_test_cv)
print('MLP:')
print('Accuracy: ', accuracy_score(Y_test, Y_preds_NR)*100)
print('Precision: ', precision_score(Y_test, Y_preds_NR)*100)
print('Recall: ', recall_score(Y_test, Y_preds_NR)*100)
print('Confusion Matrix: ', confusion_matrix(Y_test, Y_preds_NR))
print('********************************************************')
