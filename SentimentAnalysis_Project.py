import pandas as pd
import nltk
import math
from collections import Counter

# Đọc file
data = pd.read_csv('data\\5000-Test1.csv')
print("Number of review:", len(data))
print("Number of positive review:", len(data[data['sentiment'] == 'positive']))
print("Number of negative review:", len(data[data['sentiment'] == 'negative']))
data['label'] = 0

def transform(text):
    if text == 'positive':
        return 1
    if text == 'negative':
        return 0

data['label'] = data['sentiment'].apply(transform)

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

def preprocess(text):
    text = text.str.replace('http\S+', ' ')
    text = text.str.replace('\W', ' ')
    text = text.str.replace('\d', ' ')
    text = text.str.lower()
    text = text.apply(standardize)
    return text

# Chia tập dữ liệu thành Train và Test theo tỉ lệ 8:2
sample = data.sample(frac=1, random_state=1)
index = round(len(sample) * 0.8)
data_train = sample[:index].reset_index(drop=True)
data_test = sample[index:].reset_index(drop=True)
print("Number of data_train:", len(data_train))
print("Number of data_test:", len(data_test))

data_train['review'] = preprocess(data_train['review'])

# Tạo feature vector
vocabulary = []
for text in data_train['review']:
    for word in text.split():
        vocabulary.append(word)

print(len(vocabulary))
dictionary = Counter(vocabulary)
vocabulary = []
for item in dictionary:
    if int(dictionary[item]) >= 6:
        vocabulary.append(item)
print(len(vocabulary))

data_positive = data_train[data_train['label'] == 1]
data_negative = data_train[data_train['label'] == 0]

# Xác suất tiên nghiệm
p_positive = len(data_positive) / len(data_train)
p_negative = len(data_negative) / len(data_train)

    # Số lượng mỗi từ trong vocabulary xuất hiện trong positive/negative
n_positive_word = {word: 0 for word in vocabulary}
for text in data_positive['review']:
    for word in text.split():
        if word in vocabulary:
            n_positive_word[word] += 1

n_negative_word = {word: 0 for word in vocabulary}
for text in data_negative['review']:
    for word in text.split():
        if word in vocabulary:
            n_negative_word[word] += 1

    # Tổng số lượng từ trong positive/negative
total_positive_word = 0
total_negative_word = 0
for word in vocabulary:
    total_negative_word += n_negative_word[word]
    total_positive_word += n_positive_word[word]

    # Tính likelihood
p_word_given_positive = {word: 0 for word in vocabulary}
p_word_given_negative = {word: 0 for word in vocabulary}

n_vocabulary = len(vocabulary)
for word in vocabulary:
    p_word_given_positive[word] = (n_positive_word[word] + 1) / (total_positive_word + n_vocabulary)
    p_word_given_negative[word] = (n_negative_word[word] + 1) / (total_negative_word + n_vocabulary)


# Hàm dự đoán
def predict(text):
    p_positive_given = math.log(p_positive)
    p_negative_given = math.log(p_negative)
    for word in text.split():
        if word in vocabulary:
            p_positive_given += math.log(p_word_given_positive[word])
            p_negative_given += math.log(p_word_given_negative[word])

    if p_positive_given > p_negative_given:
        return 1
    else:
        return 0


# Dự đoán trên tập test
data_test['review'] = preprocess(data_test['review'])
data_test['predict'] = data_test['review'].apply(predict)
# Kết quả
TP = 0      # mô hình dự đoán đúng là positive
TN = 0      # mô hình dự đoán đúng là negative
FP = 0      # mô hình dự đoán sai positive thành negative
FN = 0      # mô hình dự đoán sai negative thành positive
for row in data_test.iterrows():
    row = row[1]
    if row['label'] == row['predict']:
        if row['label'] == 1:
            TP += 1
        else:
            TN += 1
    else:
        if row['label'] == 1:
            FP += 1
        else:
            FN += 1

print('Accuracy: ', (TP + TN) / (TP + TN + FP + FN) * 100)
print('Precision: ', (TP / (TP + FP)) * 100)
print('Recall: ', (TP / (TP + FN)) * 100)
print('[', TN, ',', FP, ']', '[', FN, ',', TP, ']')


