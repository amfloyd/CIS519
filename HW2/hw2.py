import sklearn
import numpy as np
import os
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt

class Classifier(object):
    def __init__(self, algorithm, x_train, y_train,n, iterations=1, averaged=False, eta=1, alpha=1.1):

        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        """

        :type alpha: object
        """
        self.G_w = None

        features = {feature for xi in x_train for feature in xi.keys()}

        if algorithm == 'Perceptron':
            # Initialize w, bias
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # Iterate over the training data n times
            for i in range(iterations):
                # Check each training example
                for i in range(n):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    # Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + yi * eta * value
                        self.w['bias'] = self.w['bias'] + yi * eta

        # initialize weights as 1 and the bias as the negative of the total number of features which is equal to the length of x_train
        elif algorithm == 'Winnow':
            val = float(len(features))
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -val
            for j in range(iterations):
                for i in range(n):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * pow(alpha, yi * value)

        elif algorithm == 'Adagrad':
            #self.w = np.zeros(len(x_train))
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            self.G = np.ones(len(x_train)+1)
            self.theta = 0.0
            self.eta = 1.5
            error = []
            v=[]
            for i in range(len(x_train)):
                for feature, value in x_train[i].items():
                    temp = y_train[i] * ((self.w[feature] * value) + self.w['bias'])
                    if temp <= 1:
                        #g_t = -y_train[i] * value
                        g_t = np.append(-y_train[i] * value, -y_train[i])
                        self.G[i] = self.G[i] + (g_t[0] ** 2)
                        #print (sum(self.G[:self.G.size-1]))
                        self.w[feature] = self.w[feature] + self.eta * y_train[i] * value / np.sqrt(self.G[i])
                        self.w['bias'] = self.w['bias'] + self.eta * y_train[i] / np.sqrt(self.G[i])

        elif algorithm == 'AvgPerceptron':
            # k=0
            # v=[]
            # c=[]
            # self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            # v.append(0)
            # c.append(0)
            # result=[]
            # for i in range(n):
            #     for feature,value in x_train[i].items():
            #         y_hat = self.predict_averaged(v,c,value)
            #         if y_hat==y_train[i]:
            #             c[k]=c[k]+1
            #         else:
            #             self.w[feature] = self.w[feature] + y_train[i] * eta * value
            #             self.w[feature]=(self.w[feature]*c[k])
            #             self.w['bias'] = self.w['bias'] + y_train[i] * eta
            #             self.w['bias'] = (self.w['bias'] * c[k])
            #             c.append(1)
            #             k=k+1
            # print (sum(c))
            # for feature in features:
            #     self.w[feature]=self.w[feature]/sum(c)
            # self.w['bias']=self.w['bias']/sum(c)
            self.w, self.w['bias'] = {feature: 0.0 for feature in features}, 0.0
            temp, temp['bias'] = {feature: 0.0 for feature in features}, 0.0
            self.w_cached, self.w_cached['bias'] = {feature: 0.0 for feature in features}, 0.0
            self.c=1.0
            for j in range(iterations):
                for i in range(n):
                    y_hat = self.predict(x_train[i])

                    if y_train[i] != y_hat:
                        for feature, value in x_train[i].items():
                            self.w[feature] = self.w[feature] + y_train[i] * eta * value
                            self.w_cached[feature] = self.w_cached[feature] + self.c * (y_train[i] * eta * value)
                        self.w['bias'] = self.w['bias'] + y_train[i] * eta
                        self.w_cached['bias'] = self.w_cached['bias'] + self.c * (y_train[i] * eta)
                    self.c = self.c + 1.0
            for feature in features:
                self.w[feature] = self.w[feature] - (self.w_cached[feature] / self.c)
            self.w['bias'] = self.w['bias'] - (self.w_cached['bias'] / self.c)
            #self.w = temp


        elif algorithm == 'AvgWinnow':
            val = float(len(features))
            self.w, self.w['bias'] = {feature: 1.0 for feature in features}, -val
            self.w_cached, self.w_cached['bias'] = {feature: 1.0 for feature in features}, -val
            self.c = 1.0
            for j in range(iterations):
                for i in range(n):
                    y_hat = self.predict(x_train[i])

                    if y_train[i] != y_hat:
                        for feature, value in x_train[i].items():
                            self.w[feature] = self.w[feature] * pow(alpha, y_train[i] * value)
                            self.w_cached[feature] = self.w_cached[feature] * (pow(alpha, y_train[i] * value))*self.c
                            #self.w['bias'] = self.w['bias'] + (y_train[i])
                            #self.w_cached['bias'] = self.w_cached['bias'] + (self.c * y_train[i])
                    self.c = self.c + 1.0
            for feature in features:
                self.w[feature] = self.w[feature] - (self.w_cached[feature] / self.c)
            self.w['bias'] = self.w['bias'] - (self.w_cached['bias'] / self.c)


        elif algorithm == 'SVM':
            self.model=svm.LinearSVC()
            self.v=DictVectorizer()
            x=self.v.fit_transform(x_train)
            self.model.fit(x,y_train)


    def predict(self, x):
        s = sum([self.w[feature] * value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1

    def predict_svm(self,x):
        #self.model=svm.LinearSVC()
        #ip = self.v.transform(x)
        y=self.model.predict(self.v.transform(x))
        return y

    def predict_svm_test(self,x):
        y = self.model.predict(self.v.fit_transform(x))
        return y

    def predict_averaged(self,v,c,x):
        s = sum(v[i]* c[i] for i in range(len(v))) * x
        return 1 if s > 0 else -1



    # Parse the real-world data to generate features,
# Returns a list of tuple lists
def parse_real_data(path):
    # List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path + filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# Returns a list of labels
def parse_synthetic_labels(path):
    # List of tuples for each sentence
    labels = []
    with open(path + 'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# Returns a list of features
def parse_synthetic_data(path):
    # List of tuples for each sentence
    data = []
    with open(path + 'x.txt') as file:
        features = []
        for line in file:
            # print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


if __name__ == '__main__':
    print('Loading data...')
    # Load data from folders.
    # Real world data - lists of tuple lists
    news_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    news_dev_data = parse_real_data('Data/Real-World/CoNLL/dev/')
    news_test_data = parse_real_data('Data/Real-World/CoNLL/test/')
    email_train_data = parse_real_data('Data/Real-World/CoNLL/train/')
    email_dev_data = parse_real_data('Data/Real-World/Enron/dev/')
    email_test_data = parse_real_data('Data/Real-World/Enron/test/')

    # #Load dense synthetic data
    syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/train/')
    syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/train/')
    syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/dev/')
    syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/dev/')

    # Load sparse synthetic data
    syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/train/')
    syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/train/')
    syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/dev/')
    syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/dev/')

    syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/test/')
    syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/test/')

    # Convert to sparse dictionary representations.
    # Examples are a list of tuples, where each tuple consists of a dictionary
    # and a lable. Each dictionary contains a list of features and their values,
    # i.e a feature is included in the dictionary only if it provides information. 

    # You can use sklearn.feature_extraction.DictVectorizer() to convert these into
    # scipy.sparse format to train SVM, or for your Perceptron implementation.
    print('Converting Synthetic data...')
    syn_dense_train = zip(*[({'x' + str(i): syn_dense_train_data[j][i]
                              for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1},
                             syn_dense_train_labels[j])
                            for j in range(len(syn_dense_train_data))])
    syn_dense_train_x, syn_dense_train_y = syn_dense_train
    syn_dense_dev = zip(*[({'x' + str(i): syn_dense_dev_data[j][i]
                            for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1},
                           syn_dense_dev_labels[j])
                          for j in range(len(syn_dense_dev_data))])
    syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

    syn_sparse_train = zip(*[({'x' + str(i): syn_sparse_train_data[j][i]
                               for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1},
                              syn_sparse_train_labels[j])
                             for j in range(len(syn_sparse_train_data))])
    syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train
    syn_sparse_dev = zip(*[({'x' + str(i): syn_sparse_dev_data[j][i]
                             for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1},
                            syn_sparse_dev_labels[j])
                           for j in range(len(syn_sparse_dev_data))])
    syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev

    syn_dense_test = zip(*[({'x' + str(i): syn_dense_test_data[j][i]
                              for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1},1)
                            for j in range(len(syn_dense_test_data))])
    syn_dense_test_x,rand = syn_dense_test

    syn_sparse_test = zip(*[({'x' + str(i): syn_sparse_test_data[j][i]
                               for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1},
                              1)
                             for j in range(len(syn_sparse_test_data))])
    syn_sparse_test_x,rand_1 = syn_sparse_test

    # Feature extraction
    print('Extracting features from real-world data...')
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        for i in range(1, len(padded) - 1):
            news_train_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            train_features.update(feats)
            feats = {feature: 1 for feature in feats}
            news_train_x.append(feats)
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        for i in range(1, len(padded) - 1):
            news_dev_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    email_test_y = []
    email_test_x = []
    for sentence in news_test_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        for i in range(1, len(padded) - 1):
            #email_test_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            feats = {feature: 1 for feature in feats if feature in train_features}
            email_test_x.append(feats)
    news_test_y = []
    news_test_x = []
    for sentence in email_test_data:
        padded = sentence[:]
        padded.insert(0, ('pad', None))
        padded.append(('pad', None))
        for i in range(1, len(padded) - 1):
            #news_test_y.append(1 if padded[i][1] == 'I' else -1)
            feat1 = 'w-1=' + str(padded[i - 1][0])
            feat2 = 'w+1=' + str(padded[i + 1][0])
            feats = [feat1, feat2]
            feats = {feature: 1 for feature in feats if feature in train_features}
            news_test_x.append(feats)
    # print('Extracting features from real-world data(Email)...')
    # email_train_y = []
    # email_train_x = []
    # train_features = set([])
    # for sentence in email_train_data:
    #     padded = sentence[:]
    #     padded.insert(0, ('pad', None))
    #     padded.append(('pad', None))
    #     for i in range(1, len(padded) - 1):
    #         news_train_y.append(1 if padded[i][1] == 'I' else -1)
    #         feat1 = 'w-1=' + str(padded[i - 1][0])
    #         feat2 = 'w+1=' + str(padded[i + 1][0])
    #         feats = [feat1, feat2]
    #         train_features.update(feats)
    #         feats = {feature: 1 for feature in feats}
    #         email_train_x.append(feats)
    # email_dev_y = []
    # email_dev_x = []
    # for sentence in news_dev_data:
    #     padded = sentence[:]
    #     padded.insert(0, ('pad', None))
    #     padded.append(('pad', None))
    #     for i in range(1, len(padded) - 1):
    #         news_dev_y.append(1 if padded[i][1] == 'I' else -1)
    #         feat1 = 'w-1=' + str(padded[i - 1][0])
    #         feat2 = 'w+1=' + str(padded[i + 1][0])
    #         feats = [feat1, feat2]
    #         feats = {feature: 1 for feature in feats if feature in train_features}
    #         email_dev_x.append(feats)

    # Print results
    # print('\nPerceptron Accuracy')
    #
    # # Test Perceptron on Dense Synthetic
    # p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y)
    # accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    # print('Syn Dense Dev Accuracy:', accuracy)
    #
    # # Test Perceptron on Sparse Synthetic
    # p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y)
    # accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    # print('Syn Sparse Dev Accuracy:', accuracy)
    #
    # # Test Perceptron on Real World Data
    # p = Classifier('Perceptron', news_train_x, news_train_y, iterations=10)
    # accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
    # print('News Dev Accuracy:', accuracy)

    sizes=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
    result=[]
    for type in ['Perceptron','Winnow','AvgPerceptron','AvgWinnow']:
    #for type in ['Perceptron','Winnow','AvgPerceptron','AvgWinnow']:
        dense_accuracy = []
        sparse_accuracy = []
        news_accuracy = []
        print(type,' Accuracy')
        #500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
        for n in sizes:
            # Test Perceptron on Dense Synthetic
            print("Sample size= ", n)

            p = Classifier(type, syn_dense_train_x, syn_dense_train_y, n)
            accuracy = sum(
                [1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]]) / len(
                syn_dense_dev_y) * 100
            print('Syn Dense Dev Accuracy:', accuracy)
            dense_accuracy.append(accuracy)

            # Test Perceptron on Sparse Synthetic
            p = Classifier(type, syn_sparse_train_x, syn_sparse_train_y, n)
            accuracy = sum(
                [1 for i in range(len(syn_sparse_dev_y)) if
                 p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]]) / len(
                syn_sparse_dev_y) * 100
            print('Syn Sparse Dev Accuracy:', accuracy)
            sparse_accuracy.append(accuracy)

            # Test Perceptron on Real World Data
            p = Classifier(type, news_train_x, news_train_y,n, iterations=10)
            #print (news_dev_x[1])
            accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict(news_dev_x[i]) == news_dev_y[i]]) / len(
                news_dev_y) * 100
            print('News Dev Accuracy:', accuracy)
            news_accuracy.append(accuracy)



        # print('Dense dev accuracy array for type ',type,'is ',dense_accuracy)
        # print('Sparse dev accuracy array for type ',type,'is ',sparse_accuracy)
        # print('News dev accuracy array for type ',type,'is ',news_accuracy)


    print('\nSVM Accuracy')

    # Test Perceptron on Dense Synthetic
    size=5000
    p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y,size)
    accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_svm(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
    print('Syn Dense Dev Accuracy:', accuracy)

    # Test Perceptron on Sparse Synthetic
    p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y,size)
    accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_svm(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
    print('Syn Sparse Dev Accuracy:', accuracy)

    # Test Perceptron on Real World Data
    p = Classifier('SVM', news_train_x, news_train_y,size, iterations=10)
    accuracy = sum([1 for i in range(len(news_dev_y)) if p.predict_svm(news_dev_x[i]) == news_dev_y[i]])/len(news_dev_y)*100
    print('News Dev Accuracy:', accuracy)