from sklearn import tree
from sklearn import linear_model
from subprocess import check_call
import graphviz
import pydot
import pydotplus
import numpy as np
import string
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
#X = [[0, 0], [1, 1]]
#y = [0, 1]
# clf = linear_model.SGDClassifier()
# clf=tree.DecisionTreeClassifier()
# clf = clf.fit(X,y)
# dot_data=tree.export_graphviz(clf,out_file=None)
# check_call(['dot','-Tpng','tree.dot','-o','treeOut.png'])
# graph=pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('tree.png')
# print(graph)
#clf.predict([[2.,2.]])
import pandas as pd
data=pd.read_csv("badges.modified.data.train", delimiter=' ',names=['Label','First Name','Last Name'])
labels=data['Label']
firstName=data['First Name']
lastName=data['Last Name']
#name=data['First Name','Last Name']
#X=data.values[:,1:2]
#Y=data.values[:,0]
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)
#clf_entropy = DecisionTreeClassifier(max_depth=3)
#X=np.zeros(260)
#X1=np.zeros(26)
#Y=np.zeros((len(firstName),1))
#feature 1 where len of first name is less than 5 then it's -(0) otherwise +(1)
lenW=5
def features(data,X,Y):

    labels = data['Label']
    firstName = data['First Name']
    lastName = data['Last Name']
    Y.extend(labels)
    #print (Y)
    for i, l in zip(firstName, lastName):
        k = 0
        Xtemp = []
        if (len(i) > 5):
            for j in i:
                if (k < lenW):
                    ind = string.ascii_lowercase.index(j)
                    k = k + 1
                    X1 = [0] * 26
                    X1[ind] = 1
                    Xtemp.extend(X1)
        else:
            for j in i:
                if (k < lenW):
                    ind = string.ascii_lowercase.index(j)
                    k = k + 1
                    X1 = [0] * 26
                    # X1[ind] = 1
                    Xtemp.extend(X1)
            Xtemp.extend([0]*(lenW-k)*26)
        k = 0
        if (len(j) > 5):
            for j in l:
                if (k < lenW):
                    ind = string.ascii_lowercase.index(j)
                    k = k + 1
                    X1 = [0] * 26
                    X1[ind] = 1
                    Xtemp.extend(X1)
        else:
            for j in l:
                if (k < lenW):
                    ind = string.ascii_lowercase.index(j)
                    k = k + 1
                    X1 = [0] * 26
                    # X1[ind] = 1
                    Xtemp.extend(X1)
            Xtemp.extend([0] * (lenW - k) * 26)
        #print(len(Xtemp))
        X.append(Xtemp)
    return X,Y


data1=pd.read_csv("badges.modified.data.fold1", delimiter=' ',names=['Label','First Name','Last Name'])
data2=pd.read_csv("badges.modified.data.fold2", delimiter=' ',names=['Label','First Name','Last Name'])
data3=pd.read_csv("badges.modified.data.fold3", delimiter=' ',names=['Label','First Name','Last Name'])
data4=pd.read_csv("badges.modified.data.fold4", delimiter=' ',names=['Label','First Name','Last Name'])
data5=pd.read_csv("badges.modified.data.fold5", delimiter=' ',names=['Label','First Name','Last Name'])

X1=[]
Y1=[]
X1,Y1=features(data1,X1,Y1)

X2=[]
Y2=[]
X2,Y2=features(data2,X2,Y2)

X3=[]
Y3=[]
X3,Y3=features(data3,X3,Y3)

X4=[]
Y4=[]
X4,Y4=features(data4,X4,Y4)

X5=[]
Y5=[]
X5,Y5=features(data5,X5,Y5)

#data.train. This part is for calculating tra
# X6=[]
# Y6=[]
# X6,Y6=features(data,X6,Y6)
# X_train_1=np.array(X6)
# Y_test_1=np.array(Y6)
#
# for i in range(len(Y_test_1)):
#     if (Y_test_1[i] == '+'):
#         Y_test_1[i] = 1
#         # print ("Hello")
#     elif (Y_test_1[i] == '-'):
#         Y_test_1[i] = 0
# Y_test_1 = Y_test_1.astype(np.int)
#SGD
data_cross=[(X1,Y1),(X2,Y2),(X3,Y3),(X4,Y4),(X5,Y5)]
for i in range(len(data_cross)):
    X_test,Y_test=data_cross[i]
    X_train=[]
    Y_train=[]
    for j in range(len(data_cross)):
        if(j!=i):
            # print(j)
            # print(len(data_cross[j][0]))
            X_train.extend(data_cross[j][0])
            Y_train.extend(data_cross[j][1])
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    # print(X_train.shape)
    # print(Y_train.shape)
    #model=SGDClassifier(loss="log",penalty='l2',alpha=0.001,learning_rate="optimal",eta0=0.73)
    model = tree.DecisionTreeClassifier(max_depth=8)
    X_rand=[]
    Y_rand=[]
    for r in range(100):
        X_temp1=np.random.choice(range(len(X_train)),int(len(X_train/2)))
        X_rand.extend(X_train[X_temp1])
        Y_rand.extend(Y_train[X_temp1])
        model.fit(X_rand,Y_rand)
    model_pred=model.predict(X_test)
    #this is for the training data
   # model_pred_a=model.predict(X_train_1)
    model_pred = np.array(model_pred)
    for i in range(len(Y_test)):
        if (Y_test[i]== '+') :
            Y_test[i]=1
            #print ("Hello")
        elif (Y_test[i]=='-'):
            Y_test[i]=0
    #this part is for tra
    # for i in range(len(model_pred_a)):
    #     if (model_pred_a[i]== '+') :
    #         model_pred_a[i]=1
    #         #print ("Hello")
    #     elif (model_pred_a[i]=='-'):
    #         model_pred_a[i]=0

    for i in range(len(model_pred)):
        if (model_pred[i] == '+'):
            model_pred[i] = 1
                # print ("Hello")
        elif (model_pred[i] == '-'):
            model_pred[i] = 0
    model_pred=model_pred.astype(np.int)
    #model_pred_a=model_pred_a.astype(np.int)
    #print(model_pred_a)
    Y_test = Y_test.astype(np.int)
    #model_pred = np.array(model_pred)
    accuracy=np.equal(model_pred,Y_test).sum()/len(Y_test)
    #this part is for tra
    #accuracy_a=np.equal(model_pred_a,Y_test_1).sum()/len(Y_test_1)
    print(accuracy)


part2 = tree.export_graphviz(model, out_file='2e.dot')
    # print (accuracy)








