from Tools.scripts.ndiff import fopen
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
import pickle
import sys

"""
fname = "data.txt"
f = open(fname, "r")
data = []
for line in f:
    row = line.split()
    data.append(row)
data = np.array(data)
#print(data)
#print(data[:,-1])

#data = pd.DataFrame(data, columns = [i for i in range(data.shape[1])])
data = pd.DataFrame(data)

"""


#rawdata
#fname = "train_new.txt"


def featureExtrationTraining(f):
    """
    Extracts features for train data
    :param f: data file
    :return: feature exracted data
    """
    data = []
    for line in f:
        #row = []
        div = line.split("|")
        lang = div[0]
        rowtoappend = featureExtraction(div[1])
        rowtoappend.append(lang)
        data.append(rowtoappend)

    data = pd.DataFrame(data)
    return data

def featureExtrationTest(f):
    """
    Extracts features for test data
    :param f: file
    :return: features
    """
    data = []
    for line in f:
        rowtoappend = featureExtraction(line)
        data.append(rowtoappend)
    data = pd.DataFrame(data)
    return data

def featureExtraction(div):
    """
    Extract features of each line
    :param div: a sentence
    :return: row of features
    """
    row = []
    countOfkj = 0
    numberofwords = 0
    totallength = 0
    numberofdoubleletters = 0
    istrue1 = False
    istrue2 = False

    for i in div.split():
        # print(i)
        numberofwords = numberofwords + 1
        prev = None
        for j in i:
            # print(j)
            totallength = totallength + 1

            if j == 'k' or j == 'j':
                countOfkj = countOfkj + 1

            if prev is not None:
                if j == prev:
                    numberofdoubleletters = numberofdoubleletters + 1
            prev = j

        ilow = i.lower()
        if not istrue1:
            if ilow == 'she' or ilow == 'the' or ilow == 'a' or ilow == 'an' or ilow == 'and' or ilow == 'you' or ilow == 'but':
                # print(i)
                # row.append('Yes')
                istrue1 = True

        if not istrue2:
            if ilow == 'het' or ilow == 'de' or ilow == 'een' or ilow == 'en' or ilow == 'de' or ilow == 'eng' or ilow == 'maar':
                # row.append('Yes')
                istrue2 = True

    if istrue1:
        row.append('Yes')
    else:
        row.append('No')

    if istrue2:
        row.append('Yes')
    else:
        row.append('No')

    if totallength / numberofwords > 5:
        row.append('Yes')
    else:
        row.append('No')

    if countOfkj > 3:
        row.append('Yes')
    else:
        row.append('No')

    if numberofdoubleletters > 2:
        row.append('Yes')
    else:
        row.append('No')

    if 'q' in div[1].split():
        row.append('Yes')
    else:
        row.append('No')

    return row


maxdepth = 10

#print(data)
#target = data.shape[1]-1

# def entropy2(targetcol):
#     #print(targetcol)
#     element, count = np.unique(str(targetcol).split(), return_counts=True)
#     #print(element)
#     #print(count)
#     #entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(element))])
#     entro = 0
#     for i in range(len(element)):
#         entro = entro + (-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))
#
#     #print(entro)
#     return entro

def entropy(output_col):
    """
    Calculates the entropy
    :param output_col: the column of which entropy to be found
    :return: entropy
    """
    # print(output_col)
    items, Count = np.unique(str(output_col).split(),return_counts=True)

    entro = 0

    for i in range(len(items)):

        entro = entro + (-Count[i]/np.sum(Count))*np.log2(Count[i]/np.sum(Count))

    return entro





def InfoGain(data, splita, target):
    """
    Calculates informaltion gain
    :param data: dataset
    :param splita: attribute at which the splitting is taking place
    :param target: the target attribute
    :return: information gain
    """
    #print(data)
    #print(data[target])
    totale = entropy(data[target])
    value, count = np.unique(data[splita], return_counts=True)
    WeightedE = 0
    for i in range(len(value)):
       WeightedE = WeightedE + (count[i] / np.sum(count)) * entropy(data.where(data[splita] == value[i])[target])

    ig = totale - WeightedE
    #print(ig)
    return ig





def Dtree(data, originaldata, features, targetfeature, depth = 0, parent =None):
    """
    Finds the decision tree
    :param data: training data
    :param originaldata: parent data before splitting
    :param features:features
    :param targetfeature:target feature
    :param depth:depth of the tree till now
    :param parent:parent attribute
    :return:decision tree
    """

    if len(np.unique(data[targetfeature])) <= 1:
        return np.unique(data[targetfeature])[0]

    elif len(features) == 0:
        return parent

    elif len(data) == 0:
        return np.unique(originaldata[targetfeature])[np.argmax(np.unique(originaldata[targetfeature], return_counts=True)[1])]

    elif depth >= maxdepth:
        return parent

    else:
        parent = np.unique(data[targetfeature])[np.argmax(np.unique(data[targetfeature], return_counts=True)[1])]
        #print(targetfeature)
        items = [InfoGain(data, feature, targetfeature) for feature in features]  # Return the information gain values for the features in the dataset
        #print(targetfeature)
        #print(items)

        best = features[np.argmax(items)]
        tree = {best: {}}

        #print(tree)
        features = [i for i in features if i != best]
        for value in np.unique(data[best]):
            value = value
            subtree = Dtree(data.where(data[best] == value).dropna(), data, features, targetfeature, depth+1, parent)
            tree[best][value] = subtree

        #print(type(tree))
        return (tree)

#print(data.shape[1] - 1)


def predict(eachtest, tree, default = 1):
    """
    Predicts given the model
    :param eachtest: testcase
    :param tree: model
    :param default: is 1
    :return: prediction en or nl
    """
    for key in list(eachtest.keys()):
        #print(type(tree))
        if key in list(tree.keys()):
            prediction = tree[key][eachtest[key]]
            if isinstance(prediction, dict):
                return predict(eachtest, prediction)
            else:
                return prediction


class DecisionStump():
    """
    Finds each decision stump
    """

    def __init__(self):
        """
        Initialise decision stump
        """
        self.polar = 1
        self.fidx = None
        self.thresh = None
        self.alpha = None

    def predict(self, X):
        """
        :param X: data to predict
        :return: prediction
        """
        n_samples = X.shape[0]
        X_column = X.iloc[:, self.fidx]
        predictions = np.ones(n_samples, dtype=int)
        if self.polar == 1:
            predictions[X_column < self.thresh] = -1
        else:
            predictions[X_column > self.thresh] = 1

        return predictions


class ada():

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        """
        Finds the decision stumps and the weights
        :param X: data
        :param y: target attribute
        :return: None
        """
        example, nf = X.shape
        w = np.full(example, (1 / example))

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            minim = float('inf')
            for fi in range(nf):
                X_column = X.iloc[:,fi]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(example, dtype=int)
                    predictions[X_column < threshold] = -1
                    incorrect = w[y != predictions]
                    error = sum(incorrect)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < minim:
                        clf.polar = p
                        clf.thresh = threshold
                        clf.fidx = fi
                        minim = error

            e = 1e-10

            clf.alpha = 0.5 * np.log((1.0 - minim + e) / (minim + e))

            predictions = clf.predict(X)
            z1 = np.array(np.dot(-clf.alpha, y, predictions), dtype=np.float32)
            #print(z1)
            w *= np.exp(z1)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        """
        Predicts on X
        :param X: data to predict
        :return: predictions
        """
        result = [clf.alpha * clf.predict(X) for clf in self.clfs]
        yd = np.sum(result, axis=0, dtype=int)
        # print(yd)
        yd = np.sign(yd)
        #print(yd)
        return yd


if len(sys.argv) == 5:
    #train
    #do = sys.argv[0]
    fname = sys.argv[2]
    hout = sys.argv[3]
    classifier = sys.argv[4]


    f = open(fname, "r", encoding='utf-8')

    data = featureExtrationTraining(f)

    if classifier == 'dt':
        # testing_data = data.iloc[:10].reset_index(drop=True)
        # training_data = data.iloc[10:].reset_index(drop=True)

        # training_data = data.sample(frac=0.8, random_state=200)  # random state is a seed value
        # testing_data = data.drop(training_data.index)

        # print(testing_data)

        # training_data, testing_data = train_test_split(data, train_size=0.7, random_state=42)

        tree = Dtree(data, data, [i for i in range(data.shape[1] - 1)],
                     data.shape[1] - 1)

        #test(testing_data, tree)

        #pprint(tree)

        with open('testtree.pickle', 'wb') as opf:
            pickle.dump(tree, opf)

    else:
        X = data.iloc[:, 0:6]
        y = data.iloc[:, 6]

        for i in range(len(y)):
            if y[i] == 'en':
                y[i] = 1
            else:
                y[i] = -1

        #print(X)
        #print(y)

        y[y == 0] = -1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

        # Adaboost classification with 5 weak classifiers
        clf = ada(n_clf=5)
        clf.fit(X_train, y_train)
        #print(type(clf))
        #y_pred = clf.predict(X_test)
        #print(y_pred)
        with open('testtree.pickle', 'wb') as opf:
            pickle.dump(clf, opf)


else:
    #test
    do = sys.argv[1]
    tree = sys.argv[2]
    testdata = sys.argv[3]
    with open('testtree.pickle', 'rb') as opf:
        tree1 = pickle.load(opf)
    ft = open(testdata, "r", encoding='utf-8')
    tdata = featureExtrationTest(ft)
    #print(tdata)
    #pprint(tree1)
    #print(type(tree1))
    if isinstance(tree1, dict):
        queries = tdata.to_dict(orient="records")
        predicted = pd.DataFrame(columns=["predicted"])
        for i in range(len(tdata)):
            predicted.loc[i, "predicted"] = predict(queries[i], tree1, 1)
        print(predicted)

    else:
        predicts = tree1.predict(tdata)
        predictionslist = []
        for i in range(len(predicts)):
            if predicts[i] == 1:
                predictionslist.append('en')
            else:
                predictionslist.append('nl')

        predictionslist = pd.DataFrame(predictionslist)
        print(predictionslist)
    #print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == tdata[tdata.shape[1] - 1]) / len(tdata)) * 100, '%')
