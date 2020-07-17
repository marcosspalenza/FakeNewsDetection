import os
import re
import unicodedata
import numpy as np
import wisardpkg as wsd
import spacy
from collections import defaultdict
from scipy.stats import rankdata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def std_text(text):
    txt = unicodedata.normalize('NFD', text)
    return u''.join(ch for ch in txt if unicodedata.category(ch) != 'Mn')

def genThermometer(db):
    therm = []
    for cid in range(np.shape(dbmatrix)[1]):
        column = db[:, cid].todense()
        ranking = rankdata(column, method="min")
        unique_values = np.unique(ranking).tolist()
        therm.append(int(np.ceil(np.log2(len(unique_values)))))
    return therm

dset1 = ["fakeNewsDatasets/fakeNewsDataset/fake/", "fakeNewsDatasets/fakeNewsDataset/legit/"]
dset2 = ["fakeNewsDatasets/celebrityDataset/fake/", "fakeNewsDatasets/celebrityDataset/legit/"]

documents = []
labels = []
doc_ent = []
dicio = defaultdict(set)
# nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")
for dset in [dset1, dset2]:
    print(dset)
    for dir_ in dset:
        for filename in os.listdir(dir_):
            if ".txt" in filename:
                with open(dir_+filename, "r", encoding="utf-8") as rdb:            
                    d = rdb.read()
                    dner = nlp(d)
                    labels.append(dir_.split("/")[-2])
                    documents.append(dner)
        print(len(labels))

    dictEN = dict()
    dictEN["PERSON"] = "PER"
    dictEN["NORP"] = "ORG"
    dictEN["FAC"] = "LOC"
    dictEN["ORG"] = "ORG"
    dictEN["GPE"] = "LOC"
    dictEN["LOC"] = "LOC"
    dictEN["PRODUCT"] = "MISC"
    dictEN["EVENT"] = "MISC"
    dictEN["WORK_OF_ART"] = "MISC"
    dictEN["LAW"] = "MISC"
    dictEN["LANGUAGE"] = "MISC"
    dictEN["DATE"] = "MISC"
    dictEN["TIME"] = "MISC"
    dictEN["PERCENT"] = "MISC"
    dictEN["MONEY"] = "MISC"
    dictEN["QUANTITY"] = "MISC"
    dictEN["ORDINAL"] = "MISC"
    dictEN["CARDINAL"] = "MISC"

    words = []
    for d in documents:
        en_doc = []
        for en in d:
            if en.ent_type_ in dictEN.keys():
                en_doc.append(dictEN[en.ent_type_])
            else:
                en_doc.append(en.pos_)
        words.append(" ".join(en_doc))
    # valuesEN = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    regex = re.compile('[^a-zA-Z]')
    cntvec = CountVectorizer(ngram_range=((1,5)))
    dbmatrix = cntvec.fit_transform(words)
    features = cntvec.get_feature_names()

    print(words[0])
    print(len(features))

    densemtx = dbmatrix.todense()
    therm = genThermometer(dbmatrix)

    mins = np.min(densemtx, axis=0).squeeze().tolist()[0]
    maxs = np.max(densemtx, axis=0).squeeze().tolist()[0]

    dtherm = wsd.DynamicThermometer(therm, mins, maxs)

    binX = [dtherm.transform(densemtx[i].tolist()[0]) for i in range(dbmatrix.shape[0])]

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    for train_index, test_index in skf.split(dbmatrix, labels):
        print("TRAIN - "+str(len(train_index)))
        print("TEST  - "+str(len(test_index)))
        ds_train = wsd.DataSet([binX[ix] for ix in train_index],[labels[ix] for ix in train_index])
        ds_test = wsd.DataSet([binX[ix] for ix in test_index], [labels[ix] for ix in test_index])
        

        print("WISARD"+str(15))
        wisard = wsd.ClusWisard(15, 0.7, 20, 20)
        wisard.train(ds_train)
        outTrain = np.array(wisard.classify(ds_train))
        outTest = np.array(wisard.classify(ds_test))
        print('Train accuracy:', accuracy_score(outTrain, [labels[ix] for ix in train_index]))
        print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))

        wisard = wsd.ClusWisard(15, 0.7, 20, 20)
        wisard.train(ds_test)
        outTrain = np.array(wisard.classify(ds_train))
        outTest = np.array(wisard.classify(ds_test))
        print('Train accuracy:', accuracy_score(outTrain, [labels[ix] for ix in train_index]))
        print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))
        
        svm = SVC()
        svm.fit(dbmatrix[train_index], [labels[ix] for ix in train_index])
        res = svm.predict(dbmatrix[test_index])
        print("SVM-TRAIN")
        print(accuracy_score(res, [labels[ix] for ix in test_index]))
        svm = SVC()
        svm.fit(dbmatrix[test_index], [labels[ix] for ix in test_index])
        res = svm.predict(dbmatrix[train_index])
        print("SVM-TEST")
        print(accuracy_score(res, [labels[ix] for ix in train_index]))

        gbc = GradientBoostingClassifier()
        gbc.fit(dbmatrix[train_index], [labels[ix] for ix in train_index])
        res = gbc.predict(dbmatrix[test_index])
        print("GBC-TRAIN")
        print(accuracy_score(res, [labels[ix] for ix in test_index]))
        gbc = GradientBoostingClassifier()
        gbc.fit(dbmatrix[test_index], [labels[ix] for ix in test_index])
        res = gbc.predict(dbmatrix[train_index])
        print("GBC-TEST")
        print(accuracy_score(res, [labels[ix] for ix in train_index]))
        
        rdf = RandomForestClassifier()
        rdf.fit(dbmatrix[train_index], [labels[ix] for ix in train_index])
        res = rdf.predict(dbmatrix[test_index])
        print("RDF-TRAIN")
        print(accuracy_score(res, [labels[ix] for ix in test_index]))

        rdf = RandomForestClassifier()
        rdf.fit(dbmatrix[test_index], [labels[ix] for ix in test_index])
        res = rdf.predict(dbmatrix[train_index])
        print("RDF-TEST")
        print(accuracy_score(res, [labels[ix] for ix in train_index]))
