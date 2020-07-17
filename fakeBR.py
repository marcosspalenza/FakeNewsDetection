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

directory = ["Fake.br-Corpus/full_texts/fake/", "Fake.br-Corpus/full_texts/true/"]

documents = []
labels = []
doc_ent = []
dicio = defaultdict(set)
nlp = spacy.load("pt_core_news_sm")
for dir_ in directory:
    for filename in os.listdir(dir_):
        if ".txt" in filename:
            with open(dir_+filename, "r", encoding="utf-8") as rdb:            
                d = rdb.read()
                dner = nlp(d)
                #for ent in dner.ents:
                #    dicio[ent.label_].add(ent.text)
                #doc_ent.append(" ".join([ent.text for ent in dner.ents]))
                labels.append(dir_.split("/")[-2])
                documents.append(dner)

# words = [" ".join([en.ent_type_ if en.ent_type_ in ["PER", "ORG", "LOC", "MISC"] else "Other" for en in d]) for d in documents]
words = [" ".join([en.ent_type_ if en.ent_type_ in ["PER", "ORG", "LOC", "MISC"] else en.pos_ for en in d]) for d in documents]


regex = re.compile('[^a-zA-Z]')
cntvec = CountVectorizer(ngram_range=((1,5)))
dbmatrix = cntvec.fit_transform(words)
features = cntvec.get_feature_names()

print(words[0])
print(len(features))

skf = StratifiedKFold(n_splits=10, shuffle=True)

densemtx = dbmatrix.todense()
therm = genThermometer(dbmatrix)

mins = np.min(densemtx, axis=0).squeeze().tolist()[0]
maxs = np.max(densemtx, axis=0).squeeze().tolist()[0]

dtherm = wsd.DynamicThermometer(therm, mins, maxs)

binX = [dtherm.transform(densemtx[i].tolist()[0]) for i in range(dbmatrix.shape[0])]
for train_index, test_index in skf.split(dbmatrix, labels):
    for n in range(1, 11):
        win = n * 3
        print("TRAIN - "+str(len(train_index)))
        print("TEST  - "+str(len(test_index)))
        ds_train = wsd.DataSet([binX[ix] for ix in train_index],[labels[ix] for ix in train_index])
        ds_test = wsd.DataSet([binX[ix] for ix in test_index], [labels[ix] for ix in test_index])

        wisard = wsd.ClusWisard(win, 0.7, 20, 10)
        wisard.train(ds_train)
        outTrain = np.array(wisard.classify(ds_train))
        outTest = np.array(wisard.classify(ds_test))
        print('Train accuracy:', accuracy_score(outTrain, [labels[ix] for ix in train_index]))
        print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))

        wisard = wsd.ClusWisard(win, 0.7, 20, 10)
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
