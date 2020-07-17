import os
import re
import numpy as np
import pandas as pd
import unicodedata
import spacy
import wisardpkg as wsd
from scipy.stats import rankdata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

"""
PERSON:      People, including fictional.                               PER
NORP:        Nationalities or religious or political groups.            ORG
FAC:         Buildings, airports, highways, bridges, etc.               LOC
ORG:         Companies, agencies, institutions, etc.                    ORG
GPE:         Countries, cities, states.                                 LOC
LOC:         Non-GPE locations, mountain ranges, bodies of water.       LOC
PRODUCT:     Objects, vehicles, foods, etc. (Not services.)             MISC
EVENT:       Named hurricanes, battles, wars, sports events, etc.       MISC
WORK_OF_ART: Titles of books, songs, etc.                               MISC
LAW:         Named documents made into laws.                            MISC
LANGUAGE:    Any named language.                                        MISC
DATE:        Absolute or relative dates or periods.                     MISC
TIME:        Times smaller than a day.                                  MISC
PERCENT:     Percentage, including ”%“.                                 MISC
MONEY:       Monetary values, including unit.                           MISC
QUANTITY:    Measurements, as of weight or distance.                    MISC
ORDINAL:     “first”, “second”, etc.                                    MISC
CARDINAL:    Numerals that do not fall under another type.              MISC


['barely-true' 'false' 'half-true' 'mostly-true' 'pants-fire' 'true']
'true' == 'true'
'mostly-true' == 'true'
'barely-true' == 'false'
'half-true' == 'false'
'pants-fire' == 'false'
'false' == 'false'
"""
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

directory = ["train.tsv", "valid.tsv", "test.tsv"]

documents = []
labels = []
doc_ent = []

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

nlp = spacy.load("en_core_web_sm")
for dir_ in directory:
    df = pd.read_csv(dir_, header=None, delimiter="\t")
    d = df[2]
    lbl = df[1]
    print((len(d), len(lbl)))
    for ix in range(len(d)):
        dner = nlp(d[ix])
        labels.append(lbl[ix])
        documents.append(dner)
    print(len(labels))

words = []
for d in documents:
    en_doc = []
    for en in d:
        if en.ent_type_ in dictEN.keys():
            en_doc.append(dictEN[en.ent_type_])
        else:
            en_doc.append(en.pos_)
    words.append(" ".join(en_doc))


regex = re.compile('[^a-zA-Z]')
cntvec = CountVectorizer(ngram_range=((1,5)))
dbmatrix = cntvec.fit_transform(words)
features = cntvec.get_feature_names()

train_index = [ix for ix in range(0,10240)]
valid_index = [ix for ix in range(10240:11524)]
test_index = [ix for ix in range(11524:12791)]

train = dbmatrix[train_index, :]
validation = dbmatrix[valid_index, :]
test = dbmatrix[test_index, :]

print("7 - classes")

print("SVM-TRAIN")
svm = SVC()
svm.fit(train, [labels[ix] for ix in train_index])
print("SVM-VALIDATION")
res = svm.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("SVM-TEST")
res = svm.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

print("GBC-TRAIN")
gbc = GradientBoostingClassifier()
gbc.fit(train, [labels[ix] for ix in train_index])
print("GBC-VALIDATION")
res = gbc.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("GBC-TEST")
res = gbc.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

print("RDF-TRAIN")
rdf = RandomForestClassifier()
rdf.fit(train, [labels[ix] for ix in train_index])
print("RDF-VALIDATION")
res = rdf.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("RDF-TEST")
res = rdf.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

densemtx = dbmatrix.todense()
therm = genThermometer(dbmatrix)

mins = np.min(densemtx, axis=0).squeeze().tolist()[0]
maxs = np.max(densemtx, axis=0).squeeze().tolist()[0]

dtherm = wsd.DynamicThermometer(therm, mins, maxs)

binX = [dtherm.transform(densemtx[i].tolist()[0]) for i in range(dbmatrix.shape[0])]

train = binX[:10240, :]
validation = binX[10240:11524, :]
test = binX[11524:, :]

ds_train = wsd.DataSet(train, [labels[ix] for ix in train_index])
ds_valid = wsd.DataSet(validation, [labels[ix] for ix in valid_index])
ds_test = wsd.DataSet(test, [labels[ix] for ix in rtest_index])

for n in range(1, 11):
    win = n * 3
    wisard = wsd.ClusWisard(win, 0.7, 20, 10)
    wisard.train(ds_train)
    outValid = np.array(wisard.classify(ds_valid))
    outTest = np.array(wisard.classify(ds_test))
    print('Train accuracy:', accuracy_score(outValid, [labels[ix] for ix in valid_index]))
    print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))

    wisard = wsd.ClusWisard(win, 0.7, 20, 10)
    wisard.train(ds_test)
    outValid = np.array(wisard.classify(ds_valid))
    outTest = np.array(wisard.classify(ds_test))
    print('Train accuracy:', accuracy_score(outValid, [labels[ix] for ix in valid_index]))
    print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))


for li, l in enumerate(labels):
    if l == "barely-true":
        labels[li] = "false"
    elif l == "half-true":
        labels[li] = "true"
    elif l == "mostly-true":
        labels[li] = "true"
    elif l == "pants-fire":
        labels[li] = "false"
    # true == true and false == false

train = dbmatrix[train_index, :]
validation = dbmatrix[valid_index, :]
test = dbmatrix[test_index, :]

print("2 - classes")

print("SVM-TRAIN")
svm = SVC()
svm.fit(train, [labels[ix] for ix in train_index])
print("SVM-VALIDATION")
res = svm.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("SVM-TEST")
res = svm.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

print("GBC-TRAIN")
gbc = GradientBoostingClassifier()
gbc.fit(train, [labels[ix] for ix in train_index])
print("GBC-VALIDATION")
res = gbc.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("GBC-TEST")
res = gbc.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

print("RDF-TRAIN")
rdf = RandomForestClassifier()
rdf.fit(train, [labels[ix] for ix in train_index])
print("RDF-VALIDATION")
res = rdf.predict(validation)
print(accuracy_score(res, [labels[ix] for ix in valid_index]))
print("RDF-TEST")
res = rdf.predict(test)
print(accuracy_score(res, [labels[ix] for ix in test_index]))

train = binX[:10240, :]
validation = binX[10240:11524, :]
test = binX[11524:, :]

ds_train = wsd.DataSet(train, [labels[ix] for ix in train_index])
ds_valid = wsd.DataSet(validation, [labels[ix] for ix in valid_index])
ds_test = wsd.DataSet(test, [labels[ix] for ix in rtest_index])

for n in range(1, 11):
    win = n * 3
    wisard = wsd.ClusWisard(win, 0.7, 20, 10)
    wisard.train(ds_train)
    outValid = np.array(wisard.classify(ds_valid))
    outTest = np.array(wisard.classify(ds_test))
    print('Train accuracy:', accuracy_score(outValid, [labels[ix] for ix in valid_index]))
    print('Test accuracy:', accuracy_score(outTest, [labels[ix] for ix in test_index]))