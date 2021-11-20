import os
import time
import scipy as sp
import numpy as np
import pandas as pd
import en_core_web_sm # spaCy model
import xx_ent_wiki_sm # spaCy model
import pt_core_news_sm # spaCy model
import es_core_news_sm # spaCy model
# import wisardpkg as wsd
from scipy.io import mmwrite
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def getdb_PTBR(db, method="POS+NER"):
    documents = []
    labels = []
    doc_ent = []
    dicio = defaultdict(set)
    nlp = pt_core_news_sm.load()
    for dname, dtext in db:
        dner = nlp(dtext)
        labels.append(dname.split("-")[0])
        documents.append(dner)

    valuesEN = ["PER", "ORG", "LOC", "MISC"]
    words = None
    if method == "NER":
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else "Other" for en in d]) for d in documents]
    elif method == "POS":
        words = [" ".join([en.pos_ for en in d]) for d in documents]
    elif method == "POS+ENT":
        words = [" ".join(["ENT" if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    elif method == "POS+NER":
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    else: #default "POS+NER"
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    return words, documents, labels

def getdb_ESMX(dset, method="POS+NER"):
    documents = []
    labels = []
    doc_ent = []
    dicio = defaultdict(set)
    nlp = es_core_news_sm.load()
    d = dset["Text"]
    title = dset["Headline"]
    lbl = dset["Category"]
    for ix in range(len(d)):
        dner = nlp(str(title[ix])+"\n"+str(d[ix]))
        labels.append(lbl[ix])
        documents.append(dner)

    valuesEN = ["PER", "ORG", "LOC", "MISC"]
    words = []
    if method == "NER":
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else "Other" for en in d]) for d in documents]
    elif method == "POS":
        words = [" ".join([en.pos_ for en in d]) for d in documents]
    elif method == "POS+ENT":   
        words = [" ".join(["ENT" if en.ent_type_ in valuesEN or "NUMBER" in en.text else en.pos_ for en in d]) for d in documents]
    elif method == "MORPH-NUM":
        words = []
        for d in documents:
            doc = []
            for en in d:
                if str(en.morph) != "" and "Number" in str(en.morph):
                    doc.append([tk_.split("=")[1] for tk_ in str(en.morph).split("|") if tk_.split("=")[0] == "Number"][0])
                else:
                    doc.append("Neutral")
            words.append(" ".join(doc))
    elif method == "MORPH-GEN":
        words = []
        for d in documents:
            doc = []
            for en in d:
                if str(en.morph) != "" and "Gender" in str(en.morph):
                    doc.append([tk_.split("=")[1] for tk_ in str(en.morph).split("|") if tk_.split("=")[0] == "Gender"][0])
                else:
                    doc.append("Neutral")
            words.append(" ".join(doc))
    elif method == "MORPH":
        words = [" ".join(["_".join([tk.split("=")[1] if str(en.morph) != "" else "0" for tk in str(en.morph).split("|")]) for en in d])  for d in documents]
    else:
        # method == "POS+NER":
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    return words, documents, labels

def getdb_ENUS(db="AMT", method="POS+NER"):
    dset = []
    if db == "AMT":
        dset = ["fakeNewsDatasets/fakeNewsDataset/fake/", "fakeNewsDatasets/fakeNewsDataset/legit/"]
    elif db == "CEL":
        dset = ["fakeNewsDatasets/celebrityDataset/fake/", "fakeNewsDatasets/celebrityDataset/legit/"]
    else: #default
        dset = ["fakeNewsDatasets/fakeNewsDataset/fake/", "fakeNewsDatasets/fakeNewsDataset/legit/"]

    documents = []
    labels = []
    doc_ent = []
    dicio = defaultdict(set)
    nlp = en_core_web_sm.load()
    for dir_ in dset:
        for filename in os.listdir(dir_):
            if ".txt" in filename:
                with open(dir_+filename, "r", encoding="utf-8") as rdb:
                    d = rdb.read()
                    dner = nlp(d)
                    labels.append(dir_.split("/")[-2])
                    documents.append(dner)

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

    valuesEN = ["PER", "ORG", "LOC", "MISC"]
    words = None

    if method == "NER":
        words = [" ".join([en.ent_type_ if en.ent_type_ in valuesEN else "Other" for en in d]) for d in documents]
    elif method == "POS":
        words = [" ".join([en.pos_ for en in d]) for d in documents]
    elif method == "POS+ENT":
        words = [" ".join(["ENT" if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    elif method == "POS+NER":
        words = []
        for d in documents:
            wtmp = []
            for en in d:
                if en.ent_type_ in valuesEN:
                    wtmp.append(en.ent_type_)
                elif "NUMBER" in en.text:
                    wtmp.append("MISC")
                else:
                    wtmp.append(en.pos_)
            words.append(" ".join(wtmp))
    else: #default "POS+NER"
        words = [" ".join([dictEN[en.ent_type_] if en.ent_type_ in valuesEN else en.pos_ for en in d]) for d in documents]
    return words, documents, labels


DATABASE = ["ESMX"]
NGRAM_RANGE = (1,7)
OUTPUT_FILE = "output.txt"

def main():
    words = []
    documents = []
    labels = []

    if "ESMX" in DATABASE:
        input_f = "FakeNewsCorpusSpanish/train.xlsx"
        train = pd.read_excel(input_f)
        w_, d_, l_ = getdb_ESMX(train, method="POS+NER")
        words = w_
        documents = d_
        labels = l_
        input_f = "FakeNewsCorpusSpanish/development.xlsx"
        valid = pd.read_excel(input_f)
        w_, d_, l_ = getdb_ESMX(valid, method="POS+NER")
        words = words + w_
        documents = documents + d_
        labels = labels + l_

    print("ESMX")
    print(len(documents))

    cntvec = CountVectorizer(ngram_range=NGRAM_RANGE)
    dbmatrix = cntvec.fit_transform(words)
    features = cntvec.get_feature_names()

    if not os.path.isfile("dbmatrix.mtx"):
    	mmwrite("dbmatrix.mtx", dbmatrix)

    # print(np.shape(data), np.shape(dbmatrix))
    # if data != None:
    #     dbmatrix = sp.sparse.hstack((data, dbmatrix), format="csr")

    train_index = []
    valid_index = []
    test_index = []
    for id_ in range(len(documents)):
        if id_ < len(train):
            train_index.append(id_)
        else:
            valid_index.append(id_)

    print(len(train_index), len(valid_index))

    """
    # Filter features by frequencies

    freq_min = 10

    featsum = [f for f in dbmatrix.sum(axis=0).squeeze().tolist()][0]

    fidx = [fi for fi, f in sorted(enumerate(featsum), key = lambda x: x[1]) if f > freq_min]

    print(len(fidx))
    dbmatrix = dbmatrix[:, fidx]
    """

    start_time = time.time()

    svm = SVC(probability=True)
    gbc = GradientBoostingClassifier()
    rdf = DecisionTreeClassifier()

    rdf.fit(dbmatrix[train_index, :], [labels[ix] for ix in train_index])
    wl1 = rdf.predict(dbmatrix[valid_index, :])

    svm.fit(dbmatrix[train_index, :], [labels[ix] for ix in train_index])
    wl2 = svm.predict(dbmatrix[valid_index, :])

    gbc.fit(dbmatrix[train_index, :], [labels[ix] for ix in train_index])
    wl3 = gbc.predict(dbmatrix[valid_index, :])

    trtime = (time.time() - start_time) / 60

    print("RDF")
    print(accuracy_score(wl1, [labels[ix] for ix in valid_index]))
    print(f1_score(wl1, [labels[ix] for ix in valid_index], average="micro"))

    print("SVM")
    print(accuracy_score(wl2, [labels[ix] for ix in valid_index]))
    print(f1_score(wl2, [labels[ix] for ix in valid_index], average="micro"))

    print("GBC")
    print(accuracy_score(wl3, [labels[ix] for ix in valid_index]))
    print(f1_score(wl3, [labels[ix] for ix in valid_index], average="micro"))

    with open(OUTPUT_FILE, "w") as wtr:
        wtr.write("\n".join([str(lbl) for lbl in wl3]))

if __name__ == "__main__":
    main()