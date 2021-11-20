# FakeNewsDetection

Fake News Detection using Part-Of-Speech Tags (POS-Tags) sequences and Named Entity Recognition (NER). The algorithm analyzes the news articles' language by the grammatical structures, identifying potential unlawful, defamatory, threatening, false, or misleading contents. This system was submitted to [Fake News Detection in Spanish task](https://competitions.codalab.org/competitions/29545) (FakeDeS) from Iberian Languages Evaluation Forum (IberLEF).

## Datasets

 - [Fake.BR](https://github.com/roneysco/Fake.br-Corpus)
 - [FakeNewsAMT](https://web.eecs.umich.edu/~mihalcea/downloads.html)
 - [Celebrity](https://web.eecs.umich.edu/~mihalcea/downloads.html)
 - [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
 - [FakeNewsCorpusSpanish](https://github.com/jpposadas/FakeNewsCorpusSpanish)

## Requirements

 - [pandas](https://pandas.pydata.org/)
 - [spaCy](https://spacy.io)
 - [scikit-learn](https://scikit-learn.org/)
 - [WiSARD](https://github.com/IAZero/wisardpkg) @develop

## Language Model

The example from Fake.BR dataset illustrates the sentence transformation:

### News article

The dataset input contains the plain news articles.

    Maior cientista do mundo alerta: "O Planeta Terra estará inabitavel no ano de 3016". Stephen Hawking, em entrevista, alertou mais uma vez que a humanidade corre perigo. A tecnologia e os domínios da ciência são os fatores cruciais para que o planeta Terra seja extinto.

### POS-Tag

The POS sequences indicate, grammatically, the text order, coherence, and complexity.

    Maior ADJ cientista NOUN do DET mundo NOUN alerta ADJ : PUNCT " PUNCT O DET Planeta PROPN Terra PROPN estara VERB inabitável ADJ no DET ano NOUN de ADP 3016 NUM " NOUN . PUNCT SPACE Stephen PROPN Hawking PROPN , PUNCT SPACE em ADP entrevista NOUN , PUNCT SPACE alertou VERB mais ADV uma ADP vez NOUN que SCONJ a DET humanidade NOUN corre VERB perigo NOUN . PUNCT A DET tecnologia NOUN e CCONJ os DET domínios SYM da ADP ciência NOUN são VERB os DET fatores SYM cruciais ADJ para ADP que SCONJ o DET planeta NOUN Terra PROPN seja VERB extinto VERB . PUNCT

### NER

Simultaneously, applying NER, we recognize the names as possible information targets.

    Planeta Terra (LOC), Stephen Hawking (PER), Terra (LOC)


### POS-Tag+NER

Using the NE label, we replace the targets' POS Tags, increasing the semantic level. The system applies from *1 up to 7-grams* sequences to recognize patterns inside the news articles, categorized as *true* and *fake*. Distinctly from the domain-dependent techniques, this system collects the textual features without the words as references.

    ADJ NOUN DET NOUN ADJ PUNCT PUNCT DET LOC LOC VERB ADJ DET NOUN ADP NUM NOUN PUNCT SPACE PER PER PUNCT SPACE ADP NOUN PUNCT SPACE VERB ADV ADP NOUN SCONJ DET NOUN VERB NOUN PUNCT DET NOUN CCONJ DET SYM ADP NOUN VERB DET SYM ADJ ADP SCONJ DET NOUN LOC VERB VERB PUNCT

Afterward, composing highly-sparse document vectors, the language patterns are evaluated under different classifiers. The classifiers produce models that highlight the relevant features for *true* and *fake* classes, identifying the language bias. 

## References

SPALENZA, M. A.; LUSQUINO-FILHO, L. A. D.; LIMA, P. M. V.; FRANÇA, F. M. G.; OLIVEIRA E. *LCAD-UFES at FakeDeS 2021: Fake News Detection Using Named Entity Recognition and Part-of-Speech Sequences.* In: Proceedings of the Iberian Languages Evaluation Forum. Málaga, Spain: CEUR-WS, 2021. (IberLEF - SEPLN 2021, v. 37), p. 646-654. [PDF](http://ceur-ws.org/Vol-2943/fakedes_paper7.pdf)

SPALENZA, M. A.; OLIVEIRA E.; LUSQUINO-FILHO, L. A. D.; LIMA, P. M. V.; FRANÇA, F. M. G. Using NER + ML to Automatically Detect Fake News. In: Proceedings of the 20th International Conference on Intelligent Systems Design and Applications. Online Event: Springer International Publishing, 2020. (ISDA 2020, v. 20), p. 1176-1187 [Link](https://link.springer.com/chapter/10.1007/978-3-030-71187-0_109)