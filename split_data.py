import numpy as np
import random
import csv

#read in the files as textfiles:

doc_1897 = open('/afs/inf.ed.ac.uk/user/s16/s1683998/Documents/GPNLP/chm_tools/data/1897LENZ-11_CoNNL-U.tsv', 'r')
#line_1897 = doc_1897.read()
doc_1922 = open('/afs/inf.ed.ac.uk/user/s16/s1683998/Documents/GPNLP/chm_tools/data/1922AUGU_CoNNL-U.tsv', 'r')
sentences = []
words = []

for line in doc_1897:
    if line == '\n':
        sentences.append(words)
        words = []
    elif line.startswith('#'):
        continue
    else:
        words.append(line)

for line in doc_1922:
    if line == '\n':
        sentences.append(words)
        words = []
    elif line.startswith('#'):
        continue
    else:
        words.append(line)

print(sentences)
#print(len(sentences))

#split the lists and then export 10/10/80 or if I will

def extract_data(sentences, percentage, percentage2):
    random.seed(5)
    samplenumber_test = int(round(percentage*len(sentences)))
    print(samplenumber_test)
    samplenumber_val = int(round(percentage2*len(sentences)))
    print(samplenumber_val)
    random.shuffle(sentences)

    test = sentences[:samplenumber_test]

    validation = sentences[samplenumber_test:samplenumber_test+samplenumber_val]
    training = sentences[samplenumber_test+samplenumber_val:]
    return test, validation, training

test, validation, training = extract_data(sentences, 0.1, 0.1)
print(len(test))
print(len(validation))
print(len(training))
print(test)

#the next thing we want to do is to export our list back to a .tsv file for processing:
with open('data.test', 'w', newline='') as f:
    #f.writerow(['# ID', 'FORM', 'LEMMA', 'UPOSTAG', 'XPOSTAG', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'])
    f.write("# ID\tFORM\tLEMMA\tUPOSTAG\tXPOSTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n")
    for sentence in test:
        for words in sentence:
            f.write(words)
        f.write('\n')

with open('data.valid', 'w', newline='') as f:
    f.write("# ID\tFORM\tLEMMA\tUPOSTAG\tXPOSTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n")
    for sentence in validation:
        for words in sentence:
            f.write(words)
        f.write('\n')

with open('data.train', 'w', newline='') as f:
    f.write("# ID\tFORM\tLEMMA\tUPOSTAG\tXPOSTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n")
    for sentence in training:
        for words in sentence:
            f.write(words)
        f.write('\n')