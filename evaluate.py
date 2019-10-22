#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict

# TODO: is it useful to include indices per sentence anywhere?

def baseline_copy(reference):
    """Evaluate lemmatization accuracy when copying surface form to lemma."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    # quoting stops special interpretation of " PUNCT
    with open(reference) as inf:
        ref = csv.DictReader(inf, delimiter='\t', fieldnames=ref_header, quoting=csv.QUOTE_NONE)
        total = 0
        correct = 0
        for word in ref:
            if word['xpostag'] == 'PUNCT':
                continue
            word['form'] = word['form'].lower()
            if word['form'] == word['lemma']:
                correct += 1
            total += 1
        copy_acc = correct / total
    return copy_acc

def baseline_most_frequent(train, valid):
    """Evaluate lemmatization accuracy predicting most common lemma."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    # get most frequent lemmas per form from training data
    with open(train) as inf:
        ref = csv.DictReader(inf, delimiter='\t', fieldnames=ref_header, quoting=csv.QUOTE_NONE)
        lemma_counts = defaultdict(lambda : defaultdict(int))
        for word in ref:
            if word['xpostag'] == 'PUNCT':
                continue
            word['form'] = word['form'].lower()
            lemma_counts[word['form']][word['lemma']] += 1
        most_frequent_lemmas = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in lemma_counts.items()}
    # predict over validation data
    with open(valid) as inf:
        words = csv.DictReader(inf, delimiter='\t', fieldnames=ref_header, quoting=csv.QUOTE_NONE)
        total = 0
        correct = 0
        for word in words:
            if word['xpostag'] == 'PUNCT':
                continue
            word['form'] = word['form'].lower()
            try:
                pred = most_frequent_lemmas[word['form']]
            # predict surface form for unseen words
            except KeyError:
                pred = word['form']
            if pred == word['lemma']:
                correct += 1
            total += 1
    most_frequent_acc = correct / total
    return most_frequent_acc

def evaluate_predicted(reference, predicted, merged='merged.tsv'):
    """Evaluate lemmatization accuracy of predictions in one CoNNL file against another."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    # TODO: make this less specific to Lemming output format
    # pX = predicted X
    pred_header = ["id", "form", "lemma", "plemma",
                   "pos", "ppos", "feats", "pfeats"]
    merge_header = pred_header + ["misc"]
    total = 0
    correct = 0
    with open(reference) as ref, open(predicted) as pred, open(merged, 'w') as merge:
        merge_tsv = csv.DictWriter(merge, delimiter='\t', fieldnames=merge_header, quoting=csv.QUOTE_NONE)
        ref_words = csv.DictReader(ref, delimiter='\t', fieldnames=ref_header, quoting=csv.QUOTE_NONE)
        pred_words = csv.DictReader(pred, delimiter='\t', fieldnames=pred_header, quoting=csv.QUOTE_NONE)
        for rw, pw in zip(ref_words, pred_words):
            if rw['xpostag'] == 'PUNCT':
                continue
            if rw['lemma'] == pw['plemma']:
                correct += 1
            total += 1
            merge_word = pw
            merge_word.update({'lemma': rw['lemma'], 'pos': rw['xpostag'], 'feats': rw['feats'], 'misc': rw['misc']})
            merge_tsv.writerow(merge_word)
    pred_acc = correct / total
    return pred_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', help="Reference file containing "
            "gold-standard lemmata in CoNNL-U format.")
    parser.add_argument('--predicted', help="File of predicted outputs to evaluate "
            "against gold-standard lemmata from `reference` file. Should be in "
            "(abridged) CoNNL-2009 format.")
    parser.add_argument('--train', help="File in CoNNL-U format containing "
            "distinct training data to calculate most-frequent lemmata for "
            "baseline evaluation against `reference` file.")
    parser.add_argument('--merge', default="merged.tsv", help="File to merge "
            "information from `reference` and `predicted` files.")
    args = parser.parse_args()

    if args.reference is not None:
        copy_acc = baseline_copy(args.reference)
        print("Copy baseline:\t{0:.2%}".format(copy_acc))
    if args.train is not None and args.reference is not None:
        most_frequent_acc = baseline_most_frequent(args.train, args.reference)
        print("Most frequent baseline:\t{0:.2%}".format(most_frequent_acc))
    if args.reference is not None and args.predicted is not None:
        pred_acc = evaluate_predicted(args.reference, args.predicted, args.merge)
        print("Predicted accuracy:\t{0:.2%}".format(pred_acc))
