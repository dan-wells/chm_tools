#!/usr/bin/env python3

import pandas as pd

def baseline_copy(reference):
    """Evaluate lemmatization accuracy when copying surface form to lemma."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    # TODO: add index per sentence (is it useful?)
    # quoting stops special interpretation of " PUNCT
    ref = pd.read_csv(reference, sep='\t', header=None, names=ref_header, quoting=3)
    ref['form'] = ref['form'].str.lower()
    ref = ref[ref['xpostag'] != 'PUNCT']
    copy_acc = (ref['form'] == ref['lemma']).mean()
    return copy_acc

def baseline_most_frequent(train, valid):
    """Evaluate lemmatization accuracy predicting most common lemma."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    df_train = pd.read_csv(train, sep='\t', header=None, names=ref_header, quoting=3)
    df_train['form'] = df_train['form'].str.lower()
    df_train = df_train[df_train['xpostag'] != 'PUNCT']
    most_frequent_lemmas = df_train[['form', 'lemma']].groupby('form')['lemma'].apply(lambda x:x.value_counts().idxmax())

    df_valid = pd.read_csv(valid, sep='\t', header=None, names=ref_header, quoting=3)
    df_valid['form'] = df_valid['form'].str.lower()
    df_valid = df_valid[df_valid['xpostag'] != 'PUNCT']
    df_valid['plemma'] = df_valid['form'].map(most_frequent_lemmas)
    # missing vocab ends up NaN: predict with surface form
    mask = df_valid['plemma'].isnull()
    df_valid.loc[mask, 'plemma'] = df_valid.loc[mask, 'form']

    most_frequent_acc = (df_valid['lemma'] == df_valid['plemma']).mean() 
    return most_frequent_acc

def evaluate_predicted(reference, predicted):
    """Evaluate lemmatization accuracy of predictions in one CoNNL file against another."""
    ref_header = ["id", "form", "lemma", "upostag", "xpostag",
                  "feats", "head", "deprel", "deps", "misc"]
    # TODO: add index per sentence (would be useful for more certain merging)
    # quoting stops special interpretation of " PUNCT
    ref = pd.read_csv(reference, sep='\t', header=None, names=ref_header, quoting=3)
    ref = ref[ref['xpostag'] != 'PUNCT']

    # TODO: sentence indices also would help clear out PUNCT, in case there are
    # mistakes in predicted output (not sure if marmot ignores PUNCT)
    # pX = predicted X
    pred_header = ["id", "form", "lemma", "plemma",
                   "pos", "ppos", "feats", "pfeats"]
    pred = pd.read_csv(predicted, sep='\t', header=None, names=pred_header, quoting=3)
    pred = pred[pred['ppos'] != 'PUNCT'] 
    # fill in gold values
    pred[['lemma', 'pos', 'feats', 'misc']] = ref[['lemma', 'xpostag', 'feats', 'misc']]
    pred_acc = (pred['lemma'] == pred['plemma']).mean()
    return pred, pred_acc
