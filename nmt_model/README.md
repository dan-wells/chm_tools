# NMT model for lematization

## Installation

to install used packages use ``requirements.txt`` file.

```bash
pip install -r requirements.txt
```

## Data Preparation

Use the ``util.py`` with its various modes to perform a variety of function asociated with ``nmt_model``:

- ``conllu2mt``: converts ``conllu`` files to format suitable for MT system of parallel source (src) and target (tgt) sentences.
- ``chart``: converts parallel src and tgt sentences to src and tgt for char-level lematization based on [lematus](https://www.aclweb.org/anthology/N18-1126/)
- ``bpe``: converstion of src and tgt sentences to src and tgt bpe-level lemematization based on [lematus](https://www.aclweb.org/anthology/N18-1126/) (``TODO``)
- ``eval``: evaluation nmt_model prediction and gold standard. Currently recording 0/1 accuracy and wrong lematization edit-distance between gold standard.
- ``fold``: make a dataset for fold testing given original train and valid data. Stored in ``fold_test`` folder.

## Experiments

### Fold_test

Minimialistic 7-fold cross validation for the given dataset.
