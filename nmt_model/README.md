# NMT-based models for lematization

## Installation

Envirenment setup is specified in ``requirements.txt`` and can be installed using ``pip``.

```bash
pip install -r requirements.txt
```

## Preprocessing

``util.py`` has various modes to perform a variety of function asociated with ``nmt_model`` preprocessing:

- conllu2mt : convert ``CONLLU`` style files to the parallel language sentence pairs of source and target.
- mt2char : tokenize the parallel language sentence predictions using characters.
- mt2conllu : convert paralle language sentence predictions to CONLLU format for evaluation.

## Training

To train an nmt_model use ``trainer.sh`` to which you need to specify the experiment directory that contains:

- ``data.train``: data used for model trainining.
- ``data.valid``: data used for model comparison when doing parameter tuning.
- ``train.yaml``: nmt_model specification. (see ``exp_template/train.yaml``).
- ``translate.yaml``: decoding specification (see ``exp_template/train.yaml``).

## Evaluation

To evaluate the model ``trainer.sh`` calls ``eval.py`` that takes releavant files and computes: accuracy, edit Distance, ROUGE for both seen and unseen data in training.
