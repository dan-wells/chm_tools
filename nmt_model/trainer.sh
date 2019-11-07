#!/bin/bash

exp_dir=$1

echo "Starting Experiment $exp_dir"

# connlu2mt
python util.py --mode conllu2mt \
               --conllu_name $exp_dir/data.train \
               --src_name $exp_dir/train.src \
               --tgt_name $exp_dir/train.tgt \
               --pos_name $exp_dir/train.pos

python util.py --mode conllu2mt \
               --conllu_name $exp_dir/data.valid \
               --src_name $exp_dir/valid.src \
               --tgt_name $exp_dir/valid.tgt \
               --pos_name $exp_dir/valid.pos

# mt2char
python util.py --mode mt2char \
               --src_name $exp_dir/train.src \
               --tgt_name $exp_dir/train.tgt \
               --pos_name $exp_dir/train.pos \

python util.py --mode mt2char \
               --src_name $exp_dir/valid.src \
               --tgt_name $exp_dir/valid.tgt \
               --pos_name $exp_dir/valid.pos

# preprocess
onmt_preprocess -train_src $exp_dir/train.char.src  \
                -train_tgt $exp_dir/train.char.tgt  \
                --dynamic_dict \
                -save_data $exp_dir/exp

# train
onmt_train -data $exp_dir/exp \
           -save_model $exp_dir/exp_model \
           -config $exp_dir/train.yaml

# translate (evaluate)
onmt_translate -model $exp_dir/exp_model\_step_7000.pt \
               -config $exp_dir/translate.yaml \
               -src $exp_dir/valid.char.src \
               -output $exp_dir/valid.char.prd 

# mt2conllu

python util.py --mode mt2conllu \
               --src_name $exp_dir/valid.char.src \
               --tgt_name $exp_dir/valid.char.tgt \
               --prd_name $exp_dir/valid.char.prd \
               --pos_name $exp_dir/valid.char.pos \
               --conllu_name $exp_dir/valid.connlu

python eval.py --train $exp_dir/data.train \
               --test $exp_dir/data.valid \
               --pred $exp_dir/valid.connlu \
               --eval $exp_dir/eval.txt

echo "Finished Experiment $exp_dir"
