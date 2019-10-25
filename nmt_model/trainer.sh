#!/bin/bash

exp_dir=$1
mode=$2

echo "Starting Experiment $exp_dir" 

python util.py --mode $mode \
               --src_name $exp_dir/train.src \
               --tgt_name $exp_dir/train.tgt

python util.py --mode $mode \
               --src_name $exp_dir/valid.src \
               --tgt_name $exp_dir/valid.tgt

onmt_preprocess -train_src $exp_dir/train.$2.src  \
                -train_tgt $exp_dir/train.$2.tgt  \
                -save_data $exp_dir/exp

onmt_train -data $exp_dir/exp \
           -save_model $exp_dir/exp_model \
           -config train.yaml


onmt_translate -model $exp_dir/exp_model\_step_5000.pt \
               -config translate.yaml \
               -src $exp_dir/valid.$2.src \
               -output $exp_dir/valid.$2.pred 

python util.py --mode eval \
                --predict_name $exp_dir/valid.$2.pred  \
                --tgt_name $exp_dir/valid.$2.tgt \
                --pos_name $exp_dir/valid.pos >> $exp_dir/eval.txt

echo "Finished Experiment $exp_dir"