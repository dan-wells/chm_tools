# Experiments on 7-fold cross validation

mkdir fold_test

python util.py --mode fold --file_name char_level/char

echo "Starting K-fold Experiments"
for k in 0 1 2 3 4 5 6
do
    onmt_preprocess -train_src fold_test/train_fold_$k.src  \
                    -train_tgt fold_test/train_fold_$k.tgt  \
                    -save_data fold_test/fold_$k

    onmt_train -data fold_test/fold_$k  \
               -save_model fold_test/fold_$k \
               --world_size 1 \
               --gpu_ranks 0 \
               --train_steps 5000

    onmt_translate -model fold_test/fold_$k\_step_5000.pt \
                   -src fold_test/valid_fold_$k.src \
                   -output fold_test/predict_fold_$k.tgt

    python util.py --mode eval \
                   --predict_name fold_test/predict_fold_$k.tgt \
                   --tgt_name fold_test/valid_fold_$k.tgt >> fold_test/summary.txt
        
done
echo "Finished K-fold Experiments"
