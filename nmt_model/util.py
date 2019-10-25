# Utilities working with nmt models 

import argparse
import string

import Levenshtein
from conllu import parse

WBEGIN = "<w>"
WEND = "<\w>"
CBEGIN = "<c>"
CEND = "<\c>"
SPACE = '_'

def preprocess(args):

    if args.mode == 'conllu2mt':

        with open(args.file_name) as file:

            srcs, tgts, poss = [], [], []
            src, tgt, pos = "", "", ""

            # Remove first schema line
            for line in file.readlines()[1:]:

                if line == '\n':
                    srcs.append(src)
                    tgts.append(tgt)
                    poss.append(pos)

                    src, tgt, pos = "", "", ""

                elif line.startswith('#'):
                    continue

                else:
                    tokens = line.split('\n')[0].split('\t')

                    if tokens[3] != 'PUNCT':

                        # Surface + replace to record space
                        src = "{} {}".format(src, tokens[1].lower().replace(' ', SPACE)) 
                        # Lemma
                        tgt = "{} {}".format(tgt, tokens[2].lower())

                        pos = "{} {}".format(pos, tokens[4])
                        
        # Write source file
        with open(args.src_name, 'w') as file:
            for snt in srcs:
                file.write('{}\n'.format(snt[1:]))

        # Write target file
        with open(args.tgt_name, 'w') as file:
            for snt in tgts:
                file.write('{}\n'.format(snt[1:]))

        with open(args.pos_name, 'w') as file:
            for snt in poss:
                file.write('{}\n'.format(snt[1:]))

    elif args.mode == 'char':

        srcs = open(args.src_name).readlines()
        
        with open(args.src_name[:-4]+".char.src", "w+") as src_file:

            for src in srcs:
                src_tokens = src[:-1].split() # remove \n

                for i, src_token in enumerate(src_tokens):

                    # Make left context window
                    left = src_tokens[max(0,i-1)]
                    if left == src_token: 
                        left = ""
                    else:
                        left = "{} {} {} ".format(CBEGIN," ".join(list(left)),CEND)
                        
                    # Make right context window
                    right = src_tokens[min(len(src_tokens)-1,i+1)]
                    if right == src_token:
                        right = ""
                    else:
                        right = "{} {} {}".format(CBEGIN," ".join(list(right)),CEND)

                    # Tokens in char form
                    src_token = "{} {} {} ".format(WBEGIN," ".join(list(src_token)),WEND)

                    out = left + src_token + right

                    if out[-1] == ' ':
                        out = out[:-1]

                    src_file.write("{}\n".format(out))


        tgts = open(args.tgt_name).readlines()
        with open(args.tgt_name[:-4]+".char.tgt", "w+") as tgt_file:

            for tgt in tgts:

                tgt_tokens = tgt[:-1].split() # remove \n

                for tgt_token in tgt_tokens:
                    # Tokens in char form
                    tgt_token = "{} {} {} \n".format(WBEGIN," ".join(list(tgt_token)),WEND)

                    tgt_file.write(tgt_token)

    elif args.mode == 'bpe':

        pass

    elif args.mode == 'eval':

        with open(args.tgt_name) as tgt_file:

            with open(args.predict_name) as pred_file:

                with open(args.pos_name) as pos_file:

                    # make a POS tag stream

                    poss = []
                    for pos_lines in pos_file.readlines():
                        poss = poss + pos_lines.split()

                    total = 0
                    correct = 0
                    v_total = 0
                    v_correct = 0
                    distance = 0
                    wrong_total = 0

                    for idx, tgt_line in enumerate(tgt_file.readlines()):

                        pred_line = pred_file.readline()[:-1]
                        tgt_line  = tgt_line[:-2]
                        
                        total += 1

                        if pred_line == tgt_line:
                            correct += 1

                        if poss[idx] == 'V':
                            v_total += 1
                            if pred_line == tgt_line:
                                v_correct += 1

                        else:
                            wrong_total += 1
                            distance += Levenshtein.distance(pred_line, tgt_line)

                print('{}:: acc: {:.3f}%, v_acc: {:.3f}%, edit_avg: {:.3f}'.format(args.predict_name,
                                                                                 correct/total*100,
                                                                                 v_correct/v_total*100, 
                                                                                 distance/wrong_total))

    elif args.mode == 'fold':

        with open(args.file_name+'-train.src') as train_src:
            with open(args.file_name+'-valid.src') as valid_src:

                srcs = train_src.readlines()
                src_lines = valid_src.readlines()
                srcs = srcs + src_lines


        with open(args.file_name+'-train.tgt') as train_tgt:
            with open(args.file_name+'-valid.tgt') as valid_tgt:

                tgts = train_tgt.readlines()
                tgt_lines = valid_tgt.readlines()
                tgts = tgts + tgt_lines

        # TODO Current implementation only allow folds 
        # that equally split data
        N = len(tgts) / args.num_folds

        for i in range(args.num_folds):

            valid_src = srcs[N*i:N*(i+1)]
            valid_tgt = tgts[N*i:N*(i+1)]
            train_src = srcs[:N*i] + srcs[N*(i+1):]
            train_tgt = tgts[:N*i] + tgts[N*(i+1):]

            with open('fold_test/train_fold_{}.src'.format(i), 'w') as file:

                for elem in train_src:
                    file.write("{}".format(elem))

            with open('fold_test/train_fold_{}.tgt'.format(i), 'w') as file:

                for elem in train_tgt:
                    file.write("{}".format(elem))

            with open('fold_test/valid_fold_{}.src'.format(i), 'w') as file:

                for elem in valid_src:
                    file.write("{}".format(elem))

            with open('fold_test/valid_fold_{}.tgt'.format(i), 'w') as file:

                for elem in valid_tgt:
                    file.write("{}".format(elem))
                    
    else:

        exit('Not implemented mode')

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_name', help='CONLLU file')
    argparser.add_argument('--src_name', default='src', help='source file name')
    argparser.add_argument('--tgt_name', default='tgt', help='target file name')
    argparser.add_argument('--pos_name', default='pos', help='part-of-speech file name')
    argparser.add_argument('--predict_name', default='predict.tgt', help='prediction file name')
    argparser.add_argument('--num_folds', default=7, help='number of folds')
    argparser.add_argument('--mode', default='conllu2mt', help='Possible models: [conllu2mt, char, bpe, eval, fold]')
    argparser.add_argument('--context', default=1, help='number of context words to include')

    args = argparser.parse_args()

    preprocess(args)


