# Utilities converting between MT and CONNLU formats

import argparse
import sentencepiece as spm

WBEGIN = "<w>"
WEND = "<\w>"
SPACE = '_'

def preprocess(args):

    if args.mode == 'conllu2mt':

        with open(args.conllu_name) as file:
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
                        src = f"{src} {tokens[1].lower().replace(' ', SPACE)}" 
                        # Lemma
                        tgt = f"{tgt} {tokens[2].lower()}"
                        pos = f"{pos} {tokens[4]}"
                        
        # Write source file
        with open(args.src_name, 'w') as file:
            for snt in srcs:
                file.write(f'{snt[1:]}\n')

        # Write target file
        with open(args.tgt_name, 'w') as file:
            for snt in tgts:
                file.write(f'{snt[1:]}\n')

        with open(args.pos_name, 'w') as file:
            for snt in poss:
                file.write(f'{snt[1:]}\n')

    elif args.mode == 'mt2char':

        dict_lang = {
            'src':args.src_name,
            'tgt':args.tgt_name,
            'pos':args.pos_name,
        }

        for lang in dict_lang.keys():
            name  = dict_lang[lang]
            sents = open(name).readlines()

            with open(name[:-4]+f".char.{lang}", "w+") as file:

                for sent in sents:
                    tokens = sent[:-1].split() # remove \n

                    for token in tokens:
                        token = f'{WBEGIN} {" ".join(list(token))} {WEND} \n'
                        file.write(token)

    elif args.mode == 'mt2spm':

        dict_lang = {
            'src':args.src_name,
            'tgt':args.tgt_name,
            'pos':args.pos_name,
        }

        for lang in dict_lang.keys():
            name  = dict_lang[lang]

            sp = spm.SentencePieceProcessor()
            sp.Load(f'{args.spm_name}.model')

            sents = open(name).readlines()

            with open(name[:-4]+f".spm.{lang}", "w+") as file:

                for sent in sents:
                    tokens = sent[:-1].split() # remove \n

                    for token in tokens:
                        pieces = sp.EncodeAsPieces(token)
                        token = f'{WBEGIN} {" ".join(pieces)} {WEND} \n'
                        file.write(token)

    elif args.mode == 'mt2conllu':

        srcs = open(args.src_name).readlines()
        tgts = open(args.tgt_name).readlines()
        prds = open(args.prd_name).readlines()
        poss = open(args.pos_name).readlines()

        with open(args.conllu_name, 'w+') as file:

            file.write("# ID\tFORM\tLEMMA\tPLEMMA\tXPOSTAG\n")
            
            for i in range(len(srcs)):

                form = "".join(srcs[i].split()[1:-1]).replace(SPACE, ' ')
                lemma = "".join(tgts[i].split()[1:-1]).replace(SPACE, ' ')
                plemma = "".join(prds[i].split()[1:-1]).replace(SPACE, ' ')
                pos = "".join(poss[i].split()[1:-1]).replace(SPACE, ' ')

                file.write(f"{i+1}\t{form}\t{lemma}\t{plemma}\t{pos}\n")

    else:

        exit('Not implemented mode')

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--conllu_name', help='CONLLU file name ')
    argparser.add_argument('--src_name', default='src', help='source file name')
    argparser.add_argument('--tgt_name', default='tgt', help='target file name')
    argparser.add_argument('--pos_name', default='pos', help='part-of-speech file name')
    argparser.add_argument('--prd_name', default='prd', help='prediction file name')
    argparser.add_argument('--spm_name', default=None, help='SentencePiece model name')
    argparser.add_argument('--mode', default='conllu2mt', help='Possible models: [conllu2mt, mt2char, mt2spm, mt2conllu]')

    args = argparser.parse_args()

    preprocess(args)


