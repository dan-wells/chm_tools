"""
Evaluation and Error Analysis for file output in CONLLU format
"""

import argparse
import Levenshtein
from rouge import Rouge

SPACE = '_'

def eval(args):

    dict_conllus = {
        'train': args.train,
        'test': args.test,
        'pred': args.pred,
    }

    conllus = dict()

    for conllu in dict_conllus.keys():

        with open(dict_conllus[conllu]) as file:

            lines = file.readlines()
            schema = lines[0]
            schema = schema.split()[1:]
            lines = lines[1:]

            conllu_file = dict(zip(schema,len(schema)*[[]]))

            for line in lines:

                if line == '\n':
                    continue

                else:
                    tokens = line.split('\n')[0].split('\t')

                    if 'XPOSTAG' not in schema or tokens[schema.index('XPOSTAG')] != 'PUNCT':

                        tokens = [token.lower() for token in tokens]
                        tokens[schema.index('FORM')].replace(' ', SPACE)

                        for idx, field in enumerate(conllu_file.keys()):

                            conllu_file[field] = conllu_file[field] + [tokens[idx]]
            
        conllus[conllu] = conllu_file

    u_pos = list(set(conllus['train']['XPOSTAG'] + conllus['test']['XPOSTAG']))
    stats = open(args.eval, 'w+')

    seen_correct = dict(zip(u_pos, len(u_pos)*[0]))
    seen_totals = dict(zip(u_pos, len(u_pos)*[0]))
    seen_distance = dict(zip(u_pos, len(u_pos)*[0]))
    seen_wrongs = dict(zip(u_pos, len(u_pos)*[0]))

    unseen_correct = dict(zip(u_pos, len(u_pos)*[0]))
    unseen_totals = dict(zip(u_pos, len(u_pos)*[0]))
    unseen_distance = dict(zip(u_pos, len(u_pos)*[0]))
    unseen_wrongs = dict(zip(u_pos, len(u_pos)*[0]))

    for idx in range(len(conllus['pred']['ID'])):

        prd = conllus['pred']['PLEMMA'][idx]
        tgt = conllus['test']['LEMMA'][idx]
        pos = conllus['test']['XPOSTAG'][idx]

        if conllus['test']['FORM'][idx] in conllus['train']['FORM']:
            seen_totals[pos] += 1

            if prd == tgt:
                seen_correct[pos] += 1

            else:
                seen_wrongs[pos] += 1
                seen_distance[pos] += Levenshtein.distance(prd, tgt)
        else:
            unseen_totals[pos] += 1

            if prd == tgt:
                unseen_correct[pos] += 1

            else:
                unseen_wrongs[pos] += 1
                unseen_distance[pos] += Levenshtein.distance(prd, tgt)

    stats.write(f'XPOSTAG\tSEEN_ACCURACY\tSEEN_EDIT_DISTANCE\tUNSEEN_ACCURACY\tUNSEEN_EDIT_DISTANCE\n')

    for pos in u_pos:

        seen_acc = seen_correct[pos] / seen_totals[pos] if seen_totals[pos] else 'N/A'
        seen_edit = seen_distance[pos] / seen_wrongs[pos] if seen_wrongs[pos] else 'N/A'
        unseen_acc = unseen_correct[pos] / unseen_totals[pos] if unseen_totals[pos] else 'N/A'
        unseen_edit = unseen_distance[pos] / unseen_wrongs[pos] if unseen_wrongs[pos] else 'N/A'

        stats.write(f'{pos.upper()}\t{seen_acc}\t{seen_edit}\t{unseen_acc}\t{unseen_edit}\n')

    # Accumulative Stats 
    seen_acc  = sum(seen_correct.values())/sum(seen_totals.values()) if sum(seen_totals.values()) else 'N/A'
    seen_edit = sum(seen_distance.values())/sum(seen_wrongs.values()) if sum(seen_wrongs.values()) else 'N/A'
    unseen_acc  = sum(unseen_correct.values())/sum(unseen_totals.values()) if sum(unseen_totals.values()) else 'N/A'
    unseen_edit = sum(unseen_distance.values())/sum(unseen_wrongs.values()) if sum(unseen_wrongs.values()) else 'N/A'
    
    stats.write(f'TOTAL\t{seen_acc}\t{seen_edit}\t{unseen_acc}\t{unseen_edit}\n')
    stats.write('==============================================================\n')
    correct = sum(seen_correct.values()) + sum(unseen_correct.values())
    total = sum(seen_totals.values()) + sum(unseen_totals.values())
    distance = sum(seen_distance.values()) + sum(unseen_distance.values())
    wrong = sum(seen_wrongs.values()) + sum(unseen_wrongs.values()) 
    stats.write(f'Total Accuracy: {correct/total}\n')
    stats.write(f'Edit Distance: {distance/wrong}\n')

    # Evaluate ROUGE
    rouge = Rouge()
    ref = [" ".join(elem) for elem in conllus['test']['LEMMA']]
    hyp = [" ".join(elem) for elem in conllus['pred']['PLEMMA']]
    scores = rouge.get_scores(hyp,ref,avg=True) 
    stats.write('ROUGE-1: {rouge-1}\n'.format(**scores))
    stats.write('ROUGE-2: {rouge-2}\n'.format(**scores))
    stats.write('ROUGE-L: {rouge-l}\n'.format(**scores))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.conllu', help='Training data CONLLU file')
    parser.add_argument('--test', default='test.connlu', help='Testing (Evaluation) data CONLLU file')
    parser.add_argument('--pred', default='pred.conllu', help='Prediction CONLLU file ')
    parser.add_argument('--eval', default='eval.txt', help=' Evaluation Data')

    args = parser.parse_args()

    eval(args)