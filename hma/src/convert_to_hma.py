# *_* coding: utf-8 *_*

"""
Module for converting nicely tagged data files of the Corpus of Historical Mapudungun
project (https://benmolineaux.github.io).
Step after the data obtained from https://github.com/dan-wells/chm_tools/tree/master/data
"""

import argparse
import pandas as pd
import math

def parse_args():
    """Parse script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tsvfiles", nargs='+', help="tsv files to convert")
   
    return parser.parse_args()

def convert_and_store(infile, use_pos=True, use_feats=False, use_morphene=False):
    """
    Read the tsv file and store it into the hma code format. 
    Args
        filename: string, file path which needs to be converted 
        use_pos: boolean, if True, adds the pos tag of surface to the file. False, excludes it
        use_feats: boolean *figuring out*
        use_morphene: boolean *figuring out*
    Returns
        None 
    TODO: using morphene boundaries as features, FEATS as features
    """
    print('Reading ' + infile)
    data = pd.read_csv(infile, sep='\t')

    outfile_lem = open(infile +'_lem.txt','w+')   
    outfile_seg = open(infile +'_seg.txt','w+')
    
    count = 0
    for form, lemma, pos, feats, morph in zip (data['FORM'], data['LEMMA'], data['XPOSTAG'], data['FEATS'], data['MISC']):
        #exclude punctuations and empty lines
        count = count + 1
        if pos == 'PUNCT' or pos =='_':
            continue
        #exclude empty lines like sentence starters
        try:
            if len(form) < 1:
                continue
        except:
            if math.isnan(form) or math.isnan(pos):
                continue 
        
        if len(morph) == 0:
            morph = form
        else:
            morph = morph.split('baseForms=')[1]
            if len(morph) == 0:
                morph = form
        s_seg = form.lower() + '\t' 
        s_lem = morph.lower() + '\t'
        if len(pos) > 0:
            s_seg = s_seg + 'pos=' + pos + '\t'
            s_lem = s_lem + 'pos=' + pos + '\t'
        
        s_seg = s_seg + morph.lower() + '\n'
        s_lem = s_lem + lemma.lower() + '\n'
        outfile_seg.write(s_seg)
        outfile_lem.write(s_lem)

  

    outfile_lem.close()
    outfile_seg.close()
    print('Writing Complete')    

def main():
    """Read input tsv file and convert into format expected by the code"""
    args = parse_args()
    for filename in args.tsvfiles:
        convert_and_store(filename)


if __name__ == "__main__":
    main()