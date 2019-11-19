#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Module for parsing XML data files of the Corpus of Historical Mapudungun
project (https://benmolineaux.github.io).
"""
# TODO: wrap these things into an object defined for each file processed

import argparse
import os
import pprint
import re
import string
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from xml.etree.ElementTree import fromstring
import bs4


def parse_train_file(xml_in, connlu_out, lang="arn", textnorm=False, thraxbin=None, far=None, debug=False):
    """
    Extract tagged words under XML root element for target language.

    Args
      xmlfile: Path to XML file
      lang: ISO 639-2/3 code for target language
      debug: Boolean to print debug info

    Returns
      List of word elements extracted xmlfile for the target language.
      Each item is an xml.etree.ElementTree.Element representing a <w>
      element as defined in the Text Encoding Initiative namespace
      (https://tei-c.org/ns/1.0/).
    """
    text_id = os.path.splitext(os.path.basename(xml_in))[0]
    thrax_pat = re.compile(r'.*Output string: (.*)$')
    #soup = BeautifulSoup(open(xml_in), 'lxml')
    data = pd.read_csv(xml_in,  encoding = "utf-8") #check for separator
    # can have spa words inside arn <p> so need to decide at that level
    # (but e.g. "peso" is tagged as both arn and spa in those contexts)
    # -- assuming we should extract every word in a given line if that line
    # is tagged with the target language
    line_id = 0
    i = 1
    seen_files = defaultdict(int)
    with open(connlu_out, 'w') as outf:
        outf.write('# ID\tFORM\tLEMMA\tUPOSTAG\tXPOSTAG\tFEATS\tHEAD\tDEPREL\tDEPS\tMISC\n')
        for form, doc_name, elem in zip(data['Word'], data['Text name'], data['XML'] ):
            # TODO: try and get anchor IDs from 1922AUGU as well
            text_id = doc_name
            seen_files[text_id] +=1
            line_id = seen_files[text_id]
            i = 1
            elem = BeautifulSoup(elem, 'lxml').find('w')
            if elem.get('xml:lang') == "spa":
                continue
            train_text = elem.text

            if textnorm:
                grm = subprocess.Popen((os.path.join(thraxbin, 'thraxrewrite-tester'), '--far={}'.format(far), '--rules={}_MAP'.format(text_id[4:8])), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output = grm.communicate(elem.text.encode())
                output_str = output[0].decode().split('\n')[0]
                thrax_match = thrax_pat.match(output_str)
                if thrax_match is None:
                    print("{}_{}_{}: Rewrite failed".format(text_id, line_id, i))
                else:
                    train_text = thrax_match.group(1)
            morphemes = elem.find_all('m')
            morph_tags = list(filter(lambda x:x not in [None, 'root'], [m.get('type') for m in morphemes]))
            if len(morph_tags) > 0:
                morph_tags = '|'.join(morph_tags)
            else:
                morph_tags = '_'
            # TODO: some morphemes with no baseForm, find out why (epenthetic stuff?)
            base_forms = '|'.join(str(m.get('baseform')) for m in morphemes)
            outf.write("{}_{}_{}\t{}\t{}\t_\t{}\t{}\t_\t_\t_\torigText={},baseForms={}".format(
                text_id, line_id, i, train_text, elem.get('lemma'), elem.get('pos'), morph_tags, elem.text, base_forms))
            if elem.get('xml:lang') != lang:
                word_lang = elem.get('xml:lang')
                outf.write(',wordLang={}'.format(word_lang))
            outf.write('\n')
          
 

# <w xml:lang="arn" lemma="atrulün" pos="V" corresp="make tired"><m baseForm="atru" type="root" corresp="tired">at'ü</m><m baseForm="le" type="vb">le</m><m baseForm="(u)w" type="reflex">w</m><m baseForm="(ü)n" type="ind1s">en</m></w>


def compute_stats_from_words(words, pos=None):
    """
    Calculate quick and dirty token/type/morpheme counts from tagged text.

    Args
        words: list of xml.etree.ElementTree.Element objects representing
          words as stored in <w> elements of the Text Encoding Initiative
          namespace (https://tei-c.org/ns/1.0/)
        pos: part-of-speech tag to calculate statistics for. If None, count
          statistics over all word types in the text, even if unlabelled.
          If provided, count only words with the given tag, and ignore any
          without pos attributes.

    Returns
        Dictionary containing calculated statistics
          - tokens: total number of <w> word elements counted
          - reconstructed_word_tokens: total number of word tokens counted,
              as reconstructed by concatenating <m> morpheme elements
          - reconstructed_word_types: number of unique word types found
              by concatenating the text of <m> morpheme elements
          - morpheme_tokens: total number of <m> morpheme elements counted
          - morpheme_types: number of unique morpheme types found in text
              of <m> elements
          - morpheme_word_ratio: ratio of morpheme tokens to word tokens

    Also write input text one-word-per-line to ./text_from_words.txt, useful
    for comparison with output from sentence-level processing or to check text
    normalization.
    """
    # this maybe gives inflated type counts because hyphenated compounds tend to be split?
    # line counts are not passed from parse_train_file: that would be the number of <p> elements
    stats = {}
    stats['tokens'] = 0
    reconstructed_word_counts = defaultdict(int)
    morpheme_counts = defaultdict(int)
    lemma_counts = defaultdict(int)
    with open('text_from_words.txt', 'w') as outf:
        for l in words:
            for word in l:
                if (pos is None) or (word.get('pos') == pos):
                    reconstructed_word = ""
                    morph_tags = ""
                    morphemes = word.find_all('m')
                    for morpheme in morphemes:
                        # must be careful of deletions(?) e.g. n="12", line 135
                        # <w xml:lang="arn" lemma="mapulen" pos="V" corresp="be distant"><m baseForm="mapu" type="root" corresp="land/earth">mapu</m><m baseForm="le" type="vb">le</m><m baseForm="iy" type="ind3"></m></w>
                        if morpheme.text is not None:
                            morpheme_counts[morpheme.text.lower()] += 1
                            reconstructed_word += morpheme.text.lower()
                        if morpheme.get("type") is not None:
                            morph_tags += "{} ".format(morpheme.get("type"))
                        else:
                            morph_tags += "MISSING_TYPE ".format(morpheme.get("type"))
                    lemma_counts[word.get('lemma')] += 1
                    reconstructed_word_counts[reconstructed_word] += 1
                    outf.write("{},{},{},{}\n".format(word.text, word.get('pos'), len(morphemes), morph_tags))
                    stats['tokens'] += 1
    stats['reconstructed_word_tokens'] = sum(reconstructed_word_counts.values())
    stats['reconstructed_word_types'] = len(reconstructed_word_counts)
    stats['lemmas'] = len(lemma_counts)
    stats['morpheme_tokens'] = sum(morpheme_counts.values())
    stats['morpheme_types'] = len(morpheme_counts)
    stats['morpheme_word_ratio'] = stats['morpheme_tokens'] / stats['tokens']
    return stats


def parse_test_file(xml_in, lang="arn", debug=False):
    """
    Extract lines of raw text under XML root element for target language.

    Args
      xmlfile: Path to XML file
      lang: ISO 639-2/3 code for target language
      debug: Boolean to print debug info

    Returns
      List of strings representing raw text lines extracted from <p>
      elements (as defined in the Text Encoding Initiative namespace:
      https://tei-c.org/ns/1.0/) for the target language.
    """
    soup = BeautifulSoup(open(xml_in), 'lxml')
    lines = soup.find_all('p', {'xml:lang': lang})
    lines = [p.text for p in lines]
    # TODO: write out to CoNNL-U format with blank fields for most things, needs
    #   tokenization etc.
    # TODO: make sure stuff like `<p xml:lang="arn" n="6"><w xml:lang="spa">Señor</w>...</p>`
    #   is properly handled so that any non-Mapudungun words are annotated as such
    if debug:
        for l in lines[:5]:
            print(l)
    return lines


def compute_stats_from_lines(lines):
    """
    Calculate quick and dirty line/token/type counts from raw text.

    Args
        lines: list of xml.etree.ElementTree.Element objects representing
          single sentences as stored in <p> elements of the Text Encoding
          Initiative namespace (https://tei-c.org/ns/1.0/)

    Returns
        Dictionary containing calculated statistics
          - lines: total number of <p> paragraph elements counted
          - word_tokens: total number of word tokens across lines, as found
              by splitting on whitespace.
          - word_types: number of unique word tokens counted

    Also write input text one-word-per-line to ./text_from_lines.txt, useful
    for comparison with output from word-level processing or to check text
    normalization.
    """
    # TODO: this currently gives fewer words for 1897LENZ-11 than just counting
    # <w> elements in the tagged version: 2930 vs. 2957
    # possibly some hyphenated compounds are split into separate <w>? but most
    # seem to be treated as a single word with two morphemes
    stats = {}
    word_counts = defaultdict(int)
    with open('text_from_lines.txt', 'w') as outf:
        for line in lines:
            # TODO: better text normalization
            # ' is part of orthography, some words compounded with en-dash(/hyphen?)
            # so only deal with obvious non-word chars to start with
            punct = ".,;:()[]?¿!¡«»\""
            nopunct = line.translate(str.maketrans('', '', punct)).lower()
            words = nopunct.split(" ")
            for word in words:
                # TODO: stop these from being counted at all
                # apparently those last two dashes are different characters but don't know what
                if word not in ("", " ", "-", "–", "—"):
                    word_counts[word] += 1
                    outf.write("{}\n".format(word))
    stats['lines'] = len(lines)
    stats['word_tokens'] = sum(word_counts.values())
    stats['word_types'] = len(word_counts)
    return stats


def parse_args():
    """Parse script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("xmlfiles", nargs='+', help="XML files to parse.")
    parser.add_argument("--lang", type=str, help="ISO 639-2/3 code for "
        "language to extract from XML file. Default arn = Mapuche.",
        default="arn", choices=("arn", "spa"))
    parser.add_argument("--tagged", action='store_true', help="Use to specify "
        "that input XML file has already been morphologically tagged.")
    parser.add_argument("--pos", default=None, help="Restrict calculation of "
        "statistics to words tagged with a specific POS. If None, calculate "
        "statistics over all word types. Words with missing POS attributes are "
        "ignored when specifying this argument, but not in the general case.")
    parser.add_argument("--connlu_out", default="connlu.txt", help="Output "
        "file to write in CoNNL-U format.")
    parser.add_argument("--stats", action="store_true", help="Flag to compute "
        "statistics over input file(s).")
    parser.add_argument("--textnorm", action="store_true", help="Flag to run "
        "text normalization over input XML.")
    parser.add_argument("--thraxbin", default=None, help="Path to compiled "
        "OpenGRM Thrax utilities.")
    parser.add_argument("--far", default=None, help="Path to OpenGRM Thrax "
        "FAR compiled grammar for text normalization.")
    parser.add_argument("--debug", action='store_true', help="Print sample "
        "lines from processed files to check output.")
    return parser.parse_args()


def main():
    """Read input XML file and compute summary statistics."""
    args = parse_args()
    if args.tagged:
        # TODO: neater way of handling multiple files
        words = []
        for f in args.xmlfiles:
            parse_train_file(f, args.connlu_out, args.lang, args.textnorm, args.thraxbin, args.far, args.debug)
        if args.stats:
            stats = compute_stats_from_words(words, pos=args.pos)
    else:
        #lines = []
        for f in args.xmlfiles:
            lines = parse_test_file(f, args.lang, args.debug)
            with open('{}.txt'.format(os.path.splitext(os.path.basename(f))[0]), 'w') as outf:
                outf.writelines("{}\n".format(l) for l in lines)
        if args.stats:
            stats = compute_stats_from_lines(lines)
    if args.stats:
        print("File: {}".format(args.xmlfiles))
        print("Specific POS: {}".format(args.pos))
        pprint.pprint(stats)


if __name__ == "__main__":
    main()
