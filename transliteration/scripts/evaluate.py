#!/usr/bin/env python
import os
import re
import math
import argparse

import nltk
from lxml import etree

PARSER = etree.XMLParser(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate transliteration task')
    parser.add_argument('-s', '--src', type=argparse.FileType('r'), metavar='FILE', required=True, help='source input file')
    parser.add_argument('-o', '--out', type=argparse.FileType('r'), metavar='FILE', required=True, help='output file produced by trained model')
    parser.add_argument('-r', '--ref', type=str, required=True, help='XML file with reference transliterations')
    return parser.parse_args()

def postprocess(name):
    name = re.sub(' ', '', name.rstrip())
    name = re.sub(u'⌀', ' ', name)
    return name

def get_min_edit_distance(child_nodes, out_name):
    min_ed = math.inf
    best_ref = None
    matched = False
    for child in child_nodes:
        child = child.text
        # Count exact matches for accuracy
        if child == out_name:
            matched = True
        # Find reference transliteration with minimum edit distance to model output
        child_ed = nltk.edit_distance(out_name, child)
        if child_ed < min_ed:
            min_ed = child_ed
            best_ref = child
    return min_ed, best_ref, matched

def calculate_fs(out_name, best_ref, min_ed):
    out_len = len(out_name)
    ref_len = len(best_ref)
    lcs = (out_len + ref_len - min_ed) / 2
    recall = lcs / ref_len
    precision = lcs / out_len
    fscore = (2 * recall * precision) / (recall + precision)
    return fscore

def main():
    args = parse_args()

    ref_xml = etree.parse(args.ref, parser=PARSER)
    names = ref_xml.findall('Name')

    names_correct = 0.
    total_names = 0.
    fscore = 0.

    for source_name, out_name, ref_name in zip(args.src, args.out, names):
        # Reverse character-level format
        source_name = postprocess(source_name)
        out_name = postprocess(out_name)

        # Check source names in source and reference are identical
        children = ref_name.getchildren()
        orig_source_name = children[0].text
        assert source_name == orig_source_name

        # Find the best-matching reference transliteration
        min_ed, best_ref, matched = get_min_edit_distance(children[1:], out_name)

        # Count exact matches for accuracy
        if matched:
            names_correct += 1
        total_names += 1

        # Calculate f-score for character-level mean F-score (MFS)
        if best_ref:
            fscore += calculate_fs(out_name, best_ref, min_ed)

    # Accuracy / MFS
    print('Accuracy', names_correct / total_names)
    print('MFS', fscore / total_names)

if __name__ == '__main__':
    main()
