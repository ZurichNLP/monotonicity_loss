import sys
import re
import os
import argparse
from typing import List
from dataclasses import dataclass
import numpy as np

## code from https://github.com/shijie-wu/neural-transducer

def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)

    return int(table[len(str2)][len(str1)])

@dataclass
class Eval:
    desc: str
    long_desc: str
    res: float

class Evaluator(object):
    def __init__(self):
        pass

    def evaluate_all(self, data_iter, nb_data, model, decode_fn) -> List[Eval]:
        raise NotImplementedError

class BasicEvaluator(Evaluator):
    '''docstring for BasicEvaluator'''
    def evaluate(self, predict, ground_truth):
        '''
        evaluate single instance
        '''
        correct = 1
        if len(predict) == len(ground_truth):
            for elem1, elem2 in zip(predict, ground_truth):
                if elem1 != elem2:
                    correct = 0
                    break
        else:
            correct = 0
        dist = edit_distance(predict, ground_truth)
        return correct, dist


class G2PEvaluator(BasicEvaluator):
    def evaluate(self, predict, ground_truth):
        correct, dist = super().evaluate(predict, ground_truth)
        return correct, dist / len(ground_truth)


def main(prediction, reference):
    per_evaluator = G2PEvaluator()
    basic_evaluator = BasicEvaluator()
    total = 0
    total_corr_per = 0
    total_per = 0
    total_corr_ed = 0
    total_ed = 0
    ref_words = dict()
    p_len = 0
    r_len = 0
    for pred, ref in zip(prediction, reference):
        total +=1
        ref_words[ref] =1
        pred = pred.rstrip().split(' ')
        ref = ref.rstrip().split(' ')
        p_len += len(pred)
        r_len += len(ref)
        correct_per, per = per_evaluator.evaluate(pred, ref)
        total_corr_per += correct_per
        total_per += per
        correct_ed, dist = basic_evaluator.evaluate(pred, ref)
        total_corr_ed += correct_ed
        total_ed += dist
        
        #print("corr: {}, per: {}".format(correct, per))
    print("total predictions: {}, total correct: {}".format(total, total_corr_per))
    print("WER: {:.2f}".format( ((total-total_corr_ed)/total)*100) )
    print("average PER: {:.5f}".format(total_per/total))
    print("average accuracy: {:.2f}".format((total_corr_per/total)*100))
    print("average ED {:.4f}".format(total_ed/total))
    print("avg pred len {}, avg ref len {}".format(p_len/total, r_len/total))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', '-p', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Prediction file.")
    parser.add_argument('--reference', '-r', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Reference file.")
  
    
    args = parser.parse_args()

    main(args.prediction, args.reference)

#https://arxiv.org/pdf/2004.06338.pdf    
#Phoneme   Error  Rate  (PER)is   the   Levenshtein   distance between  the  predicted  phoneme  sequences  and  the  reference phoneme  sequences,  divided  by  the  number  of  phonemes  in the   reference   pronunciation   [20].   In   case   of   multiple pronunciation  samples  for  a  word  in  the  reference  data,  the sample that has the smallest distance to the candidate is used.
#Word Error Rate (WER)is the percentage of words in which the  predicted phoneme  sequence  does  not  exactly  match  any reference pronunciation, the number of word errors is divided by the total number of unique words in the reference.
