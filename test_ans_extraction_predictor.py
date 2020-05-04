import re
import os
import sys
import time
import json
import pickle
import random
import logging
import argparse
import torch
from tqdm import tqdm

from shutil import copyfile
from datetime import datetime
from collections import Counter

from qa.ans_extraction_predictor import FusionNetPredictor
from qa.utils import *
from qa.score_ans_extraction import score_ans_extraction

parser = argparse.ArgumentParser(
    description='Eval a QA model.'
)
# system
parser.add_argument('--input_path', required=True,
                    help='input dev path')
parser.add_argument('--save_dir', default='save/debug',
                    help='path to store saved models.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
#parser.add_argument("--debug", action='store_true',
#                    help='debug mode')

# eval
#parser.add_argument('-bs', '--eval_batch_size', type=int, default=1,
#                    help='batch size for evaluation (default: 1)')
parser.add_argument('-rs', '--resume', default='best_model.pt',
                    help='previous model file name (in `save_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
#parser.add_argument('--max_eval_len', type=int, default=0,
#                    help='max len for evaluation (default: 0, i.e. unlimited)')
parser.add_argument('--ans_top_k', type=int, default=20,
                    help='number of candidates to consider, will override the value set in the saved config')

args = parser.parse_args()

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)


def main():
    log.info('[program starts.]')
    predictor =  FusionNetPredictor(os.path.join(args.save_dir, args.resume), ans_top_k=args.ans_top_k, cuda=args.cuda)
    log.info('model:\n{}'.format(predictor._model.network))

    with open(args.input_path, 'rb') as f:
        dev_data = pickle.load(f)

    results = []
    start = time.perf_counter()
    for article in tqdm(dev_data):
        results.extend(predictor.predict([article]))
    torch.cuda.synchronize()
    eval_time = time.perf_counter() - start
    dev_y = [r['answer_texts'] for r in results]
    em, f1, topk_em, topk_f1, topk_recall = score_ans_extraction(results, dev_y)
    log.info("[dev EM: {} F1: {} TopK_EM: {} TopK_F1: {} TopK_Recall: {} eval_time: {:.2f} s eval_time per example: {:.3f} ms]".format(em, f1, topk_em, topk_f1, topk_recall, eval_time, eval_time * 1000. / len(dev)))


if __name__ == '__main__':
    main()

