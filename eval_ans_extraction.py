import re
import os
import sys
import time
import json
import random
import logging
import argparse
import torch
from tqdm import tqdm

from shutil import copyfile
from datetime import datetime
from collections import Counter

#from qa.model import DocReaderModel
from qa.model_ans_extraction import DocReaderModel
from qa.utils import *
from qa.score_ans_extraction import score_ans_extraction

parser = argparse.ArgumentParser(
    description='Eval a QA model.'
)
# system
parser.add_argument('--save_dir', default='save/debug',
                    help='path to store saved models.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument("--debug", action='store_true',
                    help='debug mode')

# eval
parser.add_argument('-bs', '--eval_batch_size', type=int, default=1,
                    help='batch size for evaluation (default: 1)')
parser.add_argument('-rs', '--resume', default='best_model.pt',
                    help='previous model file name (in `save_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('--max_eval_len', type=int, default=0,
                    help='max len for evaluation (default: 0, i.e. unlimited)')
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


def split_batch_dict(d):
    batch_size = len(d[list(d.keys())[0]])
    assert all([len(v) == batch_size for k, v in d.items()])
    split_dicts = [{k: None for k in d} for _ in range(batch_size)]
    for k in d:
        for i in range(batch_size):
            split_dicts[i][k] = d[k][i]
    return split_dicts


def main():
    log.info('[program starts.]')
    log.info('[loading previous model...]')
    checkpoint = torch.load(os.path.join(args.save_dir, args.resume))
    # reset ans_top_k
    if args.ans_top_k is None:
        args.ans_top_k = checkpoint['config']['ans_top_k'] if 'ans_top_k' in checkpoint['config'] else 20
    checkpoint['config'].update(vars(args))
    opt = checkpoint['config']
    print("ans_top_k:", opt['ans_top_k'])
    log.info('[loading data...]')
    train, dev, train_y, dev_y, embedding, opt, meta = load_data(opt, log)
    log.info('[Data loaded.]')
    log.info('train_size: {}, dev_size: {}'.format(len(train), len(dev)))

    state_dict = checkpoint['state_dict']
    model = DocReaderModel(opt, embedding, state_dict)
    epoch_0 = checkpoint['epoch'] + 1
    for i in range(checkpoint['epoch']):
        random.shuffle(list(range(len(train))))  # synchronize random seed
    log.info('opt: {}'.format(opt))

    if args.cuda:
        model.cuda()

    log.info('model:\n{}'.format(model.network))


    batches = BatchGen(opt, dev, batch_size=opt['eval_batch_size'], evaluation=True, max_len=args.max_eval_len, gpu=args.cuda, with_cids=False)
    predictions = []
    start = time.perf_counter()
    for batch in tqdm(batches):
        """
        output_dict = {
            "span_start_logits",
            "span_end_logits",
            "best_span",
            "topk_span",
            "topk_logprob",
            "best_span_str",
            "topk_span_str"}
        """
        predictions.extend(split_batch_dict(model.predict(batch, full_info=True)))
    torch.cuda.synchronize()
    eval_time = time.perf_counter() - start
    em, f1, topk_em, topk_f1, topk_recall = score_ans_extraction(predictions, dev_y)
    log.info("[dev EM: {} F1: {} TopK_EM: {} TopK_F1: {} TopK_Recall: {} eval_time: {:.2f} s eval_time per example: {:.3f} ms]".format(em, f1, topk_em, topk_f1, topk_recall, eval_time, eval_time * 1000. / len(dev)))


if __name__ == '__main__':
    main()

