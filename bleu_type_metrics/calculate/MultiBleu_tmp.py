import numpy as np
import io
import random
from collections import defaultdict
import argparse
from nltk import bleu_score
import math
from collections import Counter
from nltk.util import ngrams
from scipy.stats import kendalltau, spearmanr, pearsonr
import os
import logging
from logging import getLogger, StreamHandler, DEBUG, INFO
from tqdm import tqdm
import GetEmbeddings  # nopep8

random.seed(0)

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = GetEmbeddings.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(INFO)


def parserSetting(add_help=True):
    parser = argparse.ArgumentParser(
        parents=[GetEmbeddings.parserToken(add_help=False)], add_help=add_help)
    parser.add_argument("-pseudo", type=str, required=True,
                        help="path/to/pseudos(dir)")
    parser.add_argument("-r_tgt", "--reference_tgt", type=str,
                        required=True, help="path/to/tgt/of/reference(file)")
    parser.add_argument("-gen_tgt", "--generated_tgt", type=str,
                        required=True, help="path/to/generation/tgt(dir)")
    parser.add_argument("-score", type=str, required=True,
                        help="path/to/annotation/score(dir)")
    parser.add_argument("-threshold_score",
                        "--pseudo_threshold_score", type=float, default=-1.0)
    parser.add_argument(
        "-threshold_sim", "--pseudo_threshold_sim", type=float, default=0.0)
    parser.add_argument("-size", "--pseudo_size", type=int, default=10000)
    parser.add_argument("-correction", action="store_true", default=False)
    parser.add_argument("-bleu", "--bleu", type=str,
                        default="2", help="n-gram: input 'n' or 'n-m'")
    parser.add_argument("-score_agr", "--score_aggregation", type=str, choices=["avg", "median"],
                        default="avg", help="aggregation method of annotation scores")

    parser.add_argument("-pred", "--pred_method", type=str, choices=["max", "avg", "one", "another"],
                        default="max", help="calclation method of prediction score")

    args = parser.parse_known_args()
    assert - \
        1.0 <= args[0].pseudo_threshold_sim <= 1.0, "'-threshold', '--pseudo_threshold' must be in range [-1.0, 1.0]"

    if args[0].tokenize_method == "sentencepiece":
        assert args[0].pretrained, "take option -pretrained"
        assert len(
            args[0].train_file) != 0, "no tokenize model take option -train"
    return parser


def BP(candidate, references):
    """
    calculate brevity penalty
    """
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def MP(hypothesis, references, ref_weigth, n):
    """
    calculate modified precision
    """
    counts = Counter(ngrams(hypothesis, n)) if len(
        hypothesis) >= n else Counter()
    if not counts:
        return 0

    max_counts = {}
    max_prob = Counter()
    for reference, rw in zip(references, ref_weigth):
        reference_counts = (Counter(ngrams(reference, n))) if len(
            reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(
                ngram, 0), reference_counts[ngram])
            if reference_counts[ngram] != 0:
                max_prob[ngram] = rw

    clipped_counts = dict((ngram, min(
        count, max_counts[ngram])*max_prob[ngram]) for ngram, count in counts.items())
    return sum(clipped_counts.values()) / sum(counts.values())


def calc_bleu(ref, gen, weights):
    return bleu_score.sentence_bleu([ref], gen, weights=weights)


def calc_delta_bleu(ref, gen, pseudo, pseudo_score, weights):
    refs = [ref]+pseudo
    ref_weigth = [1]+pseudo_score
    p_ns = (MP(gen, refs, ref_weigth, i)
            for i, _ in enumerate(weights, start=1))
    s = []
    for w, p_n in zip(weights, p_ns):
        try:
            s.append(w * math.log(p_n))
        except ValueError:
            s.append(-999999)
    s = math.fsum(s)

    bp = BP(gen, refs)
    return bp * math.exp(s)


def load_sents(file_path):
    f = open(file_path, "r")
    return [sent.strip("\r\n") for sent in f.readlines()]


def getPseudos(pseudo_tgt, gen_tgt):
    gen_tgt = GetEmbeddings.main(args, gen_tgt, logger=ge_logger)
    pseudo_tgt = GetEmbeddings.main(args, pseudo_tgt, logger=ge_logger)
    for i, (r, h) in enumerate(zip(pseudo_tgt, gen_tgt)):
        yield i, r, h


def main(args):
    args.vectorize_method = None

    assert int(
        args.bleu[0]) > 0, f"{args.bleu[0]} (in {args.bleu}) is not number"
    if len(args.bleu) > 2:
        assert int(
            args.bleu[2]) > 0, f"{args.bleu[2]} (in {args.bleu}) is not number"
        assert int(args.bleu[2]) < int(
            args.bleu[0]), f"{args.bleu[2]} is same or larger than {args.bleu[0]}"
        weights = [0]*int(args.bleu[2])+[1] * \
            (int(args.bleu[0])-int(args.bleu[2]))
    else:
        weights = [1]*int(args.bleu[0])
    weights = [w/sum(weights) for w in weights]

    gen = load_sents(args.generated_tgt)
    gen = GetEmbeddings.main(args, gen, logger=ge_logger)
    total = len(gen)

    tgt = load_sents(args.reference_tgt)
    tgt = GetEmbeddings.main(args, tgt, logger=ge_logger)

    assert len(tgt) == len(gen)
    iterator = getPseudos(tgt, gen)

    with open(args.score, mode="r", encoding="utf-8-sig") as f:
        annotation_scores =\
            [[int(score) for score in line.strip().split(",")]
             for line in f.readlines()]
        annotation_scores = np.array(annotation_scores)

    if args.score_aggregation == "avg":
        score = np.average(annotation_scores, axis=1)
    elif args.score_aggregation == "median":
        score = np.median(annotation_scores, axis=1)

    ge_logger.setLevel(logging.WARNING)

    logger.info("evaluating...")
    pseudo_use_sizes = []
    bleu = []
    bleu_all = []
    delta_bleu = []
    for i, r, h in tqdm(iterator, total=total):
        pseudos_score = load_sents(os.path.join(args.pseudo, "candidate_"+str(i)+"_"+args.pred_method+"_predictions.txt"))[:args.pseudo_size]  # nopep8
        pseudos_tgt = load_sents(os.path.join(
            args.pseudo, "candidate_"+str(i)+"_rpl.txt"))[:args.pseudo_size]
        pseudos_sim = load_sents(os.path.join(
            args.pseudo, "candidate_"+str(i)+"_sim.txt"))[:args.pseudo_size]
        assert len(pseudos_score) == len(
            pseudos_tgt), "line numbers must be th same"
        pseudos_sim = [float(s) for s in pseudos_sim]
        pseudos_score = [(abs(float(s))-0.5)*2*(float(s) // abs(float(s)))
                         for s in pseudos_score]
        pseudos_tgt = [t for (t, s, sim) in zip(
            pseudos_tgt, pseudos_score, pseudos_sim) if s >= args.pseudo_threshold_score and sim >= args.pseudo_threshold_sim]
        pseudos_tgt = GetEmbeddings.main(args, pseudos_tgt, logger=ge_logger)
        pseudos_score = [s for s, sim in zip(pseudos_score, pseudos_sim) if s >=
                         args.pseudo_threshold_score and sim > args.pseudo_threshold_sim]
        ones_score = [1 for _ in pseudos_score]
        pseudo_use_sizes.append(len(pseudos_score))

        bleu.append(calc_bleu(r, h, weights))
        delta_bleu.append(calc_delta_bleu(
            r, h, pseudos_tgt, pseudos_score, weights))
        bleu_all.append(calc_delta_bleu(
            r, h, pseudos_tgt, ones_score, weights))
    logger.info(f"use {sum(pseudo_use_sizes)/len(pseudo_use_sizes)} pseudos")
    return score, bleu, bleu_all, delta_bleu


if __name__ == "__main__":
    parser = parserSetting()
    args = parser.parse_args()
    score, bleu, bleu_all, delta_bleu = main(args)
    cor_pear = [pearsonr(score, bleu), pearsonr(
        score, bleu_all), pearsonr(score, delta_bleu)]
    cor_spear = [spearmanr(score, bleu), spearmanr(
        score, bleu_all), spearmanr(score, delta_bleu)]
    cor_kend = [kendalltau(score, bleu), kendalltau(
        score, bleu_all), kendalltau(score, delta_bleu)]
    print_sent = f"pearson\nbleu :{cor_pear[0]}\n bleu_all :{cor_pear[1]}\n delta_bleu:{cor_pear[2]}\n\n"
    print_sent += f"spearman\nbleu :{cor_spear[0]}\n bleu_all :{cor_spear[1]}\n delta_bleu:{cor_spear[2]}\n\n"
    print_sent += f"kendalltau\nbleu :{cor_kend[0]}\n bleu_all :{cor_kend[1]}\n delta_bleu:{cor_kend[2]}\n\n"
    print(print_sent)
