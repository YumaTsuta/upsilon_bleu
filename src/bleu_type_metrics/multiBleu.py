import numpy as np
import argparse
import math
from collections import Counter, defaultdict
from nltk.util import ngrams
from scipy.stats import kendalltau, spearmanr, pearsonr
import os
import logging
from logging import getLogger, StreamHandler, DEBUG, INFO
from tqdm import tqdm
import wordPreprocess  # nopep8


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(DEBUG)

ge_logger = wordPreprocess.logger
ge_logger.addHandler(handler)
ge_logger.setLevel(INFO)


def parserSetting(add_help=True):
    parser = argparse.ArgumentParser(
        parents=[wordPreprocess.parserToken(add_help=False)], add_help=add_help)
    parser.add_argument("test_example", type=str,
                        help="the directory which includes test examples")
    parser.add_argument("augmented_reference", type=str,
                        help="the directory which includes retrieved responses for reference responses")
    parser.add_argument("-size", "--reference_size", type=int,
                        default=15, help="the number of reference responses")
    parser.add_argument("-bleu", "--bleu", type=str,
                        default="2", help="the number of n-gram")

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

    max_weights = defaultdict(int)
    for reference, rw in zip(references, ref_weigth):
        reference_counts = (Counter(ngrams(reference, n))) if len(
            reference) >= n else Counter()
        for ngram, count in counts.items():
            max_weights[ngram] = max(
                max_weights[ngram], min(count, reference_counts[ngram])*rw)

    return sum(max_weights.values()) / sum(counts.values())


def calc_delta_bleu(refs, gen, ref_weight, weights):
    max_weight = max(ref_weight)
    p_ns = (MP(gen, refs, ref_weight, i)/max_weight
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


def main(args):
    assert int(
        args.bleu[0]) > 0, f"{args.bleu[0]} (in {args.bleu}) is not number"
    weights = [1]*int(args.bleu[0])
    weights = [w/sum(weights) for w in weights]

    gen = load_sents(os.path.join(args.test_example, "gen.rep"))
    gen = wordPreprocess.tokenize(args, gen, logger=ge_logger)

    with open(os.path.join(args.test_example, "human_score"), mode="r", encoding="utf-8") as f:
        annotation_scores =\
            [[int(score) for score in line.strip().split(",")] for line in f]
    annotation_scores = np.array(annotation_scores)

    #score = np.average(annotation_scores, axis=1)

    ge_logger.setLevel(logging.WARNING)
    logger.info("evaluating...")

    bleu_all = []
    delta_bleu = []
    for i, h in enumerate(tqdm(gen, leave=False, ncols=0)):
        reference = load_sents(os.path.join(
            args.augmented_reference, "candidate_"+str(i)+"_rpl.txt"))[:args.reference_size]
        reference_weight = load_sents(os.path.join(
            args.augmented_reference, "candidate_"+str(i)+"_predictions.txt"))[:args.reference_size]
        assert len(reference) == len(
            reference_weight), "line numbers must be th same"
        reference_weight = [float(s) for s in reference_weight]
        assert max(
            reference_weight) > 0, f"one reference weight must be positive: Line {i}"
        reference = wordPreprocess.tokenize(
            args, reference, logger=ge_logger)
        ones_weight = [1 for _ in reference_weight]

        delta_bleu.append(calc_delta_bleu(
            reference, h, reference_weight, weights))
        bleu_all.append(calc_delta_bleu(
            reference, h, ones_weight, weights))

    return annotation_scores,  bleu_all, delta_bleu


if __name__ == "__main__":
    parser = parserSetting()
    args = parser.parse_args()
    score, bleu_all, delta_bleu = main(args)
    cor_method = {"pearsonr": pearsonr,
                  "spearmanr": spearmanr, "kendalltau": kendalltau}
    calc_method = {"delta": delta_bleu, "normal": bleu_all}

    for cor in ["pearsonr", "spearmanr", "kendalltau"]:
        for scorer in ["delta", "normal"]:
            cor_list = [cor_method[cor](calc_method[scorer], ones_score)[0]
                        for ones_score in score.T]
            print(
                f"method: {cor}, scorer: {scorer}, max: {max(cor_list):.3e}, min: {min(cor_list):.3e}")
    """
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
    """
