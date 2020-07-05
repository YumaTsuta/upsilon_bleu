import argparse
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("-unref", "--unreferenced",
                    help="the scoring file by unreferenced scorer")
parser.add_argument("-ref", "--referenced",
                    help="the scoring file by referenced scorer")
parser.add_argument("-score", help="the scoring file by human")

args = parser.parse_args()


def loadHumanScore(file_path):
    with open(args.score, mode="r") as f:
        annotation_scores = \
            [[int(score) for score in line.strip().split(",")]
                for line in f.readlines()]
        annotation_scores = np.array(annotation_scores)

    return annotation_scores


def load_sents(file_path):
    f = open(file_path, "r")
    return [sent.strip("\r\n") for sent in f.readlines()]


def loadMetricScore(file_path):
    with open(file_path, "r") as f:
        return np.array([float(s) for scores in f.readlines() for s in scores.strip("\r\n").split(",")])


human_score = loadHumanScore(args.score)
unref = loadMetricScore(args.unreferenced)

cors = []
for comb in itertools.combinations(range(human_score.shape[1]), 2):
    cors.append(spearmanr(human_score[:, comb[0]], human_score[:, comb[1]])[0])
    cors.append(pearsonr(human_score[:, comb[0]], human_score[:, comb[1]])[0])
print("humans: max {}, min{}".format(max(cors), min(cors)))

#cors = [spearmanr(unref, s)[0] for s in human_score.T]
cors = [pearsonr(unref, s)[0] for s in human_score.T]
cor_ave = sum(cors)/len(cors)
cor_max = max(cors)
cor_min = min(cors)

print("US: max :{}, mean: {}, min: {}".format(cor_max, cor_ave, cor_min))


ref = loadMetricScore(args.referenced)

#cors = [spearmanr(ref, s)[0] for s in human_score.T]
cors = [pearsonr(ref, s)[0] for s in human_score.T]
cor_ave = sum(cors)/len(cors)
cor_max = max(cors)
cor_min = min(cors)

print("RS: max :{}, mean: {}, min: {}".format(cor_max, cor_ave, cor_min))


unref = (unref-np.min(unref))/(np.max(unref)-np.min(unref))
ref = (ref-np.min(ref))/(np.max(ref)-np.min(ref))

concat = np.array([unref, ref])
max_score = np.max(concat, axis=0)
min_score = np.min(concat, axis=0)
amean_score = np.sum(concat, axis=0)/2
gmean_score = np.linalg.norm(concat, axis=0)

cors = [spearmanr(amean_score, s)[0] for s in human_score.T]
cors = [pearsonr(amean_score, s)[0] for s in human_score.T]
cor_ave = sum(cors)/len(cors)
cor_max = max(cors)
cor_min = min(cors)

print("max :{}, mean: {}, min: {}".format(cor_max, cor_ave, cor_min))
