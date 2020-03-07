from scipy.stats import spearmanr
from scipy import spatial
import numpy as np
import pandas as pd


if __name__ == '__main__':
    embed_path = "output_small_wiki/embed.npy"
    vocab_path = "output_small_wiki/vocab.txt"
    embed = np.load(embed_path)
    with open(vocab_path, encoding="utf8") as f:
        vocab = f.readlines()
    vocab = [w.strip() for w in vocab]

    def lookup_table(word):
        return embed[vocab.index(word)]

    word_analogy_path = "wordSim353/set1.csv"
    word_analogy = pd.read_csv(word_analogy_path).iloc[:, :3]

    spearman_err = 0
    cosine_err = 0

    analogies = 0
    for index, row in word_analogy.iterrows():
        if row["Word 1"] not in vocab or row["Word 2"] not in vocab:
            continue

        spearman_corr, _ = spearmanr(lookup_table(row["Word 1"]), lookup_table(row["Word 2"]))
        spearman_corr = abs(spearman_corr)
        spearman_err += abs(spearman_corr - row["Human (mean)"] / 10)

        cosine_corr = 1 - spatial.distance.cosine(lookup_table(row["Word 1"]), lookup_table(row["Word 2"]))
        cosine_corr = abs(cosine_corr)
        cosine_err += abs(cosine_corr - row["Human (mean)"] / 10)
        print(float(row["Human (mean)"]))

        print(row["Word 1"], row["Word 2"], "%.3f" % spearman_corr, "%.3f" % cosine_corr,
              "%.3f" % (float(row["Human (mean)"]) / 10))
        analogies += 1

    print("Average error on Spearman correlation: ", spearman_err / analogies)
    print("Average error on Cosine similarity: ", cosine_err / analogies)

