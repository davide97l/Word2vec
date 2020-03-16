from scipy.stats import spearmanr
from scipy import spatial
import numpy as np
import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW, fetch_MTurk
from web.datasets.analogy import fetch_google_analogy
from itertools import chain
import argparse
import os


if __name__ == '__main__':
    # python embedding_eval.py -e output_wiki_m_300/embed.npy -v output_wiki_m_300/vocab.txt -sv results/ -s -a
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embed_path", type=str, required=True,
                    help="path to the embedding (embedding.npy)")
    ap.add_argument("-v", "--vocab_path", type=str, required=True,
                    help="path to the vocabulary (vocabulary.txt)")
    ap.add_argument("-s", "--similarity", default=False, action='store_true',
                    help="compute similarity score")
    ap.add_argument("-a", "--analogy", default=False, action='store_true',
                    help="compute analogy score")
    ap.add_argument("-sv", "--save_path", type=str, default="results/",
                    help="path where to save the analogy results")
    args = vars(ap.parse_args())

    embed_path = args["embed_path"]
    vocab_path = args["vocab_path"]
    similarity = args["similarity"]
    save_path = args["save_path"]
    analogy = args["analogy"]
    embed = np.load(embed_path)
    with open(vocab_path, encoding="utf8") as f:
        vocab = f.readlines()
    vocab = [w.strip() for w in vocab]

    def lookup_table(word):
        return embed[vocab.index(word)]

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, "analogy-smiliarity.txt"), "w+")

    if similarity:

        # Define tasks
        tasks = {
            "WS353": fetch_WS353(),
            "MEN": fetch_MEN(),
            "SIMLEX999": fetch_SimLex999(),
            "RW": fetch_RW(),
            "MTurk": fetch_MTurk()
        }

        spearman_errors = []
        cosine_errors = []
        print("----------SIMILARITY----------")
        f.write("----------SIMILARITY----------\n")
        for name, data in iteritems(tasks):
            # print("Sampling data from ", name)
            spearman_err = 0
            cosine_err = 0
            analogies = 0
            for i in range(len(data.X)):
                word1, word2 = data.X[i][0], data.X[i][1]
                if word1 not in vocab or word2 not in vocab:
                    continue

                spearman_corr, _ = spearmanr(lookup_table(word1), lookup_table(word2))
                spearman_corr = abs(spearman_corr)
                spearman_err += abs(spearman_corr - data.y[i] / 10)

                cosine_sim = 1 - spatial.distance.cosine(lookup_table(word1), lookup_table(word2))
                cosine_err += abs(cosine_sim - data.y[i] / 10)
                # print(word1, word2, data.y[i], cosine_sim)

                analogies += 1
            spearman_err = 1 - spearman_err / analogies
            cosine_err = 1 - cosine_err / analogies
            spearman_errors.append(spearman_err)
            cosine_errors.append(cosine_err)
            print("Spearman correlation error on {} dataset: {}".format(name, spearman_err))
            f.write("Spearman correlation error on {} dataset: {}\n".format(name, spearman_err))
            print("Cosine similarity error on {} dataset: {}".format(name, cosine_err))
            f.write("Cosine similarity error on {} dataset: {}\n".format(name, cosine_err))

    if analogy:

        # Fetch analogy dataset
        data = fetch_google_analogy()

        word_embed = dict(zip(vocab, embed))

        print("----------ANALOGY----------")
        f.write("----------ANALOGY----------\n")
        # Pick a sample of data and calculate answers
        guessed = 0
        subset = list(chain(range(50, 70), range(1000, 1020), range(4000, 4020), range(10000, 10020),
                      range(14000, 14020)))
        for id in subset:
            w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
            if w1 not in vocab or w2 not in vocab or w3 not in vocab:
                continue
            print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
            f.write("Question: {} is to {} as {} is to ?\n".format(w1, w2, w3))
            print("Answer: " + data.y[id])
            f.write("Answer: {}\n".format(data.y[id]))
            s = lookup_table(w2) - lookup_table(w1) + lookup_table(w3)
            best_match = 0.
            best_index = 0

            for i, (w, e) in enumerate(word_embed.items()):
                if w == w1 or w == w2 or w == w3:
                    continue
                cosine_sim = 1 - spatial.distance.cosine(s, e)
                if cosine_sim >= best_match:
                    best_match = cosine_sim
                    best_index = i

            print("Predicted: ", vocab[best_index])
            f.write("Predicted: {}\n".format(vocab[best_index]))
            if vocab[best_index] == data.y[id]:
                guessed += 1

        print("Questions correctly answered: {} / {}".format(guessed, len(subset)))
        f.write("Questions correctly answered: {} / {}\n".format(guessed, len(subset)))
    f.close()
