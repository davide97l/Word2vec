from scipy.stats import spearmanr
from scipy import spatial
import numpy as np
import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW, fetch_MTurk
from web.datasets.analogy import fetch_google_analogy
from itertools import chain
from word2vec import WordVectors
import argparse


if __name__ == '__main__':
    # python embedding_eval.py -e output_small_wiki_300/embed.npy -v output_small_wiki_300/vocab.txt -s -a
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embed_path", type=str, required=True,
                    help="path to the embedding (embedding.npy)")
    ap.add_argument("-v", "--vocab_path", type=str, required=True,
                    help="path to the vocabulary (vocabulary-txt)")
    ap.add_argument("-s", "--similarity", default=False, action='store_true',
                    help="compute similarity score")
    ap.add_argument("-a", "--analogy", default=False, action='store_true',
                    help="compute analogy score")
    args = vars(ap.parse_args())
    embed_path = args["embed_path"]
    vocab_path = args["vocab_path"]
    similarity = args["similarity"]
    analogy = args["analogy"]
    embed = np.load(embed_path)
    with open(vocab_path, encoding="utf8") as f:
        vocab = f.readlines()
    vocab = [w.strip() for w in vocab]

    def lookup_table(word):
        return embed[vocab.index(word)]

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

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
                # print(word1, word2, data.y[i], spearman_corr)

                cosine_sim = 1 - spatial.distance.cosine(lookup_table(word1), lookup_table(word2))
                cosine_sim = abs(cosine_sim)
                cosine_err += abs(cosine_sim - data.y[i] / 10)
                # print(word1, word2, data.y[i], cosine_sim)

                analogies += 1
            spearman_err /= analogies
            cosine_err /= analogies
            spearman_errors.append(spearman_err)
            cosine_errors.append(cosine_err)
            print("Spearman correlation error on {} dataset: {}".format(name, spearman_err))
            print("Cosine similarity error on {} dataset: {}".format(name, cosine_err))

    if analogy:

        # Fetch analogy dataset
        data = fetch_google_analogy()

        # embedding wrapper
        w = WordVectors(embed, vocab)

        print("----------ANALOGY----------")
        # Pick a sample of data and calculate answers
        guessed = 0
        subset = list(chain(range(50, 60), range(1000, 1010), range(4000, 4010), range(10000, 10010),
                      range(14000, 14010)))
        for id in subset:
            w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
            print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
            print("Answer: " + data.y[id])
            s = lookup_table(w2) - lookup_table(w1) + lookup_table(w3)
            worst = 0.
            best_word = None
            temp_embed = list(embed)
            np.delete(temp_embed, vocab.index(w1), 0)
            np.delete(temp_embed, vocab.index(w2), 0)
            np.delete(temp_embed, vocab.index(w3), 0)

            for e in temp_embed:
                cosine_sim = 1 - spatial.distance.cosine(s, e)
                if cosine_sim >= worst:
                    worst = cosine_sim
                    best_word = e

            index = np.where(np.all(embed == best_word, axis=1))[0][0]
            print("Predicted: ", vocab[index])
            if vocab[index] == data.y[id]:
                guessed += 1
        print("Questions correctly answered: {} / {}".format(len(subset), guessed))
