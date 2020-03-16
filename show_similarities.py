import numpy as np
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_similarities(vectors, labels, arrows=True):
    pca = PCA(n_components=2)
    data = pca.fit_transform(vectors)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.plot(data[:, 0], data[:, 1], '.')
    if labels is not None:
        for i in range(len(data)):
            plt.annotate(labels[i], xy=data[i])
    if arrows:
        for i in range(len(data) // 2):
            plt.annotate("",
                         xy=data[i],
                         xytext=data[i + len(data) // 2],
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="arc3")
                         )
    else:
        for i in range(len(data) // 2):
            plt.annotate("",
                         xy=data[i],
                         xytext=data[i + len(data) // 2],
                         )
    plt.show()


if __name__ == '__main__':
    # python show_similarities.py -e output_wiki_m_200/embed.npy -v output_wiki_m_200/vocab.txt
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embed_path", type=str, required=True,
                    help="path to the embedding (embedding.npy)")
    ap.add_argument("-v", "--vocab_path", type=str, required=True,
                    help="path to the vocabulary (vocabulary.txt)")
    args = vars(ap.parse_args())

    embed_path = args["embed_path"]
    vocab_path = args["vocab_path"]

    embed = np.load(embed_path)
    with open(vocab_path, encoding="utf8") as f:
        vocab = f.readlines()
    vocab = [w.strip() for w in vocab]

    def lookup_table(word):
        return embed[vocab.index(word)]

    countries = ['china', 'canada', 'germany', 'usa', 'italy', 'japan', 'turkey', 'france']
    capitals = ['beijing', 'ottawa', 'berlin', 'washington', 'rome', 'tokyo', 'istanbul', 'paris']
    labels = countries + capitals
    vectors = [lookup_table(w) for w in labels]
    plot_similarities(vectors, labels)

    names = ['paul', 'john', 'mark', 'stephen', 'paula', 'jane', 'emma', 'sophie']
    vectors = [lookup_table(w) for w in names]
    plot_similarities(vectors, names, arrows=False)

    verbs = ['eat', 'launch', 'understand', 'be', 'think', 'sleep', 'get', 'write']
    past = ['ate', 'launched', 'understood', 'was', 'thought', 'slept', 'got', 'wrote']
    labels = verbs + past
    vectors = [lookup_table(w) for w in labels]
    plot_similarities(vectors, labels, arrows=True)

    names = ['ronaldo', 'messi', 'pellegrini', 'bekele', 'ming', 'vettel', 'tyson', 'kobayashi']
    country = ['portugal', 'argentina', 'italy', 'ethiopia', 'china', 'germany', 'usa', 'japan']
    labels = names + country
    vectors = [lookup_table(w) for w in labels]
    plot_similarities(vectors, labels, arrows=True)

    cities = ['rome', 'milan', 'venice', 'florence', 'beijing', 'shanghai', 'wuhan', 'shenzhen',
              'philadelphia', 'washington', 'houston', 'boston']
    vectors = [lookup_table(w) for w in cities]
    plot_similarities(vectors, cities, arrows=False)



