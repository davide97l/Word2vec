import nltk

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
import numpy as np
from numpy import average

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy import spatial
import os
import argparse


def get_txt_lines(path):
    with open(path, encoding="utf8") as f:
        lines = f.readlines()
    lines = [w.strip() for w in lines]
    return lines


def valid_pos_tag(tag):
    if tag.startswith('J') or tag.startswith('V') or tag.startswith('N') or tag.startswith('R'):
        return True
    return False


def cosine_similarity(w1, w2):
    cosine_sim = 1 - spatial.distance.cosine(w1, w2)
    return cosine_sim


def sentence_disambiguation(sentence, embed, vocab, cosine_sim_threshold=0.05, score_margin_threshold=0.05):
    """Disambiguate the words of sentence
    Args:
      sentence: string, sentence to be disambiguated
      embed: matrix containing the words embedding, 1 word embedding each line
      vocab: list containing vocabulary words
      cosine_sim_threshold: words correlation threshold
      score_margin_threshold: sense correlation margin
    Returns:
      original_embed: the embedding of the original sentence with some words disambiguated
      definition_vocab: vocabulary containing the definitions of the disambiguated words as items and keys are their
        corresponding indices in the sentence
    """
    def lookup_table(word):
        return embed[vocab.index(word)]

    stop_words = set(stopwords.words('english'))
    sentence = sentence.lower()
    word_tokens = word_tokenize(sentence)
    # store the original sentence
    original_tokens = word_tokens
    original_embed = [lookup_table(w) for w in word_tokens]
    if len(word_tokens) == 0:
        raise Exception('A sentence can\'t be empty')
    definitions_vocab = {}
    # remove stop words (ex: to, the, a, an, in...)
    word_tokens = [w for w in word_tokens if w not in stop_words]
    # get the pos-tag for each word
    tags = nltk.pos_tag(word_tokens)
    # filter words keeping only nouns (N), verbs (V), adjective (J) and adverbs (R)
    word_tokens = [word for word, tag in tags if valid_pos_tag(tag)]
    # get the context vector of the current sentence as the average of all of its words
    try:
        embed_words = [lookup_table(w) for w in word_tokens]
    except Exception:
        raise Exception('The sentence contains unknown words')
    # compute the context vector
    context_vec = average(embed_words, 0)
    # from left to right disambiguate each word
    word_tokens_count = -1
    for i, word in enumerate(original_tokens):
        if word not in word_tokens:
            continue
        word_tokens_count += 1
        # dictionary: {sense: sense_vector}
        syn_vectors = {}
        # dictionary: {sense: cosine_similarity}
        cos_vectors = {}
        # dictionary: {sense: Lemma(sense}
        lemma_vectors = {}
        for sense in wn.lemmas(word):
            # gloss is a list containing the definition of each sense and some examples
            gloss = [sense.synset().definition()]
            gloss.extend(sense.synset().examples())
            # get all words contained in the gloss
            gloss_tokens = nltk.word_tokenize(" ".join(gloss))
            gloss_tags = nltk.pos_tag(gloss_tokens)
            # filter words gloss keeping only nouns (N), verbs (V), adjective (J) and adverbs (R)
            gloss_tokens = [word for word, tag in gloss_tags if valid_pos_tag(tag)]
            # we are going to store in this array all words in the gloss correlated with the target word
            # correlation: cosine_similarity >= cosine_sim_threshold
            sense_word_vectors = []
            for t in gloss_tokens:
                try:
                    gloss_word_vec = lookup_table(t)
                except Exception:
                    continue
                # cosine similarity between the embedding of a gloss word and the target word
                cos_sim = cosine_similarity(gloss_word_vec, original_embed[i])
                if cos_sim >= cosine_sim_threshold:
                    sense_word_vectors.append(gloss_word_vec)
            if len(sense_word_vectors) == 0:
                continue
            # get the average of these word vectors and append it to the senses dictionary with its corresponding
            # cosine similarity with the context vector
            sense_vector = average(sense_word_vectors, 0)
            syn_vectors[str(sense)] = sense_vector
            cos_vectors[str(sense)] = cosine_similarity(sense_vector, context_vec)
            lemma_vectors[str(sense)] = sense

        if len(syn_vectors) == 0:
            continue
        sorted_list = sorted(cos_vectors.items(), key=lambda x: x[1])
        # find the sense vector in the dictionary that is closer to the context vector
        most_similar_pair = sorted_list.pop()
        disambiguated_sense = most_similar_pair[0]
        cos_sim_second_most_similar_sense = 0
        if len(sorted_list) > 0:
            cos_sim_second_most_similar_sense = sorted_list.pop()[1]
        score_margin = most_similar_pair[1] - cos_sim_second_most_similar_sense
        # if there are more senses, make sure the first choice is much better than the second best one
        if score_margin >= score_margin_threshold:
            # replace the sense vector in the word embedding
            if word_tokens_count >= len(embed_words):
                continue
            embed_words[word_tokens_count] = syn_vectors[disambiguated_sense]
            original_embed[i] = syn_vectors[disambiguated_sense]
            # recompute the new context vector
            context_vec = average(embed_words, 0)
            definitions_vocab[i] = lemma_vectors[disambiguated_sense].synset().definition()

    return original_embed, definitions_vocab


if __name__ == '__main__':
    """
    python word_sense_disambiguation.py -e output_wiki_s_300/embed.npy -v output_wiki_s_300/vocab.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embed_path", type=str, required=True,
                    help="path to the embedding (embedding.npy)")
    ap.add_argument("-v", "--vocab_path", type=str, required=True,
                    help="path to the vocabulary (vocabulary.txt)")
    ap.add_argument("-r", "--rating_path", type=str, default="SCWS/ratings.txt",
                    help="path to the vocabulary (rating.txt)")
    ap.add_argument("-s", "--save_path", type=str, default="results/",
                    help="path where to save the disambiguation results")
    ap.add_argument("-c", "--cosine_sim_threshold", type=float, default=0.05,
                    help="cosine_sim_threshold")
    ap.add_argument("-t", "--score_margin_threshold", type=float, default=0.05,
                    help="score_margin_threshold")
    args = vars(ap.parse_args())

    vocab_path = args["vocab_path"]
    embed_path = args["embed_path"]
    rating_path = args["rating_path"]
    save_path = args["save_path"]
    score_margin_threshold = args["score_margin_threshold"]
    cosine_sim_threshold = args["cosine_sim_threshold"]

    vocab = get_txt_lines(vocab_path)
    embed = np.load(embed_path)
    rating = get_txt_lines(rating_path)

    """x, _ = sentence_disambiguation("open an account to deposit money in bank", embed, vocab)
    y, yy = sentence_disambiguation("ask your bank for a loan", embed, vocab, score_margin_threshold=0.01)
    z, _ = sentence_disambiguation("the boat is on the bank of the river", embed, vocab)
    print(np.array_equal([x[7]], [y[2]])) #true
    print(np.array_equal([x[7]], [z[8]])) #false
    """

    tot_err = 0
    sentences = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, "SCWS.txt"), "w+")

    for line in rating:

        line = [splits for splits in line.split("\t") if splits is not ""]
        w1 = line[1].lower()
        w2 = line[3].lower()
        if w1 not in vocab or w2 not in vocab:
            continue
        sentence1 = line[5].lower()
        sentence2 = line[6].lower()
        sentence1_split = [splits for splits in sentence1.split(" ") if splits in vocab]
        sentence2_split = [splits for splits in sentence2.split(" ") if splits in vocab]

        sentence1_clean = " ".join(sentence1_split)
        sentence2_clean = " ".join(sentence2_split)

        idx_w1 = sentence1_split.index(w1)
        idx_w2 = sentence2_split.index(w2)
        e1, _ = sentence_disambiguation(sentence1_clean, embed, vocab, cosine_sim_threshold, score_margin_threshold)
        sense1 = e1[idx_w1]
        e2, _ = sentence_disambiguation(sentence2_clean, embed, vocab, cosine_sim_threshold, score_margin_threshold)
        sense2 = e2[idx_w2]
        dist1_2 = cosine_similarity(sense1, sense2)

        values = line[-11:-1]
        values = [float(x) for x in values]
        real_value = average(values, 0) / 10
        err = abs(dist1_2 - real_value)
        tot_err += err
        print(sentences, w1, w2, dist1_2, real_value, err)
        f.write("{} {} {} {} {} {}\n".format(sentences, w1, w2, dist1_2, real_value, err))
        sentences += 1

    print("Average error on SCWS dataset: ", tot_err / sentences)
    f.close()
