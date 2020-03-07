import heapq
import itertools
import collections

import numpy as np
import tensorflow as tf

from functools import partial

OOV_ID = -1


class Word2VecDataset(object):
    """Dataset for generating matrices holding word indices to train Word2Vec
    models.
    """
    def __init__(self,
                 arch='skip_gram',
                 algm='negative_sampling',
                 epochs=1,
                 batch_size=32,
                 max_vocab_size=0,
                 min_count=10,
                 sample=1e-3,
                 window_size=5,
                 fixed_window_size=False):
        """Constructor.

        Args:
          arch: string scalar, architecture ('skip_gram' or 'cbow').
          algm: string scalar: training algorithm ('negative_sampling' or
            'hierarchical_softmax').
          epochs: int scalar, num times the dataset is iterated.
          batch_size: int scalar, the returned tensors in `get_tensor_dict` have
            shapes [batch_size, :].
          max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top
            `max_vocab_size` most frequent words are kept in vocabulary.
          min_count: int scalar, words whose counts < `min_count` are not included
            in the vocabulary.
          sample: float scalar, subsampling rate.
          window_size: int scalar, num of words on the left or right side of
            target word within a window.
        """
        self._arch = arch
        self._algm = algm
        self._epochs = epochs
        self._batch_size = batch_size
        self._max_vocab_size = max_vocab_size
        self._min_count = min_count
        self._sample = sample
        self._window_size = window_size
        self._fixed_window_size = fixed_window_size

        self._iterator_initializer = None
        self._table_words = None  # vocabulary
        self._unigram_counts = None  # words frequency
        self._keep_probs = None  # words keeping probability
        self._corpus_size = None  # number of words
        self._max_depth = None
        self._num_sentences = None  # number of sentences

    @property
    def iterator_initializer(self):
        return self._iterator_initializer

    @property
    def table_words(self):
        return self._table_words

    @property
    def unigram_counts(self):
        return self._unigram_counts

    @property
    def num_sentences(self):
        return self._num_sentences

    def _build_raw_vocab(self, filenames):
        """Builds raw vocabulary.
        Args:
          filenames: list of strings, holding names of text files.
        Returns:
          raw_vocab: a list of 2-tuples holding the word (string) and frequency count (int),
            sorted in descending order of word frequency count.
        """
        map_open = partial(open, encoding="utf-8")
        lines = itertools.chain(*map(map_open, filenames))
        raw_vocab = collections.Counter()
        for line in lines:
            raw_vocab.update(line.strip().split())
        raw_vocab = raw_vocab.most_common()
        if self._max_vocab_size > 0:
            raw_vocab = raw_vocab[:self._max_vocab_size]
        return raw_vocab

    def build_vocab(self, filenames):
        """Builds vocabulary.
        Has the side effect of setting the following attributes:
        - table_words: list of string, holding the list of vocabulary words. Index
            of each entry is the same as the word index into the vocabulary.
        - unigram_counts: list of int, holding word counts. Index of each entry
            is the same as the word index into the vocabulary.
        - keep_probs: list of float, holding words' keep prob for subsampling.
            Index of each entry is the same as the word index into the vocabulary.
        - corpus_size: int scalar, effective corpus size.

        Args:
          filenames: list of strings, holding names of text files.
        """
        raw_vocab = self._build_raw_vocab(filenames)
        raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
        self._corpus_size = sum(list(zip(*raw_vocab))[1])

        self._table_words = []
        self._unigram_counts = []
        self._keep_probs = []
        for word, count in raw_vocab:
            frac = count / float(self._corpus_size)
            # more frequents words have less probability to be kept in order to better balance the dataset
            keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
            keep_prob = np.minimum(keep_prob, 1.0)
            self._table_words.append(word)
            self._unigram_counts.append(count)
            self._keep_probs.append(keep_prob)

    def _build_binary_tree(self, unigram_counts):
        """Builds a Huffman tree for hierarchical softmax. Has the side effect
        of setting `max_depth`.

        Args:
          unigram_counts: list of int, holding word counts. Index of each entry
            is the same as the word index into the vocabulary.

        Returns:
          codes_points: an int numpy array of shape [vocab_size, 2*max_depth+1]
            where each row holds the codes (0-1 binary values) padded to
            `max_depth`, and points (non-leaf node indices) padded to `max_depth`,
            of each vocabulary word. The last entry is the true length of code
            and point (<= `max_depth`).
        """
        vocab_size = len(unigram_counts)
        heap = [[unigram_counts[i], i] for i in range(vocab_size)]
        heapq.heapify(heap)
        for i in range(vocab_size - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, [min1[0] + min2[0], i + vocab_size, min1, min2])

        node_list = []
        max_depth, stack = 0, [[heap[0], [], []]]
        while stack:
            node, code, point = stack.pop()
            if node[1] < vocab_size:
                node.extend([code, point, len(point)])
                max_depth = np.maximum(len(code), max_depth)
                node_list.append(node)
            else:
                point = np.array(list(point) + [node[1]-vocab_size])
                stack.append([node[2], np.array(list(code)+[0]), point])
                stack.append([node[3], np.array(list(code)+[1]), point])

        node_list = sorted(node_list, key=lambda items: items[1])
        codes_points = np.zeros([vocab_size, max_depth*2+1], dtype=np.int32)
        for i in range(len(node_list)):
            length = node_list[i][4]  # length of code or point
            codes_points[i, -1] = length
            codes_points[i, :length] = node_list[i][2]  # code
            codes_points[i, max_depth:max_depth+length] = node_list[i][3] # point
        self._max_depth = max_depth
        return codes_points

    def _prepare_inputs_labels(self, tensor):
        """Set shape of `tensor` according to architecture and training algorithm,
        and split `tensor` into `inputs` and `labels`.

        Args:
          tensor: rank-2 int tensor, holding word indices for prediction inputs
            and prediction labels, returned by `generate_instances` (context windows).

        Returns:
          inputs: rank-2 int tensor, holding word indices for prediction inputs.
          labels: rank-2 int tensor, holding word indices for prediction labels.
        """
        if self._arch == 'skip_gram':
            if self._algm == 'negative_sampling':
                # tensor = [[1,2],[1,3],[1,4]]
                # input = [[1]],[[1]],[[1]]
                # label = [[2]],[[3]],[[4]]
                tensor.set_shape([self._batch_size, 2])
            else:
                tensor.set_shape([self._batch_size, 2*self._max_depth+2])
            inputs = tensor[:, :1]
            labels = tensor[:, 1:]
        else:
            if self._algm == 'negative_sampling':
                # tensor = [[2,3,4,5,0,0,0,0,4,1]]
                # input = [[2,3,4,5,0,0,0,0,4]]
                # label = [[1]] corresponds to the original target word
                tensor.set_shape([self._batch_size, 2*self._window_size+2])
            else:
                tensor.set_shape([self._batch_size, 2*self._window_size+2*self._max_depth+2])
            inputs = tensor[:, :2*self._window_size+1]
            labels = tensor[:, 2*self._window_size+1:]
        return inputs, labels

    def get_tensor_dict(self, filenames):
        """Generates tensor dict mapping from tensor names to tensors.

        Args:
          filenames: list of strings, holding names of text files.

        Returns:
          tensor_dict: a dict mapping from tensor names to tensors with shape being:
            when arch=='skip_gram', algm=='negative_sampling'
              inputs: [N],                    labels: [N]
            when arch=='cbow', algm=='negative_sampling'
              inputs: [N, 2*window_size+1],   labels: [N]
            when arch=='skip_gram', algm=='hierarchical_softmax'
              inputs: [N],                    labels: [N, 2*max_depth+1]
            when arch=='cbow', algm=='hierarchical_softmax'
              inputs: [N, 2*window_size+1],   labels: [N, 2*max_depth+1]
            progress: [N], the percentage of sentences covered so far. Used to
              compute learning rate.
        """
        table_words = self._table_words
        unigram_counts = self._unigram_counts
        keep_probs = self._keep_probs
        if not table_words or not unigram_counts or not keep_probs:
            raise ValueError('`table_words`, `unigram_counts`, and `keep_probs` must',
                             'be set by calling `build_vocab()`')

        if self._algm == 'hierarchical_softmax':
            codes_points = tf.constant(self._build_binary_tree(unigram_counts))
        elif self._algm == 'negative_sampling':
            codes_points = None
        else:
            raise ValueError('algm must be hierarchical_softmax or negative_sampling')

        # returns a lookup table that converts a string tensor into
        # a int corresponding to the index of word in table list
        # ex: 'cat' has index 3 in table, than 'cat' -> 3
        table_words = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(table_words), default_value=OOV_ID)

        # probabilities to keep each word sorted by word index (high frequency -> keep_probs is low)
        keep_probs = tf.constant(keep_probs)

        # number of sentences contained in the corpus, each line corresponds to one sentence
        num_sents = sum([len(list(open(fn))) for fn in filenames]) * self._epochs
        self._num_sentences = num_sents

        # create a dataset containing one sentence each line and [N]
        dataset = tf.data.Dataset.zip((
          tf.data.TextLineDataset(filenames).repeat(self._epochs),
          tf.data.Dataset.from_tensor_slices(tf.range(num_sents) / num_sents)))
        # replace each word with its corresponding int index
        # ex: 'cat eat fish' -> [3 5 2]
        dataset = dataset.map(lambda sent, progress:
                              (get_word_indices(sent, table_words), progress))
        # randomly remove the words that compare too often, words with high frequencies have lower keep_probs.
        # ex: 'cat eat fish' -> [3 5 2] -> [3 2], 'eat' has been removed, depends on frequency and chance
        dataset = dataset.map(lambda indices, progress:
                              (subsample(indices, keep_probs), progress))
        # remove sentences containing 1 or less words (tf.size(indices)=sentence length)
        # ex: 'cat eat fish' -> OK, 'cat' -> REMOVED
        dataset = dataset.filter(lambda indices, progress:
                                 tf.greater(tf.size(indices), 1))
        # replace each word in each sentence with their context window
        # ex: (s1([c1],...,[cn]),...,sn([c1],...,[cn]))
        dataset = dataset.map(lambda indices, progress: (generate_instances(
            indices, self._arch, self._window_size, codes_points,self._fixed_window_size), progress))
        # augment progress so that each sentence has as many progress value, all equals according to the sentence number
        # as its number of words
        # ex: ((s1([c1],...,[cn]),([p1],...[pn])),...,(sn([c1],...,[cn]),([p1],...[pn])))
        dataset = dataset.map(lambda instances, progress: (
            instances, tf.fill(tf.shape(instances)[:1], progress)))
        # flatten the dataset
        # ex: dataset = ((s1([c1],...,[cn]),([p1],...[pn])),...,(sn([c1],...,[cn]),([p1],...[pn])))
        # result = (s1c1,ps1),...,(s1cn,ps1),...,(sncn,psn)
        # sxcy is the context window y of the sentence x and psz is the progress number of the sentence z
        dataset = dataset.flat_map(lambda instances, progress:
                                   tf.data.Dataset.from_tensor_slices((instances, progress)))
        # group the dataset rows into batches
        dataset = dataset.batch(self._batch_size, drop_remainder=True)

        # make an iterator to iterate over each batch of the dataset
        iterator = dataset.make_initializable_iterator()
        self._iterator_initializer = iterator.initializer
        # get the next batch
        tensor, progress = iterator.get_next()
        progress.set_shape([self._batch_size])
        # split the tensor into inputs and labels
        inputs, labels = self._prepare_inputs_labels(tensor)
        if self._arch == 'skip_gram':
            inputs = tf.squeeze(inputs, axis=1)
        if self._algm == 'negative_sampling':
            labels = tf.squeeze(labels, axis=1)
        return {'inputs': inputs, 'labels': labels, 'progress': progress}


def get_word_indices(sent, table_words):
    """Converts a sentence into a list of word indices.

    Args:
    sent: a scalar string tensor, a sentence where words are space-delimited.
    table_words: a `HashTable` mapping from words (string tensor) to word
      indices (int tensor).

    Returns:
    indices: rank-1 int tensor, the word indices within a sentence.
    """
    # split the sentence into words
    # ex: 'cat eat fish' -> ['cat', 'eat', 'fish']
    words = tf.string_split([sent]).values
    # replace each string with its corresponding int index based on its position in table_words
    # ex: ['cat', 'eat', 'fish'] -> [3 5 2]
    indices = tf.to_int32(table_words.lookup(words))
    return indices


def subsample(indices, keep_probs):
    """Filters out-of-vocabulary words and then applies subsampling on words in a
    sentence. Words with high frequencies have lower keep probs.

    Args:
    indices: rank-1 int tensor, the word indices within a sentence.
    keep_probs: rank-1 float tensor, the prob to drop the each vocabulary word.

    Returns:
    indices: rank-1 int tensor, the word indices within a sentence after
      subsampling.
    """
    indices = tf.boolean_mask(indices, tf.not_equal(indices, OOV_ID))
    keep_probs = tf.gather(keep_probs, indices)
    randvars = tf.random_uniform(tf.shape(keep_probs), 0, 1)
    indices = tf.boolean_mask(indices, tf.less(randvars, keep_probs))
    return indices


def generate_instances(indices, arch, window_size, codes_points=None, fixed_window_size=False):
    """Generates matrices holding word indices to be passed to Word2Vec models
    for each sentence. The shape and contents of output matrices depends on the
    architecture ('skip_gram', 'cbow') and training algorithm ('negative_sampling'
    , 'hierarchical_softmax').

    It takes as input a list of word indices in a subsampled-sentence, where each
    word is a target word, and their context words are those within the window
    centered at a target word. For skip gram architecture, `num_context_words`
    instances are generated for a target word, and for cbow architecture, a single
    instance is generated for a target word.

    If `codes_points` is not None ('hierarchical softmax'), the word to be
    predicted (context word for 'skip_gram', and target word for 'cbow') are
    represented by their 'codes' and 'points' in the Huffman tree (See
    `_build_binary_tree`).

    Args:
    indices: rank-1 int tensor, the word indices within a sentence after
      subsampling.
    arch: scalar string, architecture ('skip_gram' or 'cbow').
    window_size: int scalar, num of words on the left or right side of
      target word within a window.
    codes_points: None, or an int tensor of shape [vocab_size, 2*max_depth+1]
      where each row holds the codes (0-1 binary values) padded to `max_depth`,
      and points (non-leaf node indices) padded to `max_depth`, of each
      vocabulary word. The last entry is the true length of code and point
      (<= `max_depth`).

    Returns:
    instances: an int tensor holding word indices, with shape being
      when arch=='skip_gram', algm=='negative_sampling'
        shape: [N, 2]
      when arch=='cbow', algm=='negative_sampling'
        shape: [N, 2*window_size+2]
      when arch=='skip_gram', algm=='hierarchical_softmax'
        shape: [N, 2*max_depth+2]
      when arch=='cbow', algm='hierarchical_softmax'
        shape: [N, 2*window_size+2*max_depth+2]
    """
    def per_target_fn(index, init_array):
        """IMPORTANT: all the created context windows will be centered around the target word but
        can have a variable dimension between (1, window_size). Both left and right part of the window will
        have window_size-reduced_size words."""
        # index is the index of the target word
        # create a int random number between 0 and maxval excluded
        if not fixed_window_size:
            reduced_size = tf.random_uniform([], maxval=window_size, dtype=tf.int32)
        else:
            reduced_size = tf.constant(0)
        # set the left side of the current window
        left = tf.range(tf.maximum(index - window_size + reduced_size, 0), index)
        # set the right side of the current window
        right = tf.range(index + 1, tf.minimum(index + 1 + window_size - reduced_size, tf.size(indices)))
        # get the indices of the words belonging to the context window respect to their position in the sentence
        # ex: index = 10, windows_size = 2, reduced_size = 0
        # context = [8, 9, 11, 12]
        context = tf.concat([left, right], axis=0)
        # get the indices of the words belonging to the context window respect to their position in the vocabulary
        # ex: vocabulay = {'cat'=3, 'eat'=5, 'fish'=2}
        # result: context = [0, 2] -> [3, 2] with window_size=1 and sentence = 'cat eat fish'
        context = tf.gather(indices, context)

        if arch == 'skip_gram':
            # set the typical skip-gram architecture
            # ex: context = [3, 2], index=5, window = [[5, 3], [5, 2]]
            window = tf.stack([tf.fill(tf.shape(context), indices[index]),
                            context], axis=1)
        elif arch == 'cbow':
            true_size = tf.size(context)
            # create an array = [[context + padding + real_context_size + target word]]
            # ex: [3,2,0,0,2,5] (3 and 2 are context words (length 2) and target word is 5)
            # this is because windows at the limit of a sentence can have shorter windows size,
            # thus the left position on the right are padded with 0 (2*window_size-true_size).
            # true_size is the size of the window excluding the padding
            window = tf.concat([tf.pad(context, [[0, 2*window_size-true_size]]),
                               [true_size, indices[index]]], axis=0)
            window = tf.expand_dims(window, axis=0)
        else:
            raise ValueError('architecture must be skip_gram or cbow.')

        if codes_points is not None:
            window = tf.concat([window[:, :-1],
                               tf.gather(codes_points, window[:, -1])], axis=1)
        # return the index of the next word in the sentence and append the context window in the init_array
        return index + 1, init_array.write(index, window)

    size = tf.size(indices)  # get sentence length
    # create an array of the same size of the sentence
    init_array = tf.TensorArray(tf.int32, size=size, infer_shape=False)
    # loops over all words to create their context window and store it in result_array
    _, result_array = tf.while_loop(lambda i, ta: i < size, per_target_fn, [0, init_array], back_prop=False)
    instances = tf.to_int64(result_array.concat())
    # instances is a concatenation of arrays, one for each word, representing their context windows [[c1][c2]...[cn]]
    return instances

