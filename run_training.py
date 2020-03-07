"""
Executable for training Word2Vec models.
Example:
  python run_training.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
    --out_dir=/PATH/TO/OUT_DIR/ \
    --batch_size=64 \
    --window_size=5 \

  python run_training.py --filenames=wiki.txt --out_dir=output_wiki/ --window_size=5 --log_per_steps=10000 --embed_size=300

Learned word embeddings will be saved to /PATH/TO/OUT_DIR/embed.npy, and
vocabulary saved to /PATH/TO/OUT_DIR/vocab.txt
"""
import os

import tensorflow as tf
import time

from dataset import Word2VecDataset
from word2vec import Word2VecModel


flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
                    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 1, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
                     'the top `max_vocab_size` most frequent words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 10, 'Words whose counts < `min_count` are not'
                     ' included in the vocabulary.')
flags.DEFINE_float('sample', 1e-3, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 10, 'Num of words on the left or right side' 
                     ' of target word within a window.')
flags.DEFINE_boolean('fixed_window_size', False, 'If False each target word can have a variable' 
                     'context window size ranging between 1 and window_size.')

flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.0001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dot-product '
                     'between syn0 and syn1 vectors.')

flags.DEFINE_integer('log_per_steps', 10000, 'Every `log_per_steps` steps to '
                     ' output logs.')
flags.DEFINE_list('filenames', None, 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', '/tmp/word2vec', 'Output directory.')

FLAGS = flags.FLAGS


def main(_):
    dataset = Word2VecDataset(arch=FLAGS.arch,
                              algm=FLAGS.algm,
                              epochs=FLAGS.epochs,
                              batch_size=FLAGS.batch_size,
                              max_vocab_size=FLAGS.max_vocab_size,
                              min_count=FLAGS.min_count,
                              sample=FLAGS.sample,
                              window_size=FLAGS.window_size,
                              fixed_window_size=FLAGS.fixed_window_size,
                              )
    # create one hot encoded vocabulary dataset
    dataset.build_vocab(FLAGS.filenames)

    word2vec = Word2VecModel(arch=FLAGS.arch,
                             algm=FLAGS.algm,
                             embed_size=FLAGS.embed_size,
                             batch_size=FLAGS.batch_size,
                             negatives=FLAGS.negatives,
                             power=FLAGS.power,
                             alpha=FLAGS.alpha,
                             min_alpha=FLAGS.min_alpha,
                             add_bias=FLAGS.add_bias,
                             random_seed=0)
    # define training operations
    to_be_run_dict = word2vec.train(dataset, FLAGS.filenames)

    print("Corpus size: ", dataset._corpus_size, " words")
    print("Vocabulary size: ", len(dataset.table_words), " words")
    print("Number of sentences: ", dataset.num_sentences)

    with tf.compat.v1.Session() as sess:
        # init variables
        sess.run(dataset.iterator_initializer)
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        average_loss = 0.

        step = 0
        # each steps process 1 batch
        # steps <= (corpus_size / batch) * epochs if arch = cbow
        # steps <= (corpus_size / batch) * window_size * 2 * epochs if arch = cbow
        print("Begin training")
        start_time = time.time()
        while True:
            try:
                # perform one training step
                result_dict = sess.run(to_be_run_dict)
            except tf.errors.OutOfRangeError:
                print("Training completed in ", step, " steps")
                print("Total time: %.3f" % (time.time() - start_time), " seconds")
                break
            average_loss += result_dict['loss'].mean()
            if step % FLAGS.log_per_steps == 0 and step > 0:
                average_loss /= FLAGS.log_per_steps
                print('step:', step, 'average_loss:', average_loss,
                      'learning_rate:', "%.8f" % result_dict['learning_rate'],
                      'progress:', "%.6f" % result_dict['progress'])
                average_loss = 0.
            step += 1

        syn0_final = sess.run(word2vec.syn0)

    print("Corpus size: ", dataset._corpus_size, " words")
    print("Vocabulary size: ", len(dataset.table_words), " words")
    print("Number of sentences: ", dataset.num_sentences)

    # create output folders
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    """np.save(os.path.join(FLAGS.out_dir, 'embed'), syn0_final)
    print("Embedding shape: ", syn0_final.shape)
    with open(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w', encoding="utf-8") as fid:
        for w in dataset.table_words:
            fid.write(w + '\n')
    print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
    print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))"""


if __name__ == '__main__':
    tf.flags.mark_flag_as_required('filenames')
    tf.compat.v1.app.run()
