# Word2Vec

Tensorflow implementation of Word2Vec, a classic model for learning distributed word representation from large unlabeled dataset.

## Training

1) **Prepare your data**: Your data should be one or more of text files where each line contains a sentence, and words are delimited by space. Make sure all of your data is placed inside the same folder.
2) This implementation allows you to train the model under **skip gram** or **continuous bag-of-words** architectures (`--arch`), and perform training using **negative sampling** or **hierarchical softmax** (`--algm`). To see a full list of parameters, run`python run_training.py --help`.
3) For example you can train your model with the following command:
```
  python run_training.py --filenames=input/wiki1.txt,input/wiki2.txt --out_dir=output/ --window_size=5 --embed_size=300 --arch=skip_gram --algm=negative_sampling --batch_size=256
```
4) The vocabulary words and word embeddings will be saved to `vocab.txt` and `embed.npy` in the folder specified by `--out_dir` in the previous step (can be loaded using `np.load`).

## Word similarity and analogy evaluation

1) The package used to load the evaluation datasets uses `setuptools`. You can install it running:
```
  python setup.py install
```
2) If you have problems during this installation. First you may need to install the dependencies:
```
  pip install -r requirements.txt
```
3) To run the **similarity** evaluation use the following command:
```
  python embedding_eval.py -e embedding/embed.npy -v vocabulary/vocab.txt -sv results/ -s
```
4) You will find your results in the folder specified by `--results`
5) To run the **analogy** evaluation use the following command:
```
  python embedding_eval.py -e embedding/embed.npy -v vocabulary/vocab.txt -sv results/ -a
```
6) You will find your results in the folder specified by `--results`
7) To have show words similarities and analogies graphically, run the following command. You can further customize this file according to your own needs:
```
  python show_similarities.py -e embedding/embed.npy -v vocabulary/vocab.txt
```

## Word sense disambiguation and evaluation

1) Download Stanford’s Contextual Word Similarities (SCWS) at: http://ai.stanford.edu/~ehhuang/ and unzip it
2) Run words **disambiguation** script with the following command:
```
  word_sense_disambiguation.py -e embedding/embed.npy -v vocabulary/vocab.txt -save_path results/ --rating_path SCWS/ratings.txt
```
3) If you want to tune more parameters you can use the command `word_sense_disambiguation.py --help` to see a list of them.
4) You will find your results in the folder specified by `--save_path`

## Results and website

- https://davideliu.com/2020/03/16/word-similarity-and-analogy-with-skip-gram/

## References
- https://github.com/chao-ji/tf-word2vec/blob/master/README.md
- https://github.com/kudkudak/word-embeddings-benchmarks
- https://github.com/logicalfarhad/word-sense-disambiguation/blob/master/word_sense.ipynb
- https://www.researchgate.net/publication/301403994_A_Unified_Model_for_Word_Sense_Representation_and_Disambiguation
