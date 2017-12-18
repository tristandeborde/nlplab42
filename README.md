# AI Masterclass lab: NLP

## Introduction

This lab's goal is to familiarize you with natural language processing by building a sentiment analysis model using pytorch.

Pre-requisites (you should probably follow [42-AI](https://github.com/42-AI/ai-for-42-students/blob/master/HOW_TOs.md)'s instructions here)
* Python 3.5+
* Pytorch 0.2+
* Download and unzip a pre-trained embedding file, preferably the wikipedia EN fasttext embeddings [wiki-news-300d-1M.vec.zip](https://fasttext.cc/docs/en/english-vectors.html). Note: this file will be made available on a server internal to 42.

## Task definition

The dataset is an extract from Stanford Sentiment Treebank. It consists in portions of English sentences about a movie along with a sentiment label. The sentiment label is 0 for sentences that emit a negative opinion about the movie, and 1 for positive reviews. E.g., *a gorgeous , witty , seductive movie* will be labeled 1 and *forced , familiar and thoroughly condescending* is labeled 0.


## Instructions

1. Fork this repo and add your fork's URL to [this document](https://docs.google.com/spreadsheets/d/11McyuMX0TxY63nGHO87yDpIJpxq1sHAHZM8Et0peTws/edit?usp=sharing)
2. Put the embedding file described in prerequisites into a folder called `data` inside your repo root.
3. Create a python script that loads embeddings using `embedding.py` and runs two tasks:
    1. Nearest-neighbor search: print the 10 nearest neighbors of the word 'geek'
    2. Analogy: retrieve embeddings closest to a combination of embeddings that corresponds to an analogy, e.g. `'Tokyo' + 'Spain' - 'Japan' = ?`. Caveats: you will need to remove the embeddings corresponding to any of the query words (in this case, `Tokyo`, `Spain` and `Japan`) explicitly from the tokens you retrieve.
    3. Find the best analogy you can and add it to the [analogy tab of the google sheets](https://docs.google.com/spreadsheets/d/11McyuMX0TxY63nGHO87yDpIJpxq1sHAHZM8Et0peTws/edit?usp=sharing#gid=61225320). Be creative!
4. Run the baseline model using `train.py`. After the data is loaded, training will start and the loss and accuracy on train and dev sets will be reported. Remember: the goal is to have the **highest test accuracy possible**. At each epoch, the script will write a `model.pth` file in your repo that you will need to push along the rest of your code for your work to be evaluated.
5. Try to improve the model. Several options appear here:
    1. The baseline uses only the first word of the sentence to perform classification. Edit `model.py` so that it performs the averaging of all embeddings in the sentence.
    2. (requires 1. above) Use an importance weighting scheme for each word in the sentence, e.g. scale each embedding in the inverse proportion of the frequency of the token in the documents. This will reduce the importance of very common words such as "the" or "and". After that, you could go for more advanced weighting schemes, e.g. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
    3. Change the optimizer parameters: you could use different learning rates or different optimizers.
    4. Change the model altogether: you could go for a recurrent neural network such as an [LSTM](http://pytorch.org/docs/master/nn.html#lstm)

Good luck!
