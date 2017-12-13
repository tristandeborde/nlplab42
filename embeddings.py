import logging
import numpy
import torch

class EmbeddingsDictionary:
    def __init__(self, max_words=None, path='data/crawl-300d-2M.vec', normalize=True):
        self.load_embeddings(max_words, path, normalize)

    def load_embeddings(self, max_words, path, normalize):
        '''
        Load the embeddings contained in path. Returns:
        * a list of strings 'words' such that words[i] = token
        * a dictionary such that dictionary[token] = i
        * a float tensor of size n_embeddings * dim, such that tensor[i] = <data>
        '''
        logging.info('Loading embeddings')
        with open(path, 'r') as f:
            first_line = f.readline().split()
            n_embeddings = int(first_line[0])
            if max_words:
                n_embeddings = min(max_words, n_embeddings)
            dim = int(first_line[1])
            tensor = torch.zeros(n_embeddings, dim)
            dictionary = {}
            words = []
            for i, line in enumerate(f):
                if i >= n_embeddings:
                    break
                word, data = line.split(' ', 1)
                data = numpy.fromstring(data, sep=' ')

                assert(len(data) == dim)

                dictionary[word] = i
                tensor[i] = torch.from_numpy(data)
                if normalize:
                    tensor[i] = tensor[i] / tensor[i].norm(2)
                words.append(word)
                if i > 0 and i % 50000 == 0:
                    logging.info('Loading {} / {} embeddings'.format(i, n_embeddings))
            logging.info('Loaded {} embeddings'.format(n_embeddings))
            self.words = words
            self.dictionary = dictionary
            self.emb = tensor

    def embed(self, token):
        return self.emb[self.dictionary[token]]

    def emb2neighbors(self, query_embedding, top_k=20):
        return torch.mm(self.emb, query_embedding.unsqueeze(1)).squeeze(1).topk(k=top_k + 1)

    def w2neighbors(self, query_word, top_k=20):
        query_id = self.dictionary[query_word]
        _scores, neighbor_ids = self.emb2neighbors(self.emb[query_id], top_k)
        neighbor_words = [self.words[i] for i in neighbor_ids if i != query_id]
        return neighbor_words
