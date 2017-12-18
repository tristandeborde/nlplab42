import embeddings

emb = embeddings.EmbeddingsDictionary(100000)
print(emb.w2neighbors("Geek"))
def analogy(self, word1, word2, word3):
    ind1 = self.dictionary[word1]
    ind2 = self.dictionary[word2]
    ind3 = self.dictionary[word3]

    score, index = emb.emb2neighbors(self.embed(word1) + self.embed(word2) - self.embed(word3))
    for ind in index:
        print(self.words[ind])
