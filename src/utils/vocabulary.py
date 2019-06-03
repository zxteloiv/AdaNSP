# coding: utf-8

import pickle

UNK = '<UNK>'       # 1
START = '<START>'   # 2
EOS = '<EOS>'       # 3


class Vocabulary(object):
    def __init__(self):
        self.tokens = set()
        self.token_to_id = {}
        self.id_to_token = {}

        self.add_token(UNK, preserve=True)
        self.add_token(START, True)
        self.add_token(EOS, True)

    def get_tokens(self):
        return self.tokens

    def get_token(self, tid):
        token = self.id_to_token.get(tid)
        if not token:
            token = UNK
        return token

    def get_id(self, token, preserve=False):
        tid = self.token_to_id.get(self.normalize(token, preserve))
        if not tid:
            tid = self.token_to_id.get(UNK)
        return tid

    def add_token(self, token, preserve=False):
        token = self.normalize(token, preserve)
        if token not in self.tokens:
            self.tokens.add(token)
            self.token_to_id[token] = len(self.tokens)
            self.id_to_token[len(self.tokens)] = token

    def __len__(self):
        return len(self.tokens)

    def __call__(self, word, preserve=False):
        return self.get_id(word, preserve=preserve)

    def save(self, filename):
        with open(filename, 'w') as f:
            to_dump = (self.tokens, self.token_to_id, self.id_to_token)
            pickle.dump(to_dump, f)
            f.close()

    def load(self, filename):
        with open(filename, 'r') as f:
            self.tokens, self.token_to_id, self.id_to_token = pickle.load(f)
            f.close()

    @staticmethod
    def normalize(token, preserve=False):
        if preserve:
            return token
        return token.lower()


if __name__ == "__main__":
    vocab = Vocabulary()
    print("init vocab len:", len(vocab), "contains:", vocab.get_tokens())
    vocab.add_token(UNK, preserve=True)
    print("same vocab len:", len(vocab), "content fixed:", vocab.get_tokens())
    vocab.add_token("stArt")
    print("vocab len +1:", len(vocab), "new token added:", vocab.get_tokens())
    vocab.add_token("stArt")
    print("same vocab len:", len(vocab), "content fixed:", vocab.get_tokens())
    vocab.add_token("dAsh")
    print("vocab len +1:", len(vocab), "new token added:", vocab.get_tokens())
    vocab.save('tmp.pickle')
    print("vocab saved as tmp.pickle")
    newvocab = Vocabulary()
    newvocab.load('tmp.pickle')
    print("new vocab loaded:", newvocab.get_tokens())
    import os
    os.remove('tmp.pickle')
    print("tmp.pickle file removed")
    i1, i2 = vocab.get_id(UNK), newvocab.get_id(UNK)
    print("UNK ids in both are:", i1, i2)
    i1, i2 = vocab.get_id("hello"), newvocab.get_id("hello")
    print("the unseen word 'hello' has the same ID as UNK:", i1, i2)
    i1, i2 = vocab("stArt"), newvocab("stArt")
    print("word 'stArt' id using the call interface:", i1, i2)
    t1, t2 = vocab.get_token(i1), newvocab.get_token(i2)
    print("token get using stArt ID retrieved above:", t1, t2)
    t1, t2 = vocab.get_token(100), newvocab.get_token(200)
    print("token get using unseen ID:", t1, t2)

