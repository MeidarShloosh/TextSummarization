
SOS_TOKEN = 0
EOS_TOKEN = 1
UNKNOWN = 2


class Vocab:
    def __init__(self):
        self.word2index = {}
        self.index2word = {UNKNOWN: '__unk__'}
        self.n_words = 3

    def index_story(self, story, write=True):
        indexes = []
        for w in story.split(" "):
            indexes.append(self.index_word(w, write))
        return indexes

    def index_word(self, word, write=True):
        if word not in self.word2index:
            if write:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words = self.n_words + 1
            else:
                return UNKNOWN
        return self.word2index[word]