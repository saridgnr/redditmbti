class Vocab(object):
    def __init__(self):
        self.word_count = 0
        self.ind2word = {}
        self.word2ind = {}

        self.tag_count = 0
        self.tag2id = {}
        self.id2tag = {}

    def index_document(self, document):
        indexes = []
        for token in document:
            indexes.append(self.index_token(token))
        return indexes

    def index_token(self, token):
        if token not in self.word2ind:
            self.word2ind[token] = self.word_count
            self.ind2word[self.word_count] = token
            self.word_count += 1
        return self.word2ind[token]

    def get_tag_id(self, tag):
        if tag not in self.tag2id:
            self.tag2id[tag] = self.tag_count
            self.id2tag[self.tag_count] = tag
            self.tag_count += 1
        return self.tag2id[tag]
