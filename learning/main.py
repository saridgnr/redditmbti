import pandas as pd
import numpy as np
import spacy as sp
import torch
from torch import nn
from torch import optim
import random
import torch.nn.functional as F
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
from os import path
from emo_model import EmoModel
from vocab import Vocab

MAX_LENGTH = 500
PERSONALITY = "personality"
REDDIT_SCORE = "reddit_score"
CONTENT_1 = "content_1"
CONTENT_2 = "content_2"

nlp = sp.load('en', disable=["tagger", "parser", "ner"])


def main():
    data = pd.read_csv(r"D:\Users\White\Desktop\Code\colman\NLP\redditMBTI\scrapping\redditMBTIBig.tsv",
                       sep="\t",
                       header=None,
                       names=[PERSONALITY, REDDIT_SCORE, CONTENT_1, CONTENT_2],
                       encoding='UTF-8')
    data.head()

    vocab = Vocab()
    instances = build_dataset(data, vocab)

    random.seed(1)
    shuffle(instances)
    thresh = int(len(instances) * 0.9)
    train = instances[0:thresh]
    test = instances[thresh:]

    print(len(train))

    n_layers = 2
    hidden_size = 800
    embedding_size = 300
    if not path.exists("model.pth"):
        model = EmoModel(vocab.word_count, hidden_size, embedding_size, vocab.tag_count, n_layers).cuda()
    else:
        print("model found! loading...")
        torch.load("model.pth")
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = 1000000
    print_every = 30
    sample_every = 50
    eval_every = 2500
    loss = 0
    for e in range(1, n_epochs + 1):
        pair = random.choice(train)
        input_var = pair[0]
        target_var = pair[1]
        output, iter_loss = train_sentence(input_var, target_var, criterion, model, optimizer)
        loss += iter_loss

        if e % print_every == 0:
            loss = loss / print_every
            print('Epoch %d Current Loss = %.4f' % (e, loss))
            loss = 0
        if e % sample_every == 0:
            print("Target:", target_var.cpu().numpy()[0])
            print("Output:", np.argmax(F.softmax(output).cpu().detach().numpy()))
        if e % eval_every == 0:
            print("Eval:", eval(model, test))

    eval(model, test)


def build_dataset(data, vocab):
    instances = []
    print("building the dataset")
    for index, row in data.iterrows():
        for content in {CONTENT_1, CONTENT_2}:
            text = row.get(content, None)
            if isinstance(text, str):
                proc_text = nlp(text.strip().lower())
                if len(proc_text) > MAX_LENGTH:
                    continue
                indexes = []
                words = [t.text for t in proc_text]
                indexes = vocab.index_document(words)
                if len(indexes) < 4:
                    continue
                personality_id = vocab.get_tag_id(row[PERSONALITY])
                input_var = torch.LongTensor(indexes).cuda()
                target_var = torch.LongTensor([int(personality_id)]).cuda()
                instances.append((input_var, target_var))

    return instances


def train_sentence(sentence, target, criterion, model, optimizer):
    optimizer.zero_grad()
    loss = 0
    output = model(sentence)
    loss = criterion(output.view(1, -1), target)
    loss.backward()
    optimizer.step()
    return output, loss.item()


def eval(model, test):
    print("saving the model")
    torch.save(model, "model.pth")
    print("evaluate the model")
    y_true = []
    y_predicted = []
    for t in test:
        input_var = t[0]
        target_var = t[1]
        output = model(input_var)
        y_predicted.append(np.argmax(F.softmax(output).cpu().detach().numpy()))
        y_true.append(target_var.cpu().numpy()[0])
    return precision_recall_fscore_support(y_true, y_predicted, average='weighted')


if __name__ == "__main__":
    main()
