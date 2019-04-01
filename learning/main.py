from datetime import datetime
from itertools import chain

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
MODEL_FILE_NAME = "model_wdo.pth"
PERSONALITY = "personality"
REDDIT_SCORE = "reddit_score"
CONTENT_1 = "content_1"
CONTENT_2 = "content_2"

nlp = sp.load('en', disable=["tagger", "parser", "ner"])
curr_run_file = open(str(datetime.now().timestamp()), mode="w")


def main():
    data = pd.read_csv(r"D:\Users\White\Desktop\Code\colman\NLP\redditMBTI\scrapping\redditMBTIBig.tsv",
                       sep="\t",
                       header=None,
                       names=[PERSONALITY, REDDIT_SCORE, CONTENT_1, CONTENT_2],
                       encoding='UTF-8')
    data.head()
    vocab = Vocab()
    instances = build_dataset(data, vocab)
    assert set(vocab.id2tag.values()) - \
           {'ESFP', 'ENTJ', 'INTJ', 'INFP', 'ISTJ', 'ESTP', 'INTP', 'ISFJ',
            'ISFP', 'ENFJ', 'ESTJ', 'ENFP', 'INFJ', 'ENTP', 'ESFJ', 'ISTP'} != {}, "Invalid types in train data"
    random.seed(1)
    shuffle(instances)
    thresh = int(len(instances) * 0.95)
    train = instances[0:thresh]
    test = instances[thresh:]

    n_epochs = 50
    epoch_size = len(train)
    n_layers = 2
    hidden_size = 800
    embedding_size = 300
    print_progress_every = 30
    if not path.exists(MODEL_FILE_NAME):
        model = EmoModel(vocab.word_count, hidden_size, embedding_size, vocab.tag_count, n_layers).cuda()
    else:
        print("model found! loading...")
        model = torch.load(MODEL_FILE_NAME)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Train data length {0}".format(len(train)))
    for epoch in range(0, n_epochs):
        loss = 0
        print("Starting epoch: {0}".format(epoch))
        for i, pair in enumerate(random.sample(train, epoch_size)):
            if i % print_progress_every == 0:
                print("Training {0}/{1}".format(i, epoch_size))
            input_var, target_var = pair
            output, iter_loss = train_sentence(input_var, target_var, criterion, model, optimizer)
            loss += iter_loss

        print("Loss: {0}".format(loss / epoch_size))
        evaluation = eval(model, test, vocab)

        # Evaluates for each of the personality types
        curr_run_file.write("{0},{1}|".format(epoch, (loss / epoch_size)))
        for k, v in evaluation.items():
            curr_run_file.write(k+":")
            curr_run_file.write(",".join(map(str, v)))
            curr_run_file.write("|")
        curr_run_file.write("\n")
        curr_run_file.flush()
    curr_run_file.close()


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
    output = model(sentence)
    loss = criterion(output.view(1, -1), target)
    loss.backward()
    optimizer.step()

    return output, loss.item()


def eval(model, test, vocab):
    print("saving the model")
    torch.save(model, MODEL_FILE_NAME)
    print("evaluate the model")
    y_true = []
    y_predicted = []

    letters_predicted = [[], [], [], []]
    letters_true = [[], [], [], []]

    for t in test:
        input_var = t[0]
        target_var = t[1]
        output = model(input_var)
        y_predicted.append(np.argmax(F.softmax(output).cpu().detach().numpy()))
        y_true.append(target_var.cpu().numpy()[0])
        for i in range(0, 4):
            p, t = extract_letter_predicted_true(y_predicted[-1], y_true[-1], vocab, i)
            letters_predicted[i].append(p)
            letters_true[i].append(t)

    return {
        "all": precision_recall_fscore_support(y_true, y_predicted, average='weighted'),
        "EI": precision_recall_fscore_support(letters_true[0], letters_predicted[0], average='weighted'),
        "SN": precision_recall_fscore_support(letters_true[1], letters_predicted[1], average='weighted'),
        "TF": precision_recall_fscore_support(letters_true[2], letters_predicted[2], average='weighted'),
        "JP": precision_recall_fscore_support(letters_true[3], letters_predicted[3], average='weighted'),
    }


def extract_letter_predicted_true(predicted, true, vocab, letter_index):
    return vocab.id2tag[predicted][letter_index],  vocab.id2tag[true][letter_index]


if __name__ == "__main__":
    main()
