import os
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import numpy as np


class TwentyNewsgroup:

    def __init__(self, train_path, test_path):
        self.train_id2text, self.train_id2label, self.word2id, self.id2word, self.labels = self.load_data(train_path,
                                                                                                          train=True)
        self.test_id2text, self.test_id2label, _, _, _ = self.load_data(test_path, train=False)

        self.train_id2encoded_text = TwentyNewsgroup.encode_documents(self.train_id2text, self.word2id)
        self.train_id2encoded_label = TwentyNewsgroup.encode_labels(self.train_id2label, self.labels)

        self.test_id2encoded_text = TwentyNewsgroup.encode_documents(self.test_id2text, self.word2id)
        self.test_id2encoded_label = TwentyNewsgroup.encode_labels(self.test_id2label, self.labels)

    def get_training_data(self):
        X = list(); Y = list()
        for id in self.train_id2encoded_text.keys():
            text = self.train_id2encoded_text[id]
            label = self.train_id2encoded_label[id]
            X.append(text)
            Y.append(label)
        return X, Y

    def get_test_data(self):
        X = list(); Y = list()
        for id in self.test_id2encoded_text.keys():
            text = self.test_id2encoded_text[id]
            label = self.test_id2encoded_label[id]
            X.append(text)
            Y.append(label)
        return X, Y

    def get_vocab(self):
        return self.word2id, self.id2word

    def arrange_embedding_matrix(self, matrix, emb_word2id, unk_id, pad_id):
        new_matrix = np.zeros([len(self.word2id), len(matrix[0])])
        for word, wid in self.word2id.items():
            new_word_id = emb_word2id.get(word, emb_word2id.get(unk_id))
            new_matrix[wid] = matrix[new_word_id]
        return new_matrix

    @classmethod
    def encode_documents(cls, id2text, word2id):
        id2encode = dict()
        for id, text in id2text.items():
            id2encode[id] = [word2id.get(token, word2id.get("<UNK>", None)) for token in text]
        return id2encode

    @classmethod
    def encode_labels(cls, id2label, labels):
        id2encode = dict()
        for id, label in id2label.items():
            id2encode[id] = labels[label]
        return id2encode

    def get_tokens(self, path):
        with open(path) as lines:
            all_tokens = [token for line in lines for token in line.split(" ")]
        return all_tokens

    def load_data(self, path, train=False):
        word2id = {"<PAD>": 0, "<UNK>": 1}
        id2text = dict()
        id2label = dict()
        labels = set()
        wid = 2
        for d, dirs, files in os.walk(path):
            for f in files:
                label = d.split("/")[-1]
                labels.add(label)
                tokens = self.get_tokens(os.path.join(d, f))
                for t in tokens:
                    if not t in word2id:
                        word2id[t] = wid
                        wid += 1
                id2text[int(f)] = tokens
                id2label[int(f)] = label
        labels = {idx: l for idx, l in enumerate(sorted(list(labels)))}
        id2word = {id: word for word, id in word2id.items()}
        return id2text, id2label, word2id, id2word, labels

    @classmethod
    def clean_valid_lines(cls, valid_lines, tokeniser):
        email_regex = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        multidots = r"\.\.+"
        multispaces = r"  +"
        email_pattern = re.compile(email_regex)
        multidots_pattern = re.compile(multidots)
        multispaces_pattern = re.compile(multispaces)
        cleaned_lines = list()
        for line in valid_lines:
            line, _ = re.subn(email_pattern, "", line)
            line, _ = re.subn(multidots_pattern, ".", line)
            line, _ = re.subn(multispaces_pattern, " ", line)
            if line.strip().startswith(">"): line = line[1:]
            line = line.strip()
            cleaned_lines.append(line)

        cleaned_tokens = list()
        for line in cleaned_lines:
            if len(line.strip()) == 0:
                continue
            tokens = tokeniser(line)
            cleaned_tokens.append([token.text for token in tokens])
        return cleaned_tokens

    @classmethod
    def clean_file(cls, path, tokeniser):
        line_regex = "Lines: [0-9]+"
        line_pattern = re.compile(line_regex)
        last_match = -1
        num_lines = -1
        try:
            reader = open(path)
            all_lines = [line.strip() for line in reader.readlines()]
            reader.close()
        except UnicodeDecodeError:
            reader = open(path, encoding="8859")
            all_lines = [line.strip() for line in reader.readlines()]
            reader.close()
        for idx, line in enumerate(all_lines):
            if re.match(line_pattern, line):
                num_lines = int(line.split(": ")[1])
                last_match = idx
        valid_lines = all_lines[last_match + 1:last_match + num_lines + 1]
        clean_lines = cls.clean_valid_lines(valid_lines, tokeniser)
        return clean_lines

    @classmethod
    def clea_data(cls, path, out_path):
        nlp = English()
        tokeniser = Tokenizer(nlp.vocab)
        del nlp

        for r, d, f in os.walk(path):
            for file in f:
                clean_lines = cls.clean_file(os.path.join(r, file), tokeniser)
                new_dir = os.path.join(out_path, r.split("/")[-1])
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
                with open(os.path.join(new_dir, file), "w") as writer:
                    writer.write("\n".join([" ".join(tokens) for tokens in clean_lines]))


if __name__ == "__main__":
    TwentyNewsgroup.clea_data("/home/tommaso/Documents/data/20news-bydate/20news-bydate-test",
                              "/home/tommaso/Documents/data/20news-bydate/20news-bydate-test-clean")
