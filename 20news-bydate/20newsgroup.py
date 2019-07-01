import os
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class TwentyNewsgroup:
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
