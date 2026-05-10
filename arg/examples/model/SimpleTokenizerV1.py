import re
import urllib.request
from pkgutil import resolve_name


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def download_file():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "../chapter_02/the-verdict.txt"
    with urllib.request.urlopen(url) as response:
        with open("../chapter_02/doc/the-verdict.txt", "wb") as f:
            f.write(response.read())


def load_vocab_from_web():
    with open("../chapter_02/doc/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_txt = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_txt)  # split the text
    preprocessed = [item.strip() for item in preprocessed if item.strip()]  # remove the space
    all_words = sorted(set(preprocessed))
    # print("start to create the vocab")
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i > 50:
    #         break


# download_file()
# load_vocab_from_web()
vocab = load_vocab_from_web();
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
