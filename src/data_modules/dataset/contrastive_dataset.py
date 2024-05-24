from torch.utils.data import Dataset
import tqdm
import torch
import random


class SimBERTDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        padding=False,
        encoding="utf-8",
        corpus_lines=0,
        on_memory=True,
        mode='train'
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.padding = padding

        with open(corpus_path, "r", encoding=encoding) as f:
            if not (self.corpus_lines) and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [
                    line.strip().split(" <SEP> ")
                    for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)
                ]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(
                random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)
            ):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        try:
            t1, t2 = self.get_corpus_line(item)
            t1_random, t1_label = self.random_word(t1)
            t2_random, t2_label = self.random_word(t2)

            input_1 = self.preprocess_tokenized_text(t1_random, t1_label)
            input_2 = self.preprocess_tokenized_text(t2_random, t2_label, lang='en')
        except Exception as e:
            print(e)
            return self[(item+1) % len(self)]

        return [input_1, input_2]

    def preprocess_tokenized_text(self, tokenized_text, label, do_padding=False, lang='bn'):
        start_token_id = (
            self.tokenizer.bn_start_token_id
            if lang == 'bn' else self.tokenizer.en_start_token_id
        )
        end_token_id = (
            self.tokenizer.bn_end_token_id
            if lang == 'bn' else self.tokenizer.en_end_token_id
        )
        tokenized_text = (
            [start_token_id]
            + tokenized_text[: self.seq_len - 2]
            + [end_token_id]
        )
        label = (
            [self.tokenizer.pad_token_id]
            + label[: self.seq_len - 2]
            + [self.tokenizer.pad_token_id]
        )

        if do_padding:
            padding = [
                self.tokenizer.pad_token_id
                for _ in range(self.seq_len - len(tokenized_text))
            ]
            tokenized_text.extend(padding)

        processed_input = {
            "bert_input": torch.tensor(tokenized_text),
            "bert_label": torch.tensor(label),
            "segment_label": torch.tensor(1.0 if lang == 'en' else 0.0),
        }
        return processed_input

    def random_word(self, sentence):
        words = sentence.split()
        output_label = []
        input_token_ids = []

        for i, token in enumerate(words):
            prob = random.random()
            tokens_ids = self.tokenizer.tokenize(token)
            selected_token_ids = tokens_ids

            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    selected_token_ids = [self.tokenizer.mask_token_id] * len(
                        tokens_ids
                    )

                # 10% randomly change token to random token
                elif prob < 0.9:
                    selected_token_ids = random.choices(
                        list(range(self.tokenizer.vocab_size)), k=len(tokens_ids)
                    )
                # 10% randomly change token to current token
                else:
                    selected_token_ids = tokens_ids

                input_token_ids.extend(selected_token_ids)
                output_label.extend(tokens_ids)

            else:
                input_token_ids.extend(tokens_ids)
                output_label.extend([0] * len(tokens_ids))

        return input_token_ids, output_label

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2
