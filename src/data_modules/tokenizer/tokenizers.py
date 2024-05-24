import os
import sentencepiece as spm
import numpy as np


class Tokenizer(object):
    def __init__(
        self,
        tokenizer_path,
        vocab_size=1024,
        max_token_length=7,
        pad_token="<pad>",
        unk_token="<unk>",
        start_token="<s>",
        end_token="</s>",
        user_defined_symbols=''
    ):
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_token_length = max_token_length
        self.user_defined_symbols = user_defined_symbols
        self.tokenizer = spm.SentencePieceProcessor()

    def load_from_disk(self, tokenizer_path):
        if os.path.exists(tokenizer_path) and os.path.exists(
            tokenizer_path.replace("model", "vocab")
        ):
            with open(tokenizer_path.replace("model", "vocab"), "r") as f:
                lines = f.readlines()

            self.index2token = [
                i.split("\t")[0].strip() for i in lines
            ]
            self.token2index = dict(zip(self.index2token, range(self.vocab_size)))

            assert self.vocab_size == len(self.token2index), (
                "vocab size must be equal to char_to_index,"
                + "you should retrain your model"
                + " with appropriate vocab size."
            )
            if not tokenizer_path.endswith("model"):
                tokenizer_path = f"{self.tokenizer_path}.model"
            self.tokenizer.load(tokenizer_path)
        else:
            raise ValueError(f"Provided {tokenizer_path} not exists!!!")

    def add_token(self, token):
        if isinstance(token, list):
            [self.add_token(t) for t in token]
        else:
            self.token2index[token] = len(self.index2token)
            self.index2token.append(token)
            self.tokenizer.set_vocabulary(self.tokenizer.get_piece_size(), [token])

    def train_tokenizer(
        self, txt_file_path: os.path, model_type: str = "bpe", replace: bool = False
    ):
        training_cmd = (
            f"--input={txt_file_path}"
            + " --max_sentence_length=8000"
            + " --train_extremely_large_corpus=true"
            + f" --vocab_size={self.vocab_size}"
            + f" --model_prefix={self.tokenizer_path}"
            + " --pad_id=0"
            + " --unk_id=1"
            + " --bos_id=2"
            + " --eos_id=3"
            + f" --pad_piece={self.pad_token}"
            + f" --unk_piece={self.unk_token}"
            + f" --bos_piece={self.start_token}"
            + f" --eos_piece={self.end_token}"
            + f" --max_sentencepiece_length={self.max_token_length}"
            + f" --model_type={model_type}"
        )
        if self.user_defined_symbols:
            training_cmd += self.user_defined_symbols

        spm.SentencePieceTrainer.train(training_cmd)

        self.load_from_disk(self.tokenizer_path)

    def __call__(
        self,
        text,
        max_token_length: int = 512,
        padding: bool = False,
        return_attention_mask: bool = False,
    ):
        token_info = {}

        tokenized_text = self.tokenizer.encode_as_ids(text)
        token_info["if_truncated"] = len(tokenized_text) > max_token_length - 2

        tokenized_text = tokenized_text[: max_token_length - 2]
        pad_len = max_token_length - len(tokenized_text) - 2

        tokenized_text = [self.start_token_id] + tokenized_text + [self.end_token_id]
        token_info["n_tokens_without_padding"] = len(tokenized_text)

        if padding:
            tokenized_text += [self.pad_token_id] * pad_len * int(padding)

        if return_attention_mask:
            mask = np.zeros(tokenized_text)
            mask[: token_info["n_tokens_without_padding"]] = 1
            token_info["attention_mask"] = mask

        token_info["tokens"] = tokenized_text

        return token_info

    def tokenize(self, text):
        return self.encode(text)

    def encode(self, text, *args, **kwargs):
        text = self.tokenizer.encode_as_ids(text)
        return text

    @property
    def pad_token_id(self):
        return self.token2index[self.pad_token]

    @property
    def start_token_id(self):
        return self.token2index[self.start_token]

    @property
    def end_token_id(self):
        return self.token2index[self.end_token]

    @property
    def unk_token_id(self):
        return self.token2index[self.unk_token]

    def get_id(self, token):
        return self.token2index.get(token, self.unk_token_id)

    @property
    def cls_token_id(self):
        return self.token2index[self.start_token]


class MultilingualContrastiveTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['user_defined_symbols'] = " --user_defined_symbols=<mask>,<cls>,<en>,</en>"
        self.mask_token = "<mask>"
        self.cls_token = "<cls>"
        self.eng_start_token = "<en>"
        self.eng_end_token = "</en>"

        super().__init__(*args, **kwargs)

    @property
    def en_start_token_id(self):
        return self.token2index[self.eng_start_token]

    @property
    def en_end_token_id(self):
        return self.token2index[self.eng_end_token]

    @property
    def bn_start_token_id(self):
        return self.token2index[self.start_token]

    @property
    def bn_end_token_id(self):
        return self.token2index[self.end_token]

    @property
    def mask_token_id(self):
        return self.token2index[self.mask_token]

    @property
    def cls_token_id(self):
        return self.token2index[self.cls_token]
