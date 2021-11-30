from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
from tokenizers import Tokenizer
import numpy as np


class HuggingFaceTokenizer(AbsTokenizer):

    tokenizer: Tokenizer
    model_file: str

    def __init__(
        self,
        model_file: str
    ):
        assert check_argument_types()
        self.tokenizer = Tokenizer.from_file(model_file)
        self.model_file = model_file

    def __repr__(self):
        return f'{self.__class__.__name__}(model_file="{self.model_file}, model_type={self.tokenizer.model.__class__.__name__}")'

    def text2tokens(self, line: str) -> List[str]:
        return self.tokenizer.encode(line).tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.decode(self.tokenizer.token_to_id(tokens))

    def get_num_vocabulary_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return [self.tokenizer.id_to_token(i) for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.tokenizer.token_to_id(token) for token in tokens]
