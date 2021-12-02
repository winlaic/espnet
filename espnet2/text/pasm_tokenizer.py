from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
from tokenizers import Tokenizer
import numpy as np

import re


class PASMTokenizer(AbsTokenizer):

    subwords_file: str

    def __init__(
        self,
        subwords_file: str,
        space_symbol: str = '_',
        chinese_token_list: str = None,
    ):
        # assert check_argument_types()
        self.subwords_file = subwords_file
        self.space_symbol = space_symbol
        with open(self.subwords_file, 'r', encoding='utf8') as f:
          self.subwords = [item for item in f.read().split('\n') if item]

        self.subwords = {k: r'(((?<= )|^)' + ' '.join(k) + r'((?= )|$))' for k in self.subwords}

        self.tokens = ["'"]
        self.tokens.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        self.tokens.extend(list(self.subwords.keys()))
        self.tokens.extend([item + self.space_symbol for item in self.tokens])
        self.tokens.sort()

        if chinese_token_list is not None:
          import json
          with open(chinese_token_list, encoding='utf8') as f:
            chinese_token_list = json.load(f)
          self.tokens.extend(list(item + self.space_symbol for item in chinese_token_list.keys()))
          self.has_chinese = True
        else:
          self.has_chinese = False

        self.tokens.insert(0, '<blank>')
        self.tokens.insert(1, '<unk>' + self.space_symbol)
        self.tokens.append('<sos/eos>')

        self.tokens = {k: v for v, k in enumerate(self.tokens)}


    def __repr__(self):
        return f'{self.__class__.__name__}(subwords_file="{self.subwords_file}")'

    def text2tokens(self, line: str) -> List[str]:
        line = line.strip()
        line = line.replace(" ", self.space_symbol)
        line = ' '.join(line)
        for w, rew in self.subwords.items():
          line = re.sub(rew, w, line)
        line = line.replace(f" {self.space_symbol} ", f"{self.space_symbol} ")
        line = line.strip()
        if len(line) > 0:
          line += self.space_symbol

        return line.split(' ')

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return ''.join(tokens).replace(self.space_symbol, ' ').strip()

    
    def get_num_vocabulary_size(self) -> int:
        return len(self.tokens)


    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return [list(self.tokens)[i] for i in integers]


    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.tokens.get(t, 1) for t in tokens]


