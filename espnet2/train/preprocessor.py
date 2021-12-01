from abc import ABC
from abc import abstractmethod
from codecs import encode
from dataclasses import replace
from pathlib import Path
from typing import Any, Collection
from typing import Dict
from typing import List
from typing import Iterable
from typing import Union
import kaldiio

import numpy as np
from numpy.lib.arraysetops import isin
import sentencepiece
import scipy.signal
import soundfile
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter
from functools import partial
import re
import logging
from pathlib import Path
from espnet2.text.huggingface_tokenizer import HuggingFaceTokenizer

# About UPOS
# https://universaldependencies.org/u/pos/

POS_SIMPLIFY_MAPPER = {
    'ADJ': 'ADJ',
    'NOUN': 'NOUN',
    'PRON': 'NOUN',
    'PROPN': 'NOUN',
    'VERB': 'VERB',
}

from collections import defaultdict

def sorted_dict(d, reverse=True):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))

def pos_determinator(all_cased_pos):
    poll = defaultdict(float)
    for k, v in all_cased_pos.items():
        poll[v] += 1.0

    ret = list(sorted_dict(poll, reverse=True).keys())[0]
    return ret



def kill(key, self):
    align_info = self.splicing_data.source_align_info[key]
    source_text_tokens = ' '.join([item['text'] for item in align_info])
    source_text_parsed = self.splicing_data.nlp_chn(source_text_tokens)
    source_pos = [item.to_dict()[0]['upos'] for item in source_text_parsed.iter_tokens()]

    # Change english slices use english model.
    source_pos_eng_spans = Xspans(source_pos)
    for eng_spans_slice in source_pos_eng_spans:
        eng_span = source_text_tokens.split(' ')[eng_spans_slice]
        eng_span = ' '.join(eng_span)
        eng_span_parsed = self.splicing_data.nlp_eng(eng_span)
        source_pos[eng_spans_slice] = [item.to_dict()[0]['upos'] for item in eng_span_parsed.iter_tokens()]

    return source_pos



def save_spliced(path: Path, uttid, speech, text, insert_words, insert_word_positions):
    path.mkdir(exist_ok=True, parents=True)
    insert_word_positions = list(map(str, insert_word_positions))
    file_prefix = uttid + '_' + '_'.join('-'.join(item) for item in zip(insert_words, insert_word_positions))
    speech = speech[:, 0:80]
    np.save((path / (file_prefix + '.npy')), speech)
    with (path / (file_prefix + '.lab')).open('w', encoding='utf8') as f:
        f.write(text + '\n')


# logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
#     )

class PsudoRandomChoicer:
    def __init__(self, x):
        self.x = x
        self.indices = list(np.random.choice(len(self.x), len(self.x), replace=False))

    def reset(self):
        remained = [self.x[i] for i in self.indices]
        logging.info(f'Choicer epoch finished. Remained: {remained}')
        self.indices = list(np.random.choice(len(self.x), len(self.x), replace=False))
    
    def choice(self, n):
        ret = []
        if len(self.indices) < n:
            self.reset()
        retincides = []
        for _ in range(n):
            idx = self.indices.pop()
            ret.append(self.x[idx])
            retincides.append(idx)

        return ret, retincides


# class GroupedPsudoRandomChoicer:
#     def __init__(self, x, x_pos=None):
#         self.x = x
#         if x_pos is None:
#             self.x_pos = ['NONE' for _ in x]
#         self.x_librarys = {}
#         self.reset()

#     @property
#     def pos_classes(self):
#         return set(self.x_pos)

#     def reset(self, pos=None):
#         if pos is None:
#             for pos in self.pos_classes:
#                 self.reset(pos)
#         self.x_librarys[pos] = [self.x[i] for i in range(len(self.x)) if self.x_pos[i] == pos]


#     def choice(self, n, source=None, source_pos=None, source_idx=None):
#         pass


class LazyInserter:
    def __init__(self, x) -> None:
        self.x = x
        self.spaces = [[] for _ in range(len(self.x) + 1)]


    def insert(self, index: int, item: Any):
        self.spaces[index].append(item)

    def __getitem__(self, index):
        return self.x[index]

    def apply(self):
        offset = 0
        for idx, space in enumerate(self.spaces):
            for item in space:
                self.x.insert(idx + offset, item)
                offset += 1
        self.spaces = [[] for _ in range(len(self.x) + 1)]

    def __repr__(self):
        return f'{__class__.__name__}({self.x}, pending={self.spaces})'

    
def spm_encode_english(text, model: sentencepiece.SentencePieceProcessor):
    if re.match(r"[A-Za-z0-9']", text):
        text = model.EncodeAsPieces(text)
        text = ' '.join(text)
    else:
        return text
    return text


TIME_AXIS = 0

ENGLISH_WORD_PATTERN = re.compile(r"(▁[A-Za-z0-9']*( [A-Za-z0-9']+)*)")
ENGLISH_SPAN_PATTERN = re.compile(r"([A-Za-z0-9']+( [A-Za-z0-9']+)*)")
CHINESE_SPACE_PATTERN = re.compile(r"(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])")
CHINESE_BORDER_PATTERN = re.compile(r"(?<=[\u4e00-\u9fa5])(?=[\u4e00-\u9fa5])")

class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w ** 2).mean(axis=-1)
    # mean_power: (C,)
    mean_power = power.mean(axis=-1)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=np.bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

def Xspans(pos: List[str]):
    ret = []
    current_start = None
    for i, p in enumerate(pos):
        if p == 'X':
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                ret.append(slice(current_start, i))
                current_start = None
    else:
        if current_start is not None:
            ret.append(slice(current_start, None))

    return ret




class SpeechChunk:

    chunks: List[np.ndarray]
    words: List[str]

    def time_to_frame(self, time):
        ret = np.round((time - self.frame_length / 2) / self.frame_shift)
        if isinstance(ret, np.ndarray):
            return ret.astype(int)
        elif isinstance(ret, float):
            return int(ret)
        else:
            raise NotImplementedError
    
    def frame_to_time(self, frame_pos):
        return frame_pos * self.frame_shift + self.frame_length / 2

    def __init__(self, speech: np.ndarray, align: dict, frame_length=0.025, frame_shift=0.01) -> None:
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        
        split_points = []
        split_words = []

        current_frame = 0
        for part in align:
            start_frame_pos = self.time_to_frame(part['start'])
            end_frame_pos = self.time_to_frame(part['end'])
            if start_frame_pos > current_frame:
                split_points.append(start_frame_pos)
                split_words.append('')
            split_points.append(end_frame_pos)
            split_words.append(part['text'])
            current_frame = end_frame_pos
        else:
            if current_frame < speech.shape[0]:
                split_words.append('')
            elif current_frame == speech.shape[0]:
                split_points = split_points[:-1]
            else:
                raise ValueError('Alignment unmatchd!')

        self.chunks = np.split(speech, split_points, axis=0)
        self.words = split_words

        assert len(self.chunks) == len(self.words)

    def range(self, index):
        start = sum(item.shape[0] for item in self.chunks[:index])
        return (start, start + self.chunks[index].shape[0])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        return (self.chunks[index], self.words[index])

    def __setitem__(self, index, value):
        speech, word = value
        assert speech.shape[1] == self.chunks[0].shape[1]
        self.chunks[index] = speech
        self.words[index] = word

    def insert(self, index, value):
        speech, word = value
        assert speech.shape[1] == self.chunks[0].shape[1]
        self.chunks.insert(index, speech)
        self.words.insert(index, word)


    @property
    def speech(self):
        return np.concatenate(self.chunks, axis=0)

    @property
    def text(self):
        return ' '.join(item for item in self.words if len(item) != 0)

    @property
    def non_silence_chunk_indices(self):
        return [i for i in range(len(self)) if len(self.words[i]) > 0]





class CommonPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        huggingface_tokenizer_file: str = None,
        rir_scp: str = None,
        rir_apply_prob: float = 1.0,
        noise_scp: str = None,
        noise_apply_prob: float = 1.0,
        noise_db_range: str = "3_10",
        splicing_config: str = None,
        speech_volume_normalize: float = None,
        speech_name: str = "speech",
        text_name: str = "text",
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name
        self.speech_volume_normalize = speech_volume_normalize
        self.rir_apply_prob = rir_apply_prob
        self.noise_apply_prob = noise_apply_prob

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            if token_type != 'huggingface':

                self.tokenizer = build_tokenizer(
                    token_type=token_type,
                    bpemodel=bpemodel,
                    delimiter=delimiter,
                    space_symbol=space_symbol,
                    non_linguistic_symbols=non_linguistic_symbols,
                    g2p_type=g2p_type,
                )
                self.token_id_converter = TokenIDConverter(
                    token_list=token_list,
                    unk_symbol=unk_symbol,
                )
            else:
                self.tokenizer = build_tokenizer(
                    token_type=token_type,
                    huggingface_tokenizer_file=huggingface_tokenizer_file
                )
                self.token_id_converter = self.tokenizer
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

        if train and rir_scp is not None:
            self.rirs = []
            with open(rir_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.rirs.append(sps[0])
                    else:
                        self.rirs.append(sps[1])
        else:
            self.rirs = None

        if train and noise_scp is not None:
            self.noises = []
            with open(noise_scp, "r", encoding="utf-8") as f:
                for line in f:
                    sps = line.strip().split(None, 1)
                    if len(sps) == 1:
                        self.noises.append(sps[0])
                    else:
                        self.noises.append(sps[1])
            sps = noise_db_range.split("_")
            if len(sps) == 1:
                self.noise_db_low, self.noise_db_high = float(sps[0])
            elif len(sps) == 2:
                self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
            else:
                raise ValueError(
                    "Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]"
                )
        else:
            self.noises = None

        if train and splicing_config is not None:
            import yaml, json, re, copy
            import sentencepiece, jieba
            from espnet2.fileio.read_text import read_2column_text
            from omegaconf import OmegaConf
            from types import SimpleNamespace
            self.splicing_config = OmegaConf.load(splicing_config)
            self.splicing_data = SimpleNamespace()
            # with open(splicing_config, 'r', encoding='utf8') as f:
            #     self.splicing_config = yaml.safe_load(f)

            with Path(self.splicing_config.align_info_json).open('r', encoding='utf8') as f:
                self.splicing_data.source_align_info = json.load(f)

            
            with open(self.splicing_config.audio_dictionary_index, 'r', encoding='utf8') as f:
                self.splicing_data.audio_dictionary_index = json.load(f)

            audio_dictionary_type = self.splicing_config.audio_dictionary_type
            if audio_dictionary_type == 'kaldi_ark':
                import kaldiio
                audio_dictionary_book = read_2column_text(self.splicing_config['audio_dictionary_book'])
                from os.path import exists
                assert all(exists(item.split(':')[0]) for item in audio_dictionary_book.values())
                self.splicing_data.audio_dictionary_book = audio_dictionary_book


            elif audio_dictionary_type == 'wav':
                raise NotImplementedError
            
            else:
                raise NotImplementedError



            if hasattr(self.splicing_config, 'bpe_model') and (self.splicing_config.bpe_model is not None):
                self.splicing_data.bpe_model = sentencepiece.SentencePieceProcessor()
                self.splicing_data.bpe_model.Load(self.splicing_config.bpe_model)
                self.splicing_data.spm_encode_english = partial(spm_encode_english, model=self.splicing_data.bpe_model)
            
            if 'insert' in self.splicing_config.configs:

                insert_configs = self.splicing_config.configs.insert
                
                with Path(insert_configs.words_to_insert).open('r', encoding='utf8') as f:
                    self.splicing_data.words_to_insert = json.load(f)

            if 'replace' in self.splicing_config.configs:
                replace_configs = self.splicing_config.configs.replace

                with Path(replace_configs['words_to_replace']).open('r', encoding='utf8') as f:
                    self.splicing_data.words_to_replace = json.load(f)

            if self.splicing_config.stanza:
                stanza_train_text_pos = Path(self.splicing_config.stanza_train_text_pos)
                with stanza_train_text_pos.open('rb') as f:
                    self.splicing_data.stanza_train_text_pos = json.load(f)


                stanza_oov_pos = Path(self.splicing_config.stanza_oov_pos)
                with stanza_oov_pos.open('rb') as f:
                    self.splicing_data.stanza_oov_pos = json.load(f)


        else:
            self.splicing_config = None
            self.splicing_data = None



    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()


        if self.splicing_config is not None:
            text: str = data[self.text_name]
            speech: np.ndarray = data[self.speech_name] # (time, channel)

            align_info = self.splicing_data.source_align_info.get(uid, None)

            replaced_positions = []

            if align_info is None:
                logging.warning(f"Utter {uid} has no align info, skipped insertion.")
            else:
                if hasattr(self.splicing_data, 'bpe_model') and (self.splicing_data.bpe_model is not None):
                    squeezed_text = ENGLISH_WORD_PATTERN.sub(lambda x: self.splicing_data.bpe_model.DecodePieces(x[0].split(' ')), text)
                    squeezed_text = CHINESE_SPACE_PATTERN.sub('', squeezed_text)
                elif isinstance(self.tokenizer, HuggingFaceTokenizer):
                    squeezed_text = CHINESE_SPACE_PATTERN.sub('', text)
                else:
                    raise NotImplementedError

                align_info_text = ' '.join(item['text'].upper() for item in align_info)
                align_info_text = CHINESE_SPACE_PATTERN.sub('', align_info_text)
                assert squeezed_text == align_info_text, (squeezed_text, align_info_text)

                # From here we can confirm that align info is correspoding with tokenized text.
                # text variable is not useful here.

                speech_chunk = SpeechChunk(speech, align_info)


                if 'replace' in self.splicing_config['configs']:
                    replace_configs = self.splicing_config['configs']['replace']
                    
                    # if np.random.random() < replace_configs['prob']:
                    if True:
                        words_to_replace = self.splicing_data.words_to_replace
                        if replace_configs['position'] == 'all':
                            positions_to_replace = speech_chunk.non_silence_chunk_indices
                        else:
                            raise NotImplementedError

                        if replace_configs.english_only:
                            positions_to_replace = [item for item in positions_to_replace if (len(ENGLISH_SPAN_PATTERN.findall(speech_chunk.words[item])) > 0)]

                        if len(positions_to_replace) < replace_configs['num_in_selected_positions']:
                            logging.warning(f"Utter {uid} has less than {replace_configs['num_in_selected_positions']}, ignored.")
                        else:

                            if not hasattr(self, 'replace_word_choicer'):
                                self.replace_word_choicer = PsudoRandomChoicer(words_to_replace)

                            replace_words, replace_word_indices = self.replace_word_choicer.choice(replace_configs['num_in_selected_positions'])
                            # replace_words = [item.lower() for item in replace_words]

                            if self.splicing_config.configs.replace.use_stanza:

                                source_pos = self.splicing_data.stanza_train_text_pos[uid]

                                # replace_words_pos = [list(self.splicing_data.nlp_eng(item).iter_tokens())[0].to_dict()[0]['upos'] if len(ENGLISH_SPAN_PATTERN.findall(item)) != 0 else list(self.splicing_data.nlp_chn(item).iter_tokens())[0].to_dict()[0]['upos'] for item in replace_words ]
                                
                                replace_words_pos = [self.splicing_data.stanza_oov_pos[item] for item in replace_words]
                                
                                if self.splicing_config.configs.replace.stanza_simplify_pos:
                                    source_pos = [POS_SIMPLIFY_MAPPER.get(item, 'OTHER') for item in source_pos]
                                    replace_words_pos = [{k: POS_SIMPLIFY_MAPPER.get(v, 'OTHER') for k, v in item.items()} for item in replace_words_pos]

                                replace_words_pos = [pos_determinator(item) for item in replace_words_pos]

                                positions_to_replace_selected = []
                                vacent_terms = list(zip(positions_to_replace, source_pos))

                                # Get valid 'positions_to_replace_selected'
                                for replace_word, replace_word_pos in zip(replace_words, replace_words_pos):
                                    if len(vacent_terms) == 0:
                                        break
                                    valid_nominators_indices = [i for i, item in enumerate(vacent_terms) if item[1] == replace_word_pos]
                                    if len(valid_nominators_indices) == 0:
                                        if self.splicing_config.configs.replace.force_replace_if_stanza_failed:
                                            selected_vacent_idx = int(np.random.choice(len(vacent_terms), 1))
                                            force_replaced = speech_chunk.words[vacent_terms[selected_vacent_idx][0]]
                                            logging.debug(f'Word "{replace_word}" has no matched POS. Force replaced with "{force_replaced}".')
                                        else:
                                            logging.debug(f'Word "{replace_word}" has no matched POS. Skipped.')
                                            continue
                                    else:
                                        selected_vacent_idx = valid_nominators_indices[int(np.random.choice(len(valid_nominators_indices), 1))]
                                    
                                    positions_to_replace_selected.append(vacent_terms[selected_vacent_idx][0])
                                    del vacent_terms[selected_vacent_idx]

                                pass
                            else:
                                positions_to_replace_selected = [positions_to_replace[i] for i in np.random.choice(len(positions_to_replace), replace_configs['num_in_selected_positions'], replace=False)]
                            

                            audio_dictionary_index = self.splicing_data.audio_dictionary_index
                            audio_dictionary_book = self.splicing_data.audio_dictionary_book


                            replace_audio_pos = [audio_dictionary_index[item.lower()][np.random.choice(len(audio_dictionary_index[item.lower()]), 1)[0]] for item in replace_words]
                            replace_audios = []
                            for audio_pos in replace_audio_pos:
                                utt_feature_path = audio_dictionary_book[audio_pos['uttid']]
                                utt_feature = kaldiio.load_mat(utt_feature_path)
                                start_frame = int(np.round((audio_pos['start'] - self.splicing_config['frame_length'] / 2 ) / self.splicing_config['frame_shift']))
                                end_frame = int(np.round((audio_pos['end'] - self.splicing_config['frame_length'] / 2 ) / self.splicing_config['frame_shift']))
                                replace_audios.append(utt_feature[start_frame:end_frame, :])

                            
                            for index, audio, word in zip(positions_to_replace_selected, replace_audios, replace_words):
                                speech_chunk[index] = (audio, word)
                                replaced_positions.append(speech_chunk.range(index))

                            speech = speech_chunk.speech
                            words = [item.upper() for item in speech_chunk.words if len(item) > 0]

                            if not isinstance(self.tokenizer, HuggingFaceTokenizer):
                                words = list(map(self.splicing_data.spm_encode_english, words))
                                text = ' '.join(words)
                                text = CHINESE_SPACE_PATTERN.sub('', text)
                                text = CHINESE_BORDER_PATTERN.sub(' ', text)
                            else:
                                text = ' '.join(words)
                                text = CHINESE_BORDER_PATTERN.sub(' ', text)

                            data[self.speech_name] = speech
                            data[self.text_name] = text


            



                if 'insert' in self.splicing_config['configs'] and self.splicing_config.configs.insert.prob > 0:
                    raise NotImplementedError
        
            data['replaced_positions'] = np.array(replaced_positions) if len(replaced_positions) > 0 else np.zeros((0, 2))


        if self.speech_name in data:
            if self.train and self.rirs is not None and self.noises is not None:
                speech = data[self.speech_name]
                nsamples = len(speech)

                # speech: (Nmic, Time)
                if speech.ndim == 1:
                    speech = speech[None, :]
                else:
                    speech = speech.T
                # Calc power on non shlence region
                power = (speech[detect_non_silence(speech)] ** 2).mean()

                # 1. Convolve RIR
                if self.rirs is not None and self.rir_apply_prob >= np.random.random():
                    rir_path = np.random.choice(self.rirs)
                    if rir_path is not None:
                        rir, _ = soundfile.read(
                            rir_path, dtype=np.float64, always_2d=True
                        )

                        # rir: (Nmic, Time)
                        rir = rir.T

                        # speech: (Nmic, Time)
                        # Note that this operation doesn't change the signal length
                        speech = scipy.signal.convolve(speech, rir, mode="full")[
                            :, : speech.shape[1]
                        ]
                        # Reverse mean power to the original power
                        power2 = (speech[detect_non_silence(speech)] ** 2).mean()
                        speech = np.sqrt(power / max(power2, 1e-10)) * speech

                # 2. Add Noise
                if (
                    self.noises is not None
                    and self.noise_apply_prob >= np.random.random()
                ):
                    noise_path = np.random.choice(self.noises)
                    if noise_path is not None:
                        noise_db = np.random.uniform(
                            self.noise_db_low, self.noise_db_high
                        )
                        with soundfile.SoundFile(noise_path) as f:
                            if f.frames == nsamples:
                                noise = f.read(dtype=np.float64, always_2d=True)
                            elif f.frames < nsamples:
                                offset = np.random.randint(0, nsamples - f.frames)
                                # noise: (Time, Nmic)
                                noise = f.read(dtype=np.float64, always_2d=True)
                                # Repeat noise
                                noise = np.pad(
                                    noise,
                                    [(offset, nsamples - f.frames - offset), (0, 0)],
                                    mode="wrap",
                                )
                            else:
                                offset = np.random.randint(0, f.frames - nsamples)
                                f.seek(offset)
                                # noise: (Time, Nmic)
                                noise = f.read(
                                    nsamples, dtype=np.float64, always_2d=True
                                )
                                if len(noise) != nsamples:
                                    raise RuntimeError(f"Something wrong: {noise_path}")
                        # noise: (Nmic, Time)
                        noise = noise.T

                        noise_power = (noise ** 2).mean()
                        scale = (
                            10 ** (-noise_db / 20)
                            * np.sqrt(power)
                            / np.sqrt(max(noise_power, 1e-10))
                        )
                        speech = speech + scale * noise

                speech = speech.T
                ma = np.max(np.abs(speech))
                if ma > 1.0:
                    speech /= ma
                data[self.speech_name] = speech

            if self.speech_volume_normalize is not None:
                speech = data[self.speech_name]
                ma = np.max(np.abs(speech))
                data[self.speech_name] = speech * self.speech_volume_normalize / ma

        if self.text_name in data and self.tokenizer is not None:
            text = data[self.text_name]
            text = self.text_cleaner(text)
            tokens = self.tokenizer.text2tokens(text)
            text_ints = self.token_id_converter.tokens2ids(tokens)
            data[self.text_name] = np.array(text_ints, dtype=np.int64)

        assert check_return_type(data)
        return data


class CommonPreprocessor_multi(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        text_cleaner: Collection[str] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        speech_name: str = "speech",
        text_name: list = ["text"],
    ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name

        if token_type is not None:
            if token_list is None:
                raise ValueError("token_list is required if token_type is not None")
            self.text_cleaner = TextCleaner(text_cleaner)

            self.tokenizer = build_tokenizer(
                token_type=token_type,
                bpemodel=bpemodel,
                delimiter=delimiter,
                space_symbol=space_symbol,
                non_linguistic_symbols=non_linguistic_symbols,
                g2p_type=g2p_type,
            )
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            self.text_cleaner = None
            self.tokenizer = None
            self.token_id_converter = None

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()

        if self.speech_name in data:
            # Nothing now: candidates:
            # - STFT
            # - Fbank
            # - CMVN
            # - Data augmentation
            pass

        for text_n in self.text_name:
            if text_n in data and self.tokenizer is not None:
                text = data[text_n]
                text = self.text_cleaner(text)
                tokens = self.tokenizer.text2tokens(text)
                text_ints = self.token_id_converter.tokens2ids(tokens)
                data[text_n] = np.array(text_ints, dtype=np.int64)
        assert check_return_type(data)
        return data
