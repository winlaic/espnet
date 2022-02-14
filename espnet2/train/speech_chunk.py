#!/usr/bin/env python3
from types import SimpleNamespace
import numpy as np
from typing import List

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

    def get_time_stamp(self, index):
        start_frame = sum(item.shape[0] for item in self.chunks[:index])
        end_frame = start_frame + self.chunks[index].shape[0]
        start_time, end_time = map(self.frame_to_time, (start_frame, end_frame))
        ret = SimpleNamespace()
        ret.t_start = start_time
        ret.t_end = end_time
        ret.f_start = start_frame
        ret.f_end = end_frame
        ret.text = self.words[index]
        return ret

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
            if current_frame < speech.shape[0]: # If slience exists at the end
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


    def mesh(self, filepath, figsize=(7, 2.5), upper_text=False, highlights=None, **kwargs):
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle

        non_sil_chunk_ids = self.non_silence_chunk_indices

        x_ticks = set(item.f_start for item in [self.get_time_stamp(i) for i in non_sil_chunk_ids])
        x_ticks = x_ticks | set(item.f_end for item in [self.get_time_stamp(i) for i in non_sil_chunk_ids])
        x_ticks = list(x_ticks)
        x_ticks = sorted(x_ticks)
        x_ticklabels = list(map(self.frame_to_time, x_ticks))
        x_ticklabels = list(map(lambda x: '{:.1f}'.format(x), x_ticklabels))

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.imshow(self.speech.T[:80], origin='lower')
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticklabels)

        yax = ax1.get_yaxis()
        yax.set_visible(False)

        ax1.set_xlabel('Time (s)')
        ax1.vlines(x_ticks, ymin=0, ymax=ax1.get_ylim()[1], colors='r', linestyles='dashed', linewidth=1)

        for i in self.non_silence_chunk_indices:
            test_trunk = self.get_time_stamp(i)
            ax1.text((test_trunk.f_start + test_trunk.f_end) / 2, 82, test_trunk.text.upper() if upper_text else test_trunk.text, ha='center', va='bottom')
            if highlights and i in highlights:
                rect = Rectangle((test_trunk.f_start, 0), test_trunk.f_end - test_trunk.f_start, 80, alpha=0.3, color='k')
                ax1.add_patch(rect)

        fig.savefig(filepath, **kwargs)


    @property
    def speech(self):
        return np.concatenate(self.chunks, axis=0)

    @property
    def text(self):
        return ' '.join(item for item in self.words if len(item) != 0)

    @property
    def non_silence_chunk_indices(self):
        return [i for i in range(len(self)) if len(self.words[i]) > 0]


