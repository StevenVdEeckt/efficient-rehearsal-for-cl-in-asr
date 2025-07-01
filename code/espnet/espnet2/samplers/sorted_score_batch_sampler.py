import logging
from typing import Iterator, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

import random

from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler

class SelectiveBatchSampler(AbsSampler):
    """Abstract BatchSampler that defines a method for setting the selection percentage."""

    def set_percentage(self, percentage: float):
        pass

class SortedScoreBatchSampler(SelectiveBatchSampler):
    """BatchSampler with samples sorted by dynamic scores.

    Args:
        batch_size:
        shape_file:
        initial_scores: A dict with initial scores for each utterance.
        sort_in_batch: 'descending', 'ascending', 'random', or None.
        sort_batch: 'descending', 'ascending', 'random',  or None.
        drop_last: Whether to drop the last incomplete batch.
        max_score: Utterances with a score higher than this value will be skipped.
    """

    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        initial_scores: Optional[Dict[str, float]] = None,
        sort_in_batch: str = "asscending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        max_score: float = float("inf"),
        mean_score: float = None,
        grouped: bool = False,
    ):
        assert check_argument_types()
        assert batch_size > 0
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.scores = initial_scores
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # Load utterance shapes
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        self.utt_keys = list(utt2shape.keys())

        # set scores
        if initial_scores is not None:
            self.scores = initial_scores
            missing_keys = 0
            for key in self.utt_keys:
                if not key in self.scores.keys():
                    self.scores[key] = max(self.scores.values())
                    missing_keys += 1
            logging.info(f"Using GIVEN SCORES -- There were {missing_keys} missing keys..")
            self.grouped = False
        else:
            self.scores = {k: random.random() for k in self.utt_keys}
            logging.info(f"Using RANDOM SCORES")
            self.grouped = grouped

        if self.grouped:
            self.utt2lang = self._get_utt2lang(self.utt_keys)
            logging.info(f"Utterance to language: {self.utt2lang}")
        else:
            self.utt2lang = None
 
        self.max_score = max_score
        if self.sort_in_batch.startswith("mid"):
            self.sort_in_batch = 'ascending'
            self.sort_batch = 'ascending'
            mean_score_method  = self.sort_in_batch.split("-")
            if mean_score is None and mean_score_method == 'mean':
                mean_score = 1.0 * sum(list(self.scores.values())) / max(1, len(self.scores))
            elif mean_score is None:
                mean_score = np.median(list(self.scores.values()))
            self.scores = {k: abs(v - mean_score) for k, v in self.scores.items()}
            logging.info(f"Using sort_in_batch=mid, with MEAN_SCORE = {mean_score:.2f}") 

        if len(self.utt_keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_file}")
        self._prepare_batches()

    @staticmethod
    def _get_utt2lang(utt_keys):
        def get_lang(utt_key):
            speaker_id = utt_key.split("-")[1]
            lang = speaker_id.split("_")[2]
            return lang
        utt2lang = {utt: get_lang(utt) for utt in utt_keys}
        return utt2lang

    def _prepare_batches(self):
        """Prepare batches based on the current scores."""
        # Filter out utterances exceeding max_score
        filtered_keys = [
            k for k in self.utt_keys if self.scores[k] <= self.max_score
        ]

        if len(filtered_keys) == 0:
            logging.warning(f"All utterances are filtered out by max_score={self.max_score}.")
            self.batch_list = []
            return

        if self.grouped:
            self.batch_list = self._prepare_grouped_batches(filtered_keys)
        else:
            self.batch_list = self._prepare_ungrouped_batches(filtered_keys)

        if self.sort_batch == "descending":
            self.batch_list.reverse()
        elif self.sort_batch == "ascending":
            pass  # Already in ascending order
        elif self.sort_batch == "random":
            random.shuffle(self.batch_list)
        else:
            raise ValueError(
                f"sort_batch must be one of 'ascending', 'descending', or 'random': {self.sort_batch}"
            )

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    def _prepare_grouped_batches(
            self,
            filtered_keys,
    ):
        lang2utts = defaultdict(list)
        for k in filtered_keys:
            lang2utts[self.utt2lang[k]].append(k)

        logging.info(f"Sampler found {len(lang2utts)} languages: {['%s : %d' % (lang, len(utts)) for (lang, utts) in lang2utts.items()]}")

        batch_list = []
        for lang, keys in lang2utts.items():
            if self.sort_in_batch == "random":
                random.shuffle(keys)
            elif self.sort_in_batch == "ascending":
                keys = sorted(keys, key=lambda k: self.scores[k])
            elif self.sort_in_batch == "descending":
                keys = sorted(keys, key=lambda k: -self.scores[k])
            else:
                raise ValueError(f"Unsupported sort_in_batch: {self.sort_in_batch}")

            # Create batches within this language
            N = max(len(keys) // self.batch_size, 1)
            if not self.drop_last:
                lang_batches = [
                    tuple(keys[i * len(keys) // N : (i + 1) * len(keys) // N])
                    for i in range(N)
                ]
            else:
                lang_batches = [
                    tuple(keys[i * self.batch_size : (i + 1) * self.batch_size])
                    for i in range(N)
                    if (i + 1) * self.batch_size <= len(keys)
                ]

            batch_list.extend(lang_batches)
        return batch_list

    def _prepare_ungrouped_batches(
            self,
            filtered_keys,
    ):
        # Sort the filtered keys
        if self.sort_in_batch == "descending":
            sorted_keys = sorted(filtered_keys, key=lambda k: -self.scores[k])
        elif self.sort_in_batch == "ascending":
            sorted_keys = sorted(filtered_keys, key=lambda k: self.scores[k])
        elif self.sort_in_batch == "random":
            sorted_keys = filtered_keys[:]
            random.shuffle(sorted_keys)
        else:
            raise ValueError(
                f"sort_in_batch must be either one of ascending, descending, or None: {self.sort_in_batch}"
            )

        # Split sorted keys into batches
        N = max(len(sorted_keys) // self.batch_size, 1)
        if not self.drop_last:
            batch_list = [
                tuple(sorted_keys[i * len(sorted_keys) // N : (i + 1) * len(sorted_keys) // N])
                for i in range(N)
            ]
        else:
            batch_list = [
                tuple(sorted_keys[i * self.batch_size : (i + 1) * self.batch_size])
                for i in range(N)
            ]
        return batch_list

    def update_scores(self, new_scores: Dict[str, float]):
        """Update scores and reorder batches."""
        self.scores.update(new_scores)
        self._prepare_batches()

    def set_max_score(self, max_score: float):
        """Set a new max_score and update batches."""
        self.max_score = max_score
        self._prepare_batches()

    def set_percentage(self, percentage: float):
        """Set a new selection percentage and update batches."""
        assert 0 < percentage <= 1, "Percentage must be between 0 and 1"
        scores = np.array(list(self.scores.values()))
        threshold = np.percentile(scores, percentage * 100)
        self.set_max_score(threshold)
        logging.info(f"Threshold in WER: {threshold:.2f}") 

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch}, "
            f"grouped={self.grouped})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)

