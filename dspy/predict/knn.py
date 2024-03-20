"""Predict module for K-nearest neighbors search."""

import numpy as np

from dsp.modules.sentence_vectorizer import SentenceTransformersVectorizer
from dsp.primitives.demonstrate import Example
from dsp.utils import settings


class KNN:
    def __init__(self, k: int, trainset: list[Example]):
        self.k = k
        self.trainset = trainset
        self.vectorizer = SentenceTransformersVectorizer()
        trainset_casted_to_vectorize = [
            " | ".join(
                [
                    f"{key}: {value}"
                    for key, value in example.items()
                    if key in example._input_keys
                ],
            )
            for example in self.trainset
        ]
        self.trainset_vectors = self.vectorizer(trainset_casted_to_vectorize).astype(
            np.float32,
        )

    def __call__(self, **kwargs) -> list[Example]:
        with settings.context(vectorizer=self.vectorizer):
            input_example_vector = self.vectorizer(
                [" | ".join([f"{key}: {val}" for key, val in kwargs.items()])],
            )
            scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
            nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
            train_sampled = [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]

            return train_sampled
