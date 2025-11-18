import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator


def retrieve_sparse_docs():
    scores = pd.read_csv("scores_relevant_question.csv")
    scores = scores.values[:50]
    scores = list(map(lambda x: x[0], scores))

    ranges = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    max_knee = 0
    max_kneedle = None
    best_range = 0
    for bucket_range in ranges:
        bucket_lower = scores[0] - bucket_range

        doc_frequencies = []
        scores_new = []
        freq = 0
        for i in scores:
            if i >= bucket_lower:
                freq += 1
            else:
                scores_new.append(bucket_lower)
                doc_frequencies.append(freq)
                bucket_lower -= bucket_range
                freq = 0

        kneedle = KneeLocator(scores_new, doc_frequencies, S=1.0, curve='convex', direction='decreasing')

        if kneedle.knee:
            if kneedle.knee > max_knee:
                max_knee = kneedle.knee
                best_range = bucket_range
                max_kneedle = kneedle

    print(max_knee, best_range)
    num_docs = 0
    for i in scores:
        if i >= max_knee:
            num_docs += 1

    print(num_docs)

    # plt.style.use('ggplot')
    # max_kneedle.plot_knee()
    # plt.show()

    return num_docs
