"""Cluster relation embeddings, and map clusters to relations."""
import numpy as np
from sklearn.cluster import KMeans


def kmeans_clusters(embeddings, n_clusters, seed):
    """Return the k-means cluster of each embedding, using the paper's settings."""
    model = KMeans(init="k-means++", n_init=10, n_clusters=n_clusters, random_state=seed)
    return model.fit_predict(embeddings)


class ClusterTyper:
    """PromptORE clusters of related pairs, each mapped to the relations it contains.

    k-means runs on the embeddings of related train pairs, without their labels. The
    labels are only used afterwards: the score of relation r in a cluster is the share
    of the cluster's train pairs that have r. A pair is typed by its nearest cluster.
    """

    def __init__(self, relations, n_clusters, seed=0, n_init=3):
        self.relations = list(relations)
        self.kmeans = KMeans(init="k-means++", n_init=n_init, n_clusters=n_clusters,
                             random_state=seed)

    def fit(self, embeddings, labels):
        """Fit on the embeddings of related pairs and their gold relation codes."""
        clusters = self.kmeans.fit_predict(embeddings)
        position = {code: column for column, code in enumerate(self.relations)}
        counts = np.zeros((self.kmeans.n_clusters, len(self.relations)))
        for cluster, codes in zip(clusters, labels):
            for code in codes:
                counts[cluster, position[code]] += 1
        sizes = np.bincount(clusters, minlength=self.kmeans.n_clusters)
        self.precision = counts / np.maximum(sizes, 1)[:, None]
        return self

    def scores(self, embeddings):
        """Return the (pairs, relations) precision of each relation in the pair's cluster."""
        return self.precision[self.kmeans.predict(embeddings)]
