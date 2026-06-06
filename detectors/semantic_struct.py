import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import entropy as scipy_entropy


def tokenize_words(txt):
    return re.findall(r"\b\w+\b", txt.lower())


def build_semantic_space(corpus_txts, n=2, max_ftrs=2000, n_clusters=50):
    vec = TfidfVectorizer(
        ngram_range=(n, n),
        max_features=max_ftrs,
        min_df=5,
        sublinear_tf=True,
    )
    X = vec.fit_transform(corpus_txts)
    X_norm = normalize(X, norm="l2")

    print(f"fitting kmeans (k={n_clusters}) on {X_norm.shape}...")
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5, batch_size=1000)
    km.fit(X_norm)

    return vec, km


def cluster_dist_ftrs(txt, vec, km):
    X = vec.transform([txt])
    X_norm = normalize(X, norm="l2")

    dists = km.transform(X_norm)[0]
    sim = 1.0 / (dists + 1e-8)
    sim = sim / sim.sum()

    k = len(sim)
    h = float(scipy_entropy(sim))
    h_max = float(np.log(k))
    h_norm = h / h_max if h_max > 0 else 0

    hard_assign = km.predict(X_norm)[0]
    top3_mass = float(np.sort(sim)[-3:].sum())
    gini = float(1 - np.sum(sim ** 2))

    return {
        "sem_entropy": h,
        "sem_entropy_norm": h_norm,
        "sem_top3_mass": top3_mass,
        "sem_gini": gini,
        "sem_max_sim": float(sim.max()),
        "sem_cluster_id": int(hard_assign),
    }


def cluster_dist_ftrs_vec(txt, vec, km):
    features = cluster_dist_ftrs(txt, vec, km)
    features.pop("sem_cluster_id", None)
    return features


class SemanticStructClf:
    def __init__(self, n_clusters=50, n_gram=2, max_ftrs=2000):
        self.n_clusters = n_clusters
        self.n_gram = n_gram
        self.max_ftrs = max_ftrs
        self.vec = None
        self.km = None
        from lightgbm import LGBMClassifier
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.clf = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )

    def fit(self, trn_recs, val_recs=None):
        txts = [r["text"] for r in trn_recs]
        labels = np.array([r["label"] for r in trn_recs])

        self.vec, self.km = build_semantic_space(
            txts, self.n_gram, self.max_ftrs, self.n_clusters
        )

        print("extracting train semantic features...")
        X_trn = self._extract(trn_recs)
        X_trn = self.scaler.fit_transform(X_trn)

        if val_recs:
            print("extracting val semantic features...")
            X_val = self._extract(val_recs)
            X_val = self.scaler.transform(X_val)
            y_val = np.array([r["label"] for r in val_recs])
            from lightgbm import early_stopping, log_evaluation
            self.clf.fit(X_trn, labels,
                         eval_set=[(X_val, y_val)],
                         callbacks=[early_stopping(50, verbose=False), log_evaluation(100)])
        else:
            self.clf.fit(X_trn, labels)

    def _extract(self, recs):
        rows = []
        for i, r in enumerate(recs):
            features = cluster_dist_ftrs_vec(r["text"], self.vec, self.km)
            rows.append(list(features.values()))
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(recs)}")
        return np.array(rows, dtype=np.float32)

    def predict_proba(self, recs):
        X = self._extract(recs)
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)[:, 1]
        return [r["id"] for r in recs], proba

    def feature_names(self):
        return ["sem_entropy", "sem_entropy_norm", "sem_top3_mass", "sem_gini", "sem_max_sim"]
