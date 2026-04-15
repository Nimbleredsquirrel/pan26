import json
import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import get_all_ftrs, ftrs_to_vec, FTR_NAMES_STYLO, FTR_NAMES_ALL

DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
MDL_DIR = Path("~/Documents/pan-project/models").expanduser()


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def extract_ftrs(recs, gltr=None, desc=""):
    X, ids = [], []
    for i, r in enumerate(recs):
        features = get_all_ftrs(r["text"], gltr)
        X.append(ftrs_to_vec(features))
        ids.append(r["id"])
        if (i + 1) % 500 == 0:
            print(f"  {desc} {i+1}/{len(recs)}")
    return np.array(X), ids


class StyloClf:
    def __init__(self, use_gltr=False, gltr_mdl="gpt2"):
        self.use_gltr = use_gltr
        self.gltr = None
        self.gltr_mdl = gltr_mdl
        self.scaler = StandardScaler()
        self.clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.feature_names = FTR_NAMES_ALL if use_gltr else FTR_NAMES_STYLO

    def _init_gltr(self):
        if self.use_gltr and self.gltr is None:
            from features import GLTRFeatures
            self.gltr = GLTRFeatures(self.gltr_mdl)

    def fit(self, trn_recs, val_recs=None):
        self._init_gltr()
        print("extracting train features...")
        X_trn, _ = extract_ftrs(trn_recs, self.gltr, "train")
        y_trn = np.array([r["label"] for r in trn_recs])

        X_trn = self.scaler.fit_transform(X_trn)

        if val_recs:
            print("extracting val features...")
            X_val, _ = extract_ftrs(val_recs, self.gltr, "val")
            X_val = self.scaler.transform(X_val)
            y_val = np.array([r["label"] for r in val_recs])
            self.clf.fit(
                X_trn, y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[lgbm_early_stop(50, verbose=False), lgbm_log(100)]
            )
        else:
            self.clf.fit(X_trn, y_trn)

        print(f"best iteration: {self.clf.best_iteration_}")

    def predict_proba(self, recs):
        self._init_gltr()
        X, ids = extract_ftrs(recs, self.gltr)
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)[:, 1]
        return ids, proba

    def feature_importance(self):
        imp = self.clf.feature_importances_
        pairs = sorted(zip(self.feature_names, imp), key=lambda x: -x[1])
        return pairs

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def lgbm_early_stop(n, verbose=True):
    from lightgbm import early_stopping
    return early_stopping(n, verbose=verbose)


def lgbm_log(n):
    from lightgbm import log_evaluation
    return log_evaluation(n)
