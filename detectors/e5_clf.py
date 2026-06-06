import json
import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("~/Documents/pan25-generative-ai-detection-task1-train").expanduser()
MDL_DIR = Path("~/Documents/pan-project/models").expanduser()

MDL_NM = "intfloat/e5-small-v2"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def _get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_texts(txts, mdl_nm=MDL_NM, batch_sz=32, max_len=512):
    import torch
    from transformers import AutoTokenizer, AutoModel

    dev = _get_device()
    print(f"loading {mdl_nm} on {dev}")
    tok = AutoTokenizer.from_pretrained(mdl_nm)
    model = AutoModel.from_pretrained(mdl_nm).to(dev)
    model.eval()

    # e5 models expect "query: " or "passage: " prefix
    prefixed = [f"passage: {t}" for t in txts]

    all_embs = []
    for i in range(0, len(prefixed), batch_sz):
        batch = prefixed[i:i + batch_sz]
        enc = tok(batch, return_tensors="pt", max_length=max_len,
                  truncation=True, padding=True).to(dev)
        with torch.no_grad():
            out = model(**enc)
        mask = enc.attention_mask.unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu().float().numpy())

        if (i // batch_sz + 1) % 10 == 0:
            print(f"  embedded {i + len(batch)}/{len(txts)}")

    return np.vstack(all_embs)


class E5Clf:
    def __init__(self, mdl_nm=MDL_NM):
        self.mdl_nm = mdl_nm
        self.scaler = StandardScaler()
        self.clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def fit(self, trn_recs, val_recs=None):
        print("embedding train texts...")
        X_trn = embed_texts([r["text"] for r in trn_recs], self.mdl_nm)
        y_trn = np.array([r["label"] for r in trn_recs])
        X_trn = self.scaler.fit_transform(X_trn)

        if val_recs:
            print("embedding val texts...")
            X_val = embed_texts([r["text"] for r in val_recs], self.mdl_nm)
            y_val = np.array([r["label"] for r in val_recs])
            X_val = self.scaler.transform(X_val)
            from lightgbm import early_stopping, log_evaluation
            self.clf.fit(X_trn, y_trn,
                         eval_set=[(X_val, y_val)],
                         callbacks=[early_stopping(50, verbose=False), log_evaluation(100)])
        else:
            self.clf.fit(X_trn, y_trn)

    def predict_proba(self, recs):
        X = embed_texts([r["text"] for r in recs], self.mdl_nm)
        X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)[:, 1]
        return [r["id"] for r in recs], proba

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
