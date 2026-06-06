import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FastDetectGPT:
    def __init__(self, mdl_nm="gpt2-xl", max_len=512, n_smpl=100, offset=0.0, scale=1.0):
        self.max_len = max_len
        self.n_smpl = n_smpl
        self.offset = offset
        self.scale = scale

        dev = _get_device()
        print(f"loading {mdl_nm} on {dev}")
        self.tok = AutoTokenizer.from_pretrained(mdl_nm)
        self.model = AutoModelForCausalLM.from_pretrained(mdl_nm, torch_dtype=torch.float32)
        self.model.to(dev)
        self.model.eval()

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _raw_score(self, txt):
        inputs = self.tok(
            txt, return_tensors="pt", max_length=self.max_len,
            truncation=True, padding=False
        ).to(self.model.device)

        ids = inputs.input_ids
        if ids.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            logits = self.model(**inputs).logits

        log_p = F.log_softmax(logits[0, :-1, :], dim=-1)
        actual_log_prob = log_p.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)

        probs = log_p.exp().clamp(min=1e-9)
        samples = torch.multinomial(probs, self.n_smpl, replacement=True)
        sample_log_probs = log_p.gather(1, samples)

        mu = sample_log_probs.mean(dim=1)
        sigma = sample_log_probs.std(dim=1).clamp(min=1e-8)
        return ((actual_log_prob - mu) / sigma).mean().item()

    def score(self, txt):
        raw = self._raw_score(txt)
        return float(torch.sigmoid(torch.tensor(self.scale * (raw - self.offset))).item())

    def calibrate(self, trn_path, n_max=2000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        print("computing raw scores for calibration...")
        recs = []
        with open(trn_path) as f:
            for l in f:
                recs.append(json.loads(l))

        rng = np.random.default_rng(42)
        if len(recs) > n_max:
            recs = list(rng.choice(recs, n_max, replace=False))

        scores, labels = [], []
        for i, r in enumerate(recs):
            scores.append(self._raw_score(r["text"]))
            labels.append(r["label"])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(recs)}")

        scores = np.array(scores).reshape(-1, 1)
        labels = np.array(labels)

        scaler = StandardScaler()
        scores_scaled = scaler.fit_transform(scores)

        clf = LogisticRegression()
        clf.fit(scores_scaled, labels)

        std = scaler.scale_[0]
        mean = scaler.mean_[0]
        k = clf.coef_[0][0] / std
        b = clf.intercept_[0] - clf.coef_[0][0] * mean / std

        self.scale = float(k)
        self.offset = float(-b / k) if k != 0 else 0.0

        print(f"calibrated: scale={self.scale:.4f}  offset={self.offset:.4f}")
        return self.scale, self.offset
