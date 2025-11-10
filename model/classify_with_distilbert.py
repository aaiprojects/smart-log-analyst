#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import re
import json
from statistics import median
from collections import Counter, defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Regex helpers ----------
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
ERROR_KWS = re.compile(
    r"\b(error|exception|fatal|panic|stacktrace|refused|failed|failure|crash)\b",
    re.I
)
SECURITY_KWS = re.compile(
    r"\b(unauthori[sz]ed|forbidden|permission denied|invalid cert|certificate|ssl|tls|"
    r"handshake|token expired|csrf|xss|secrets?|private key|keystore|vault|kms)\b", re.I
)
WARNING_KWS = re.compile(
    r"\b(timeout|retry|backoff|degrad|slow|throttl|overload|queue length|high latency)\b",
    re.I
)
INFO_KWS = re.compile(
    r"\b(subscribed|assigned|connected|started|completed|initialized|init|ready|healthy|"
    r"joined|sync(ed)?|refresh(ed)?)\b",
    re.I
)

# Canonical alias tags for semantic duplicates
ALIASES = [
    (re.compile(r"(timeout.*(db|database)|cannot.*(db|database).*reach|connection.*(db|database).*(timeout|refused))", re.I), "DB_CONNECT_TIMEOUT"),
    (re.compile(r"(connection.*refused|econnrefused|broken pipe|connection reset)", re.I), "NET_CONNECT_REFUSED"),
    (re.compile(r"(host.*unreachable|no route to host|network.*unreachable)", re.I), "NET_UNREACHABLE"),
    (re.compile(r"(authentication failed|auth.*failed|invalid cred|bad credentials)", re.I), "AUTH_FAIL"),
]

# Context chains (ordered markers) — extend as needed
CONTEXT_CHAINS = [
    (["offset commit failed", "rebalance", "partition revoked"], "KAFKA_CONSUMER_INSTABILITY"),
    (["connection timeout", "retry", "backoff"], "NET_RETRY_BACKOFF"),
]

# ---- New aggregation helpers ----
def aggregate_topk_mean(scores, k: int) -> float:
    if not scores:
        return 0.0
    k = max(1, min(k, len(scores)))
    topk = sorted(scores, reverse=True)[:k]
    return float(sum(topk) / k)

def split_window_lines(seq_text: str):
    return [s.strip() for s in str(seq_text).split("[SEP]") if s and s.strip()]

# ---------- HF scorer ----------
class DistilBERTScorer:
    """
    Expects a HF sequence classifier (binary or multi-logit).
    Returns an anomaly/relevance-like score in [0,1]:
      - 1-logit: sigmoid(logit)
      - 2-logit: softmax(logits)[positive_class]
      - >2-logit: max softmax prob (proxy for confidence)
    """
    def __init__(self, model_name_or_path: str, device: str = None, max_length: int = 512, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model/tokenizer from '{model_name_or_path}': {e}"
            )
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        # resolve positive class index if possible
        self.pos_idx = 1
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict):
            normalized = {int(i): str(v).lower() for i, v in id2label.items()}
            cand = [i for i, name in normalized.items()
                    if any(k in name for k in ["anomaly","positive","pos","relevant","label_1","toxic","error"])]
            if cand:
                self.pos_idx = int(cand[0])

    @torch.no_grad()
    def score_texts(self, texts):
        """Batched scoring for a list of texts."""
        if not texts:
            return np.array([], dtype=float)
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits

            # 1-logit binary model
            if logits.shape[-1] == 1:
                z = logits.squeeze(-1)
                scores = torch.sigmoid(z)

            # 2-logit binary model
            elif logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)
                scores = probs[..., self.pos_idx]

            # multi-class: use max prob as relevance proxy
            else:
                probs = torch.softmax(logits, dim=-1)
                scores = probs.max(dim=-1).values

            all_scores.extend(scores.float().cpu().tolist())

        return np.array(all_scores, dtype=float).reshape(-1)

# ---------- Data utils ----------
def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv)
    assert {"line_idx", "template_id", "raw"}.issubset(set(df.columns))
    return df.sort_values("line_idx").reset_index(drop=True)

def build_vocab(template_ids):
    tids = list(dict.fromkeys(template_ids))  # preserve order of first appearance
    id2ix = {tid: i+1 for i, tid in enumerate(tids)}  # 0 reserved for PAD
    ix2id = {i+1: tid for i, tid in enumerate(tids)}
    return id2ix, ix2id

def make_sequences(template_ids, line_idx, window=20, stride=1):
    seqs = []
    n = len(template_ids)
    i = 0
    while i + window <= n:
        seqs.append((template_ids[i:i+window], line_idx[i:i+window]))
        i += stride
    return seqs

def extract_numbers(text: str):
    return [float(x) for x in NUM_RE.findall(text)]

def majority(iterable):
    c = Counter(iterable)
    if not c:
        return None
    return c.most_common(1)[0][0]

def compute_template_stats(df: pd.DataFrame):
    """
    - freq per template
    - numeric baselines per template (median of all numbers seen in lines of that template)
    """
    tpl_freq = df["template_id"].value_counts().to_dict()
    tpl_num_values = defaultdict(list)
    for _, row in df.iterrows():
        nums = extract_numbers(str(row["raw"]))
        if nums:
            tpl_num_values[row["template_id"]].extend(nums)
    tpl_num_median = {k: (median(v) if v else None) for k, v in tpl_num_values.items()}
    return tpl_freq, tpl_num_median

def canonical_tags_for_text(text: str):
    tags = []
    for rx, tag in ALIASES:
        if rx.search(text):
            tags.append(tag)
    return tags

def has_context_chain(raw_texts_lower):
    joined = " || ".join(raw_texts_lower)
    for markers, tag in CONTEXT_CHAINS:
        pos = 0
        ok = True
        for m in markers:
            idx = joined.find(m)
            if idx == -1 or idx < pos:
                ok = False
                break
            pos = idx
        if ok:
            return True, tag
    return False, None

def drift_against_template_baseline(raw_texts, tpl_ids, tpl_medians, drift_factor=3.0):
    """
    If any numeric value in a line exceeds (median * drift_factor) for that template,
    we consider it performance drift.
    """
    reasons = []
    for raw, tid in zip(raw_texts, tpl_ids if tpl_ids is not None else []):
        med = tpl_medians.get(tid)
        if med is None or med <= 0:
            continue
        nums = extract_numbers(str(raw))
        for v in nums:
            if v > med * drift_factor:
                reasons.append(f"value {v} > {drift_factor}x median {med} for {tid}")
                return True, reasons
    return False, reasons

# ---------- Core taxonomy classifier ----------
def classify_sequence(
    seq_score: float,
    seq_tpl_ids,
    seq_raws,
    tpl_freq: dict,
    tpl_medians: dict,
    hi_thresh: float,
    lo_thresh: float,
    heartbeat_freq_threshold: int = 50,
    drift_factor: float = 3.0,
):
    """
    Returns (category, sub_type, confidence, reasons)
    Categories: {Noise, Info, Warning, Error, Security, Unknown}
    SubTypes : {Heartbeat, NormalOperation, RecoverableAnomaly, PerformanceDrift,
                SemanticDuplicate, ContextChain, HiddenAnomaly, FailureEvent,
                PolicyViolation, OOD}
    """
    reasons = []
    raws = [str(x) for x in seq_raws]
    raws_lower = [r.lower() for r in raws]

    # Band by score (window_score or line_score)
    if seq_score >= hi_thresh:
        label_band = "relevant"
    elif seq_score <= lo_thresh:
        label_band = "irrelevant"
    else:
        label_band = "unknown"

    # NEW: only fast-exit on explicit 'irrelevant'
    if label_band == "irrelevant":
        reasons.append("model_irrelevant")
        return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)

    # From here: 'relevant' or 'unknown' → run rules.

    # Security
    if any(SECURITY_KWS.search(r) for r in raws):
        reasons.append("security_keyword_hit")
        return ("Security", "PolicyViolation", float(max(seq_score, hi_thresh)), reasons)

    # Errors
    if any(ERROR_KWS.search(r) for r in raws):
        reasons.append("error_keyword_hit")
        return ("Error", "FailureEvent", float(max(seq_score, hi_thresh)), reasons)

    # Context chain detection
    ctx_hit, ctx_tag = has_context_chain(raws_lower)
    if ctx_hit:
        reasons.append(f"context_chain:{ctx_tag}")
        return ("Warning", "ContextChain", float(max(seq_score, lo_thresh)), reasons)

    # Performance drift detection
    drift_hit, drift_reasons = drift_against_template_baseline(
        raws, seq_tpl_ids, tpl_medians or {}, drift_factor=drift_factor
    )
    if drift_hit:
        reasons.extend(drift_reasons)
        return ("Warning", "PerformanceDrift", float(max(seq_score, lo_thresh)), reasons)

    # Semantic alias / duplicate
    tags = []
    for r in raws:
        tags.extend(canonical_tags_for_text(r))
    if len(set(tags)) >= 1 and len(tags) >= 2:
        reasons.append(f"semantic_aliases:{sorted(set(tags))}")
        cat = "Error" if label_band == "relevant" else "Warning"
        sub = "FailureEvent" if cat == "Error" else "SemanticDuplicate"
        return (cat, sub, float(max(seq_score, lo_thresh)), reasons)

    # Heartbeat / Noise (use template IDs for frequency)
    if tpl_freq and seq_tpl_ids:
        seq_tpl_freqs = [tpl_freq.get(t, 0) for t in seq_tpl_ids]
        if np.median(seq_tpl_freqs) >= heartbeat_freq_threshold and label_band != "relevant":
            reasons.append(f"heartbeat_freq_median>={heartbeat_freq_threshold}")
            return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)

    # Warnings and info
    if any(WARNING_KWS.search(r) for r in raws):
        reasons.append("warning_keyword_hit")
        return ("Warning", "RecoverableAnomaly", float(max(seq_score, lo_thresh)), reasons)

    if any(INFO_KWS.search(r) for r in raws) and label_band != "relevant":
        reasons.append("info_keyword_hit")
        return ("Info", "NormalOperation", float(1.0 - seq_score), reasons)

    # Fallbacks (if still unknown, use keyword hints)
    if label_band == "unknown":
        if any(ERROR_KWS.search(r) for r in raws):
            reasons.append("error_keyword_hit_fallback")
            return ("Error", "FailureEvent", float(max(seq_score, hi_thresh)), reasons)
        if any(WARNING_KWS.search(r) for r in raws):
            reasons.append("warning_keyword_hit_fallback")
            return ("Warning", "RecoverableAnomaly", float(max(seq_score, lo_thresh)), reasons)
        if any(INFO_KWS.search(r) for r in raws):
            reasons.append("info_keyword_hit_fallback")
            return ("Info", "NormalOperation", float(1.0 - seq_score), reasons)

    # Final fallback by band
    if label_band == "relevant":
        reasons.append("high_score_band")
        return ("Warning", "RecoverableAnomaly", float(seq_score), reasons)

    reasons.append("no_rule_matched")
    return ("Unknown", "OOD", float(seq_score), reasons)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Classify logs with DistilBERT (window or line-by-line)")
    ap.add_argument("--mapping_csv", default="logs_to_templates.csv",
                    help="Output produced by Program 1")
    ap.add_argument("--templates_jsonl", default="drain3_templates.jsonl",
                    help="Templates jsonl (optional; used to build textual sequences)")
    ap.add_argument("--distilbert_model", required=True,
                    help="HF model name or local path (e.g., 'distilbert-base-uncased' or ./ckpt)")
    ap.add_argument("--window", type=int, default=20, help="Sequence length")
    ap.add_argument("--stride", type=int, default=5, help="Sliding stride")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512,
                    help="Max tokens per line after tokenization")
    ap.add_argument("--hi_thresh", type=float, default=0.75,
                    help="score>=hi → relevant band")
    ap.add_argument("--lo_thresh", type=float, default=0.25,
                    help="score<=lo → irrelevant band")
    ap.add_argument("--auto_calibrate", action="store_true", default=True,
                    help="Auto set hi/lo to 90th/10th percentiles (default: on). Use --no-auto_calibrate to disable.")
    ap.add_argument("--no-auto_calibrate", dest="auto_calibrate", action="store_false")
    ap.add_argument("--heartbeat_freq_threshold", type=int, default=50,
                    help="Median template frequency in a sequence to call Heartbeat")
    ap.add_argument("--drift_factor", type=float, default=3.0,
                    help="x * template median numeric value → PerformanceDrift")
    ap.add_argument("--seq_out", default="distilbert_sequences.csv",
                    help="Per-sequence scores and labels")
    ap.add_argument("--tpl_out", default="distilbert_templates_summary.csv",
                    help="Per-template average score & label summary")
    # NEW:
    ap.add_argument("--top_k", type=int, default=3,
                    help="Aggregate mean of top-k per-line scores → window_score")
    ap.add_argument("--emit_line_scores", action="store_true",
                    help="Emit per-line scores per window (as a compact string)")
    # LINE-BY-LINE mode
    ap.add_argument("--line_by_line", action="store_true",
                    help="Score each raw log line independently (no windows). Outputs to --line_out.")
    ap.add_argument("--line_out", default="distilbert_lines.csv",
                    help="Per-line scores and labels (only when --line_by_line).")
    args = ap.parse_args()

    # Load mapping
    df = load_mapping(Path(args.mapping_csv))
    template_ids = df["template_id"].tolist()
    line_idx = df["line_idx"].tolist()

    # Common stats for rule helpers
    tpl_freq, tpl_medians = compute_template_stats(df)
    idx2raw = dict(zip(df["line_idx"].tolist(), df["raw"].tolist()))

    # -------- LINE-BY-LINE BRANCH --------
    if args.line_by_line:
        # Prepare texts: just the raw lines in original order
        raw_texts = [idx2raw[i] for i in line_idx]

        scorer = DistilBERTScorer(
            args.distilbert_model, max_length=args.max_length, batch_size=args.batch_size
        )
        print("Scoring lines...")
        line_scores = scorer.score_texts(raw_texts).astype(float)

        # Instrumentation: show distribution
        if len(line_scores):
            q25 = float(np.quantile(line_scores, 0.25))
            q50 = float(np.median(line_scores))
            q75 = float(np.quantile(line_scores, 0.75))
            print(f"[line_scores] min={line_scores.min():.3f} p25={q25:.3f} median={q50:.3f} p75={q75:.3f} max={line_scores.max():.3f}")

        # Auto-calibrate on line scores
        if args.auto_calibrate and len(line_scores) >= 10:
            q10, q90 = np.quantile(line_scores, [0.10, 0.90])
            if q90 - q10 > 1e-6:
                args.hi_thresh = float(q90)
                args.lo_thresh = float(q10)
                print(f"[auto-calibrate] hi={args.hi_thresh:.3f}, lo={args.lo_thresh:.3f}")

        def to_label(s):
            if s >= args.hi_thresh:
                return "relevant"
            elif s <= args.lo_thresh:
                return "irrelevant"
            else:
                return "unknown"

        # Classify each line via the same taxonomy (wrap as length-1 sequence)
        rows = []
        for i, (idx, tid, raw, score) in enumerate(zip(line_idx, template_ids, raw_texts, line_scores)):
            category, sub_type, confidence, reasons = classify_sequence(
                seq_score=float(score),           # per-line score
                seq_tpl_ids=[tid],                # single template
                seq_raws=[raw],                   # single raw line
                tpl_freq=tpl_freq,
                tpl_medians=tpl_medians,
                hi_thresh=args.hi_thresh,
                lo_thresh=args.lo_thresh,
                heartbeat_freq_threshold=args.heartbeat_freq_threshold,
                drift_factor=args.drift_factor,
            )
            rows.append({
                "line_idx": idx,
                "template_id": tid,
                "raw": raw,
                "line_score": round(float(score), 6),
                "label_band": to_label(float(score)),
                "category": category,
                "sub_type": sub_type,
                "confidence": round(float(confidence), 6),
                "reasons": "|".join(reasons),
            })

        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            print("[bands]\n", df_out["label_band"].value_counts(dropna=False))
            print("[top categories]\n", df_out["category"].value_counts(dropna=False).head(10))
        df_out.to_csv(args.line_out, index=False)
        print(f"[OK] Per-line → {args.line_out}")
        print("Bands:",
              f"line_score >= {args.hi_thresh:.3f} → relevant;",
              f"<= {args.lo_thresh:.3f} → irrelevant; else unknown")
        return

    # -------- WINDOW (DEFAULT) BRANCH --------
    id2ix, ix2id = build_vocab(template_ids)
    seqs = make_sequences(template_ids, line_idx, window=args.window, stride=args.stride)
    if not seqs:
        raise RuntimeError("Not enough lines to form any sequence. Reduce --window or add logs.")

    # Optional: load templates (turn sequences into text)
    templates_map = {}
    tpl_jsonl = Path(args.templates_jsonl)
    if tpl_jsonl.exists():
        with tpl_jsonl.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    tid = obj.get("template_id") or obj.get("cluster_id") or obj.get("id")
                    txt = obj.get("template") or obj.get("text") or ""
                    if tid:
                        templates_map[str(tid)] = txt
                except Exception:
                    continue

    # Prepare texts per sequence (join with [SEP]) and raw members
    seq_member_tpls = []
    seq_member_raws = []
    seq_texts = []
    for (seq_tids, member_idxs) in seqs:
        seq_member_tpls.append(seq_tids)
        raws = [idx2raw[i] for i in member_idxs]
        seq_member_raws.append(raws)
        parts = []
        for tid, raw in zip(seq_tids, raws):
            ttxt = templates_map.get(str(tid))
            parts.append(ttxt if (ttxt and ttxt.strip()) else str(raw))
        seq_texts.append(" [SEP] ".join(parts))

    # Score with DistilBERT (per-line → top-k mean aggregation)
    scorer = DistilBERTScorer(
        args.distilbert_model, max_length=args.max_length, batch_size=args.batch_size
    )

    window_scores = []
    line_scores_dump = []  # optional column if --emit_line_scores
    for seq_txt in tqdm(seq_texts, desc="Scoring by lines"):
        lines = split_window_lines(seq_txt)
        line_scores = scorer.score_texts(lines).tolist() if lines else []
        wscore = aggregate_topk_mean(line_scores, args.top_k)
        window_scores.append(wscore)

        if args.emit_line_scores:
            joined = ";".join(
                [f"{lines[i]}\t{float(line_scores[i]):.4f}" for i in range(len(line_scores))]
            )
            line_scores_dump.append(joined)

    scores_all = np.array(window_scores, dtype=float)

    # Instrumentation: show distribution
    if len(scores_all):
        q25 = float(np.quantile(scores_all, 0.25))
        q50 = float(np.median(scores_all))
        q75 = float(np.quantile(scores_all, 0.75))
        print(f"[window_scores] min={scores_all.min():.3f} p25={q25:.3f} median={q50:.3f} p75={q75:.3f} max={scores_all.max():.3f}")

    # Optional auto-calibration of bands on window scores
    if args.auto_calibrate and len(scores_all) >= 10:
        q10, q90 = np.quantile(scores_all, [0.10, 0.90])
        if q90 - q10 > 1e-6:
            args.hi_thresh = float(q90)
            args.lo_thresh = float(q10)
            print(f"[auto-calibrate] hi={args.hi_thresh:.3f}, lo={args.lo_thresh:.3f}")

    # Band function on window scores
    def to_label(s):
        if s >= args.hi_thresh:
            return "relevant"
        elif s <= args.lo_thresh:
            return "irrelevant"
        else:
            return "unknown"

    labels = [to_label(s) for s in scores_all]

    # Save per-sequence with rich taxonomy
    seq_rows = []
    for idx, (score, label, (seq_tids, member_idxs), seq_raws) in enumerate(
        zip(scores_all, labels, seqs, seq_member_raws)
    ):
        category, sub_type, confidence, reasons = classify_sequence(
            seq_score=float(score),  # window_score
            seq_tpl_ids=seq_tids,
            seq_raws=seq_raws,
            tpl_freq=tpl_freq,
            tpl_medians=tpl_medians,
            hi_thresh=args.hi_thresh,
            lo_thresh=args.lo_thresh,
            heartbeat_freq_threshold=args.heartbeat_freq_threshold,
            drift_factor=args.drift_factor,
        )
        row = {
            "seq_idx": idx,
            "seq_text": seq_texts[idx],
            "window_score": round(float(score), 6),  # renamed for clarity
            "label_band": label,                     # relevant / unknown / irrelevant
            "category": category,                    # high-level category
            "sub_type": sub_type,                    # sub-type
            "confidence": round(float(confidence), 6),
            "reasons": "|".join(reasons),
            "member_line_idxs": ";".join(map(str, member_idxs))
        }
        if args.emit_line_scores:
            # guard in case list lengths differ
            if idx < len(line_scores_dump):
                row["line_scores"] = line_scores_dump[idx]
        seq_rows.append(row)

    df_seq = pd.DataFrame(seq_rows)
    df_seq.to_csv(args.seq_out, index=False)

    # Per-template summary (avg of window scores where the template appears)
    tpl_scores = {tid: [] for tid in set(template_ids)}
    tpl_cats = defaultdict(list)
    tpl_subs = defaultdict(list)
    for (seq_tids, _), score, row in zip(seqs, scores_all, seq_rows):
        for tid in set(seq_tids):
            tpl_scores[tid].append(float(score))
            tpl_cats[tid].append(row["category"])
            tpl_subs[tid].append(row["sub_type"])

    tpl_summary = []
    for tid, arr in tpl_scores.items():
        if arr:
            avg = float(np.mean(arr))
            lbl = to_label(avg)
            tpl_summary.append({
                "template_id": tid,
                "avg_window_score": round(avg, 6),
                "label_band": lbl,
                "top_category": majority(tpl_cats[tid]),
                "top_sub_type": majority(tpl_subs[tid]),
                "n_sequences": len(arr),
            })

    df_tpl = pd.DataFrame(tpl_summary).sort_values("avg_window_score", ascending=False)
    df_tpl.to_csv(args.tpl_out, index=False)

    # Instrumentation: quick label/category breakdown
    if len(labels):
        print("[bands]\n", pd.Series(labels).value_counts(dropna=False))
    if not df_seq.empty:
        print("[top categories]\n", df_seq['category'].value_counts(dropna=False).head(10))

    print(f"[OK] Per-sequence → {args.seq_out}")
    print(f"[OK] Per-template → {args.tpl_out}")
    print("Bands:",
          f"window_score >= {args.hi_thresh:.3f} → relevant;",
          f"<= {args.lo_thresh:.3f} → irrelevant; else unknown")
    print("Added columns: window_score, label_band, category, sub_type, confidence, reasons"
          + (" + line_scores" if args.emit_line_scores else ""))

if __name__ == "__main__":
    main()
