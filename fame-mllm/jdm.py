import json
import math
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


JSON_PATHS = (
    "/path/to/results.json",
)

WEIGHT = None

LABEL_IDS = {"real": 0, "fake": 1, "真实": 0, "虚假": 1}
VARIANTS = {
    "tt": (("real", " real", "Real", " Real"), ("fake", " fake", "Fake", " Fake")),
    "sv": (("真实", " 真实"), ("虚假", " 虚假")),
}


def metrics(truth, predictions):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "macro_precision": float(precision_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predictions, labels=[0, 1], average="macro", zero_division=0)),
    }


def extract_samples(results):
    samples, languages, variants = [], set(), set()
    for record in results:
        annotation = str(record.get("annotation") or "").strip().lower()
        if annotation not in LABEL_IDS:
            continue
        cls = (record.get("cls_result") or {}).get("cls_logits")
        text = (record.get("text_result") or {}).get("logits") or {}
        if not isinstance(cls, (list, tuple)) or len(cls) != 2:
            continue

        preferred = "sv" if annotation in ("真实", "虚假") else "tt"
        for language in (preferred, "tt" if preferred == "sv" else "sv"):
            real_keys, fake_keys = VARIANTS[language]
            real_key = next((k for k in real_keys if k in text), None)
            fake_key = next((k for k in fake_keys if k in text), None)
            if real_key is not None and fake_key is not None:
                break
        else:
            continue
        try:
            values = tuple(float(v) for v in (*cls, text[real_key], text[fake_key]))
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        samples.append((LABEL_IDS[annotation], *values))
        languages.add(language)
        variants.add(f"{real_key}/{fake_key}")
    if not samples:
        raise ValueError("No valid samples with annotations and both CLS/text logits")
    return samples, ", ".join(sorted(languages)), ", ".join(sorted(variants))


def evaluate(payload, weight=None):
    if weight is not None:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("WEIGHT must be None or a finite nonnegative number")
    samples, language, variant = extract_samples(payload.get("results", []))
    truth = [s[0] for s in samples]

    cls_preds = [1 if s[2] > s[1] else 0 for s in samples]
    cls_metrics = metrics(truth, cls_preds)

    def predictions(w):

        return [0 if cr + w * tr > cf + w * tf else 1
                for _, cr, cf, tr, tf in samples]

    if weight is None:
        best_acc, selected_weight, selected_preds = -1.0, 0.0, cls_preds
        for i in range(1, 11):
            candidate = i / 10.0
            preds = predictions(candidate)
            acc = sum(t == p for t, p in zip(truth, preds)) / len(truth)
            print(f"  text_weight={candidate:.1f} accuracy={acc:.12f}")
            if acc > best_acc:
                best_acc, selected_weight, selected_preds = acc, candidate, preds
        if best_acc <= cls_metrics["accuracy"]:
            selected_weight, selected_preds = 0.0, cls_preds
    else:

        selected_weight = float(weight)
        selected_preds = cls_preds if weight == 0 else predictions(weight)

    ensemble_metrics = metrics(truth, selected_preds)
    return {
        "best_text_weight": selected_weight,
        "is_better_than_cls": ensemble_metrics["accuracy"] > cls_metrics["accuracy"],
        "num_samples": len(samples),
        "variant_used": variant,
        "language": language,
        "cls_metrics": cls_metrics,
        "ensemble_metrics": ensemble_metrics,
    }


def main():
    for filename in JSON_PATHS:
        path = Path(filename)
        print(f"Processing: {path}")
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        entry = evaluate(payload, WEIGHT)

        updated = {"logit_ensemble": entry}
        updated.update({k: v for k, v in payload.items() if k != "logit_ensemble"})
        with path.open("w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
