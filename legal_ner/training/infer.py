"""Inference: judgment PDF or text file -> extracted entities JSON.

CLI:
    python -m training.infer --model data/models/legal-ner/final --pdf "/path/x.pdf"
    python -m training.infer --model ... --text data/text/100123.txt --out result.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MAX_SEQ_LENGTH, MODELS_DIR  # noqa: E402
from corpus.extract import extract_pdf_text  # noqa: E402
from corpus.normalize import flatten_for_matching, normalize_text  # noqa: E402
from labeling.weak_label import tokenize_with_offsets  # noqa: E402

WINDOW = 200   # syllable tokens per window
STRIDE = 150   # advance per window (overlap keeps boundary entities intact)


def predict_tags(model, tokenizer, syllables: list[str], device) -> list[str]:
    """Predict a BIO tag for every syllable, with overlapping windows."""
    id2label = model.config.id2label
    tags = ["O"] * len(syllables)
    decided = [False] * len(syllables)
    start = 0
    while start < len(syllables):
        window = syllables[start: start + WINDOW]
        enc = tokenizer(window, is_split_into_words=True, truncation=True,
                        max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()
        word_ids = enc.word_ids(0)
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            gi = start + wid
            if not decided[gi]:
                tags[gi] = id2label[pred_ids[pos]]
                decided[gi] = True
        if start + WINDOW >= len(syllables):
            break
        start += STRIDE
    return tags


def tags_to_entities(tokens: list[tuple[str, int, int]], tags: list[str],
                     text: str) -> list[dict]:
    entities = []
    current = None  # (label, start_tok, end_tok)
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = [tag[2:], i, i]
        elif tag.startswith("I-") and current and tag[2:] == current[0]:
            current[2] = i
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return [
        {
            "label": label,
            "start": tokens[s][1],
            "end": tokens[e][2],
            "text": text[tokens[s][1]: tokens[e][2]],
        }
        for label, s, e in entities
    ]


def load_model(model_path: str, device: str):
    """Load tokenizer + token-classification model onto ``device`` (eval mode).

    Returned ``(tokenizer, model)`` are reusable across many inference calls;
    callers (e.g. the API singleton) load once and reuse.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()
    return tokenizer, model


def infer_entities(text: str, model, tokenizer, device) -> list[dict]:
    """Run NER over already-flattened text and return entity dicts.

    ``text`` must be the flattened-for-matching stream so that the returned
    ``start``/``end`` offsets index into it. Each entity:
    ``{"label", "start", "end", "text"}``.
    """
    tokens = tokenize_with_offsets(text)
    tags = predict_tags(model, tokenizer, [t for t, _, _ in tokens], device)
    return tags_to_entities(tokens, tags, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract legal entities from a judgment")
    parser.add_argument("--model", default=str(MODELS_DIR / "legal-ner" / "final"),
                        help="fine-tuned checkpoint dir")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="judgment PDF (text layer required)")
    source.add_argument("--text", help="plain-text judgment file")
    parser.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    args = parser.parse_args()

    if args.pdf:
        raw = extract_pdf_text(Path(args.pdf))
        if raw is None:
            sys.exit("ERROR: PDF has no text layer (scan?) — not supported.")
        text = flatten_for_matching(normalize_text(raw))
    else:
        text = flatten_for_matching(Path(args.text).read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model, device)

    entities = infer_entities(text, model, tokenizer, device)

    result = json.dumps({"source": args.pdf or args.text, "entities": entities},
                        ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"Wrote {len(entities)} entities to {args.out}")
    else:
        print(result)


if __name__ == "__main__":
    main()
