import pandas as pd
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


# --- Configuration (edit if needed) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Dataset locations:

dataset_type = "test"

if dataset_type == "train":
    INPUT_DIR = Path.cwd().parent / "dataset" / "training_data" / "annotations"
    OUT_PARQUET = Path.cwd().parent / "dataset" / "parsed_OCR_data_train.parquet"

elif dataset_type == "test":
    INPUT_DIR = Path.cwd().parent / "dataset" / "testing_data" / "annotations"
    OUT_PARQUET = Path.cwd().parent / "dataset" / "parsed_OCR_data_test.parquet"

OUT_CSV = None  # e.g., os.path.join(SCRIPT_DIR, "parsed_OCR_data_train.csv")

TEXT_COL = "text"        # or "words_text" if you prefer
EMB_PREFIX = "emb_"      # prefix for embedding columns


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def parse_box(box: Optional[List[float]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    # box expected as [x1, y1, x2, y2]
    if not box or len(box) != 4:
        return None, None, None, None, None, None
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    xc = float(x1) + w / 2.0
    yc = float(y1) + h / 2.0
    aspect = (w / h) if h > 0 else None
    area = w * h
    return w, h, xc, yc, aspect, area


def words_to_text(words: Optional[List[Dict[str, Any]]]) -> str:
    if not words:
        return ""
    toks = [w.get("text", "") for w in words if isinstance(w, dict)]
    toks = [t for t in toks if isinstance(t, str) and t.strip() != ""]
    return " ".join(toks)


def build_rows_from_doc(doc: Dict[str, Any], source_path: str) -> List[Dict[str, Any]]:
    rows = []
    elements = doc.get("form", [])
    if not isinstance(elements, list):
        return rows

    # Per-document max coordinates for normalization later
    xs, ys = [], []
    for e in elements:
        box = e.get("box")
        if isinstance(box, list) and len(box) == 4:
            xs.extend([box[0], box[2]])
            ys.extend([box[1], box[3]])
    max_x = max(xs) if xs else None
    max_y = max(ys) if ys else None

    for e in elements:
        label = e.get("label")
        text = e.get("text", "")
        words = e.get("words", None)
        words_text = words_to_text(words)
        box = e.get("box", None)
        w, h, xc, yc, aspect, area = parse_box(box)
        eid = e.get("id", None)
        linking = e.get("linking", None)
        link_count = len(linking) if isinstance(linking, list) else 0

        row = {
            "source_path": source_path,
            "page_id": os.path.basename(source_path),
            "element_id": eid,
            "label": label,
            "text": text,
            "words_text": words_text,
            "box": box,
            "box_w": w,
            "box_h": h,
            "box_xc": xc,
            "box_yc": yc,
            "box_aspect": aspect,
            "box_area": area,
            "page_max_x": max_x,
            "page_max_y": max_y,
            "linking": linking,
            "link_count": link_count,
        }
        rows.append(row)

    return rows


def build_dataframe(input_dir: str) -> pd.DataFrame:
    # Collect *.json files recursively
    patterns = [
        os.path.join(input_dir, "**", "*.json"),
        os.path.join(input_dir, "*.json"),
    ]
    files = set()
    for p in patterns:
        for f in glob(p, recursive=True):
            files.add(os.path.abspath(f))
    files = sorted(list(files))

    all_rows: List[Dict[str, Any]] = []
    for path in files:
        doc = load_json(path)
        if doc is None:
            continue
        rows = build_rows_from_doc(doc, source_path=path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Normalized geometry columns
    def safe_div(a, b):
        try:
            return a / b if (a is not None and b and b != 0) else None
        except Exception:
            return None

    if not df.empty:
        df["x_center_norm"] = df.apply(lambda r: safe_div(r["box_xc"], r["page_max_x"]), axis=1)
        df["y_center_norm"] = df.apply(lambda r: safe_div(r["box_yc"], r["page_max_y"]), axis=1)
        df["width_norm"] = df.apply(lambda r: safe_div(r["box_w"], r["page_max_x"]), axis=1)
        df["height_norm"] = df.apply(lambda r: safe_div(r["box_h"], r["page_max_y"]), axis=1)
        df["area_frac"] = df.apply(
            lambda r: safe_div(r["box_area"], (r["page_max_x"] * r["page_max_y"]) if (r["page_max_x"] and r["page_max_y"]) else None),
            axis=1,
        )

    return df



def main():
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Input directory:  {INPUT_DIR}")

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    df = build_dataframe(INPUT_DIR)
    print(f"Loaded {len(df)} elements from {INPUT_DIR}")
    print("Columns:", list(df.columns))

    if OUT_PARQUET:
        try:
            df.to_parquet(OUT_PARQUET, index=False)
            print(f"Wrote parquet: {OUT_PARQUET}")
        except Exception as e:
            print(f"[WARN] Could not write parquet ({OUT_PARQUET}): {e}")

    if OUT_CSV:
        try:
            df.to_csv(OUT_CSV, index=False)
            print(f"Wrote CSV: {OUT_CSV}")
        except Exception as e:
            print(f"[WARN] Could not write CSV ({OUT_CSV}): {e}")


if __name__ == "__main__":
    main()
