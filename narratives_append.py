import json
from pathlib import Path

import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path("Vensim_Models")
JSONLD_DIR = BASE_DIR / "JSONLD-Files"
BACKGROUND_DIR = BASE_DIR / "Background_Information"
EXCEL_PATH = BACKGROUND_DIR / "Models_Info.xlsx"

OUTPUT_DIR = BASE_DIR / "JSONLD-Narratives"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================
def normalize_model_name(name: str) -> str:
    """
    Normalize model names so these can match:
    - kissafrog.mdl
    - kissafrog.jsonld
    - KissaFrog.MDL
    """
    if pd.isna(name):
        return ""

    name = str(name).strip().lower()

    for ext in [".mdl", ".xmile", ".jsonld", ".json", ".xml"]:
        if name.endswith(ext):
            name = name[: -len(ext)]

    return name


def load_model_info(excel_path: Path) -> dict:
    """
    Load the Excel file and return:
    {
        "kissafrog": "background text...",
        ...
    }
    Column matching is case-insensitive.
    """
    df = pd.read_excel(excel_path)

    col_map = {str(c).strip().lower(): c for c in df.columns}

    required_cols = ["model_name", "information"]
    missing = [c for c in required_cols if c not in col_map]
    if missing:
        raise ValueError(
            f"Missing required columns in Excel: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    model_col = col_map["model_name"]
    info_col = col_map["information"]

    model_info_map = {}

    for _, row in df.iterrows():
        model_name = normalize_model_name(row[model_col])
        information = "" if pd.isna(row[info_col]) else str(row[info_col]).strip()

        if model_name:
            model_info_map[model_name] = information

    return model_info_map


def append_information_to_jsonld(jsonld_data: dict, information_text: str) -> dict:
    """
    Insert backgroundInformation at the beginning of the JSON-LD object,
    keeping the rest of the original structure unchanged.
    """
    enriched = {
        "backgroundInformation": information_text
    }
    enriched.update(jsonld_data)
    return enriched


# =========================
# MAIN
# =========================
def enrich_jsonld_files():
    if not JSONLD_DIR.exists():
        raise FileNotFoundError(f"JSON-LD folder not found: {JSONLD_DIR}")

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    model_info_map = load_model_info(EXCEL_PATH)

    jsonld_files = sorted(JSONLD_DIR.glob("*.jsonld"))
    if not jsonld_files:
        raise RuntimeError(f"No .jsonld files found in: {JSONLD_DIR}")

    matched = 0
    unmatched = []
    empty_information = []

    for jsonld_file in jsonld_files:
        model_base = normalize_model_name(jsonld_file.name)

        with open(jsonld_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if model_base not in model_info_map:
            unmatched.append(jsonld_file.name)
            information_text = ""
        else:
            information_text = model_info_map[model_base]
            matched += 1
            if not information_text.strip():
                empty_information.append(jsonld_file.name)

        enriched_data = append_information_to_jsonld(data, information_text)

        output_path = OUTPUT_DIR / jsonld_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, ensure_ascii=False, separators=(",", ":"))

    print("=== PROCESS FINISHED ===")
    print(f"JSON-LD files processed: {len(jsonld_files)}")
    print(f"Matched with Excel: {matched}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Empty information: {len(empty_information)}")

    if unmatched:
        print("\nFiles without match in Excel:")
        for name in unmatched:
            print(f" - {name}")

    if empty_information:
        print("\nFiles matched but with empty information:")
        for name in empty_information:
            print(f" - {name}")


if __name__ == "__main__":
    enrich_jsonld_files()