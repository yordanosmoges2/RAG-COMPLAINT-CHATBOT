from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


RAW_PATH_DEFAULT = Path("data/raw/cfpb_complaints.csv")
OUT_PATH_DEFAULT = Path("data/filtered_complaints.csv")
PLOTS_DIR_DEFAULT = Path("data/processed/plots")


# These are the product categories referenced in the PDF narrative/business objective :contentReference[oaicite:6]{index=6}
# CFPB product names vary across years; we match via keywords.
PRODUCT_KEYWORDS = {
    "Credit Cards": ["credit card"],
    "Personal Loans": ["personal loan", "payday loan", "loan"],  # keep "personal loan" first
    "Savings Accounts": ["savings account", "savings"],
    "Money Transfers": ["money transfer", "money transfers", "remittance", "money service", "wire transfer"],
}


def _normalize_text(s: str) -> str:
    s = s.lower()
    # remove common boilerplate lines like "I am writing to file a complaint..."
    s = re.sub(r"\bi am writing to file a complaint\b.*", "", s, flags=re.IGNORECASE)
    # keep letters/numbers/basic punctuation, collapse whitespace
    s = re.sub(r"[^a-z0-9\s\.\,\;\:\?\!\-\'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"None of these columns found: {list(candidates)}. Available: {list(df.columns)}")


def map_product_category(product_value: str) -> Optional[str]:
    """Map raw CFPB 'Product' to one of the project categories."""
    if not isinstance(product_value, str) or not product_value.strip():
        return None
    p = product_value.lower()

    for cat, keys in PRODUCT_KEYWORDS.items():
        # require strong match for personal loans to avoid pulling every loan type
        if cat == "Personal Loans":
            if "personal loan" in p:
                return cat
            continue

        if any(k in p for k in keys):
            return cat

    # If you want to be more permissive:
    # if "loan" in p: return "Personal Loans"
    return None


def main(
    raw_path: Path = RAW_PATH_DEFAULT,
    out_path: Path = OUT_PATH_DEFAULT,
    plots_dir: Path = PLOTS_DIR_DEFAULT,
) -> None:
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    plots_dir = Path(plots_dir)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Put your CFPB dataset there (CSV)."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    product_col = _pick_column(df, ["Product", "product"])
    narrative_col = _pick_column(df, ["Consumer complaint narrative", "consumer_complaint_narrative", "narrative"])
    complaint_id_col = None
    try:
        complaint_id_col = _pick_column(df, ["Complaint ID", "complaint_id", "ComplaintID"])
    except KeyError:
        # Not fatal; we can proceed without it
        pass

    # ---- EDA 1: Complaints per product ----
    product_counts = df[product_col].fillna("(missing)").value_counts().head(30)
    plt.figure()
    product_counts.plot(kind="bar")
    plt.title("Top 30 Product values (raw)")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_product_distribution_top30.png", dpi=150)
    plt.close()

    # ---- EDA 2: Missing narratives ----
    missing_narratives = df[narrative_col].isna().sum()
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print(f"Missing narratives: {missing_narratives} ({missing_narratives/total_rows:.2%})")

    # ---- EDA 3: Word count distribution ----
    narrative_series = df[narrative_col].fillna("").astype(str)
    word_counts = narrative_series.apply(lambda x: len(x.split()))
    plt.figure()
    plt.hist(word_counts, bins=50)
    plt.title("Narrative word count distribution (raw)")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_narrative_wordcount_hist.png", dpi=150)
    plt.close()

    # ---- Filter to required product categories + remove empty narratives ----
    df["product_category"] = df[product_col].apply(map_product_category)
    df = df[df["product_category"].notna()].copy()

    # drop empty narrative
    df[narrative_col] = df[narrative_col].fillna("").astype(str)
    df = df[df[narrative_col].str.strip() != ""].copy()

    # ---- Clean narrative text ----
    df["clean_narrative"] = df[narrative_col].apply(_normalize_text)

    # after cleaning, drop any now-empty
    df = df[df["clean_narrative"].str.strip() != ""].copy()

    # minimal output columns (keep useful metadata if present)
    keep_cols = ["product_category", "clean_narrative", product_col]
    if complaint_id_col:
        keep_cols.insert(0, complaint_id_col)

    # keep other helpful fields if exist
    for extra in ["Issue", "Sub-issue", "Company", "State", "Date received"]:
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)

    df_out = df[keep_cols].rename(
        columns={
            product_col: "product",
            "Issue": "issue",
            "Sub-issue": "sub_issue",
            "Company": "company",
            "State": "state",
            "Date received": "date_received",
            complaint_id_col or "": "complaint_id",
        }
    )

    # If we didn't have complaint_id, create one
    if "complaint_id" not in df_out.columns:
        df_out.insert(0, "complaint_id", range(1, len(df_out) + 1))

    df_out.to_csv(out_path, index=False)
    print(f"Saved cleaned + filtered dataset to: {out_path}")
    print(df_out["product_category"].value_counts())


if __name__ == "__main__":
    main()
