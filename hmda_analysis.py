"""
Mortgage denial gaps in Maryland: an HMDA analysis.

Question: How much of the gap in home-purchase mortgage denial rates across
racial/ethnic groups remains after controlling for the financial and loan
characteristics lenders report under HMDA?

Usage:
    python hmda_analysis.py                  # downloads MD 2024 data from the CFPB
    python hmda_analysis.py --year 2023      # a different year
    python hmda_analysis.py --input my.csv   # use a CSV you already downloaded
"""
import argparse
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

DATA_DIR, OUT_DIR = Path("data"), Path("output")
API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
GROUPS = ["White", "Black", "Hispanic", "Asian"]  # White is the reference group
USECOLS = ["county_code", "derived_race", "derived_ethnicity", "action_taken",
           "loan_type", "occupancy_type", "business_or_commercial_purpose",
           "reverse_mortgage", "open-end_line_of_credit", "loan_amount",
           "income", "debt_to_income_ratio", "loan_to_value_ratio",
           "denial_reason-1"]
DENIAL_REASONS = {"1": "Debt-to-income ratio", "2": "Employment history",
                  "3": "Credit history", "4": "Collateral", "5": "Insufficient cash",
                  "6": "Unverifiable information", "7": "Incomplete application",
                  "8": "Mortgage insurance denied", "9": "Other"}


def download(state: str, year: int) -> Path:
    """Pull first-lien, site-built, home-purchase applications with a decision."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"hmda_{state}_{year}.csv"
    if path.exists():
        return path
    params = {"states": state, "years": year, "actions_taken": "1,2,3",
              "loan_purposes": "1", "lien_statuses": "1",
              "dwelling_categories": "Single Family (1-4 Units):Site-Built"}
    url = f"{API}?{urlencode(params)}"
    print(f"Downloading {url}")
    pd.read_csv(url, usecols=USECOLS, dtype=str, low_memory=False).to_csv(path, index=False)
    return path


def dti_bucket(x: str) -> str:
    """HMDA reports DTI as ranges below 36% and above 49%, exact values in between."""
    if pd.isna(x) or x in ("Exempt", "NA"):
        return np.nan
    if x in ("<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%"):
        return x
    try:
        v = float(x)
    except ValueError:
        return np.nan
    return "36%-<43%" if v < 43 else "43%-<50%"


def ltv_bucket(x) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or v <= 0:
        return np.nan
    return "<=80" if v <= 80 else "80-90" if v <= 90 else "90-95" if v <= 95 else ">95"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df[(df["occupancy_type"] == "1")                      # owner-occupied
            & (df["business_or_commercial_purpose"] != "1")
            & (df["reverse_mortgage"] != "1")
            & (df["open-end_line_of_credit"] != "1")]

    hisp = df["derived_ethnicity"] == "Hispanic or Latino"
    non_hisp = df["derived_ethnicity"] == "Not Hispanic or Latino"
    df["group"] = np.select(
        [hisp,
         non_hisp & (df["derived_race"] == "White"),
         non_hisp & (df["derived_race"] == "Black or African American"),
         non_hisp & (df["derived_race"] == "Asian")],
        ["Hispanic", "White", "Black", "Asian"], default="drop")
    df = df[df["group"] != "drop"]

    df["denied"] = (df["action_taken"] == "3").astype(int)
    df["income"] = pd.to_numeric(df["income"], errors="coerce")        # $ thousands
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df = df[(df["income"] > 0) & (df["loan_amount"] > 0)]
    df["log_income"] = np.log(df["income"] * 1000)
    df["log_loan"] = np.log(df["loan_amount"])
    df["dti"] = df["debt_to_income_ratio"].map(dti_bucket)
    df["ltv"] = df["loan_to_value_ratio"].map(ltv_bucket)
    df = df.dropna(subset=["dti", "ltv", "county_code"])
    df["group"] = pd.Categorical(df["group"], categories=GROUPS)
    return df


def fit(df: pd.DataFrame):
    formula = ("denied ~ C(group, Treatment('White')) + log_income + log_loan"
               " + C(dti, Treatment('30%-<36%')) + C(ltv, Treatment('<=80'))"
               " + C(loan_type) + C(county_code)")
    return smf.logit(formula, data=df).fit(disp=False, cov_type="HC1")


def adjusted_rates(model, df: pd.DataFrame) -> pd.Series:
    """Average predicted denial rate if every applicant were assigned to each group,
    holding their own income, loan, DTI, LTV, loan type, and county fixed."""
    out = {}
    for g in GROUPS:
        cf = df.copy()
        cf["group"] = pd.Categorical([g] * len(cf), categories=GROUPS)
        out[g] = model.predict(cf).mean()
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="MD")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--input", type=Path, help="Use an existing HMDA CSV instead of downloading")
    args = ap.parse_args()

    path = args.input or download(args.state, args.year)
    df = clean(pd.read_csv(path, dtype=str, low_memory=False))
    OUT_DIR.mkdir(exist_ok=True)

    raw = df.groupby("group", observed=True)["denied"].agg(applications="size", denial_rate="mean")
    model = fit(df)
    adj = adjusted_rates(model, df)

    summary = raw.assign(adjusted_rate=adj)
    summary["raw_gap_pp"] = (summary["denial_rate"] - summary.loc["White", "denial_rate"]) * 100
    summary["adjusted_gap_pp"] = (summary["adjusted_rate"] - summary.loc["White", "adjusted_rate"]) * 100
    summary["share_of_gap_unexplained"] = (summary["adjusted_gap_pp"] / summary["raw_gap_pp"]).where(summary.index != "White")

    terms = [t for t in model.params.index if t.startswith("C(group")]
    ci = model.conf_int().loc[terms]
    odds = pd.DataFrame({"odds_ratio": np.exp(model.params[terms]),
                         "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                         "p_value": model.pvalues[terms]})
    odds.index = [t.split("T.")[-1].rstrip("]") for t in terms]

    denied = df[df["denied"] == 1]
    reasons = (pd.crosstab(denied["denial_reason-1"].map(DENIAL_REASONS), denied["group"], normalize="columns") * 100).round(1)

    # Chart: raw vs. adjusted gap relative to White applicants
    plot = summary.drop("White")[["raw_gap_pp", "adjusted_gap_pp"]]
    ax = plot.plot.bar(rot=0, figsize=(7, 4), color=["#9aa5b1", "#1f4e79"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Denial-rate gap vs. White applicants (pp)")
    ax.set_title(f"Home-purchase mortgage denial gaps, {args.state} {args.year}")
    ax.legend(["Raw gap", "After controls"])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "denial_gaps.png", dpi=200)

    with open(OUT_DIR / "results.md", "w") as f:
        f.write(f"# Results: {args.state} {args.year}\n\n")
        f.write(f"Sample: {len(df):,} owner-occupied, first-lien, site-built home-purchase applications with a lender decision.\n\n")
        f.write("## Denial rates and gaps\n\n" + summary.round(3).to_markdown() + "\n\n")
        f.write("## Logit odds ratios (reference: White)\n\n" + odds.round(3).to_markdown() + "\n\n")
        f.write("## Primary denial reason, share of denials (%)\n\n" + reasons.to_markdown() + "\n\n")
        f.write(f"Pseudo R-squared: {model.prsquared:.3f}\n")
    print(summary.round(3).to_string())
    print(f"\nWrote {OUT_DIR / 'results.md'} and {OUT_DIR / 'denial_gaps.png'}")


if __name__ == "__main__":
    main()
