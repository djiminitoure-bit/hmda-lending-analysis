# Saran Toure
# 9/15/2026
# program pulls maryland home-purchase mortgage data from the CFPB HMDA api and compares
# denial rates for white, black, hispanic, and asian applicants before and after controls
#
# how to run:
#   python hmda_analysis.py                  (maryland, 2024)
#   python hmda_analysis.py --year 2023      (different year)
#   python hmda_analysis.py --input my.csv   (use a csv already downloaded)

import argparse
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# constants
datadir = Path("data")
outdir = Path("output")
api = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
groups = ["White", "Black", "Hispanic", "Asian"]  # white is the reference group
usecols = ["county_code", "derived_race", "derived_ethnicity", "action_taken",
           "loan_type", "occupancy_type", "business_or_commercial_purpose",
           "reverse_mortgage", "open-end_line_of_credit", "loan_amount",
           "income", "debt_to_income_ratio", "loan_to_value_ratio",
           "denial_reason-1"]
denialreasons = {"1": "Debt-to-income ratio", "2": "Employment history",
                 "3": "Credit history", "4": "Collateral", "5": "Insufficient cash",
                 "6": "Unverifiable information", "7": "Incomplete application",
                 "8": "Mortgage insurance denied", "9": "Other"}


# download data num 1
# only first-lien, site-built, home-purchase apps where the lender made a decision
def getdata(state, year):
    datadir.mkdir(exist_ok=True)
    path = datadir / f"hmda_{state}_{year}.csv"
    if path.exists():
        return path
    params = {"states": state, "years": year, "actions_taken": "1,2,3",
              "loan_purposes": "1", "lien_statuses": "1",
              "dwelling_categories": "Single Family (1-4 Units):Site-Built"}
    url = f"{api}?{urlencode(params)}"
    print(f"Downloading {url}")
    pd.read_csv(url, usecols=usecols, dtype=str, low_memory=False).to_csv(path, index=False)
    return path


# dti buckets
# hmda gives dti as ranges under 36% and over 49%, exact numbers in between
def dtibucket(x):
    if pd.isna(x) or x in ("Exempt", "NA"):
        return np.nan
    if x in ("<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%"):
        return x
    try:
        v = float(x)
    except ValueError:
        return np.nan
    return "36%-<43%" if v < 43 else "43%-<50%"


# ltv buckets
def ltvbucket(x):
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or v <= 0:
        return np.nan
    return "<=80" if v <= 80 else "80-90" if v <= 90 else "90-95" if v <= 95 else ">95"


# clean data num 2
def cleandata(raw):
    df = raw.copy()
    # keep owner-occupied, drop business loans, reverse mortgages, and credit lines
    df = df[(df["occupancy_type"] == "1")
            & (df["business_or_commercial_purpose"] != "1")
            & (df["reverse_mortgage"] != "1")
            & (df["open-end_line_of_credit"] != "1")]

    # set groups (hispanic = any race, others = non-hispanic)
    hisp = df["derived_ethnicity"] == "Hispanic or Latino"
    nonhisp = df["derived_ethnicity"] == "Not Hispanic or Latino"
    df["group"] = np.select(
        [hisp,
         nonhisp & (df["derived_race"] == "White"),
         nonhisp & (df["derived_race"] == "Black or African American"),
         nonhisp & (df["derived_race"] == "Asian")],
        ["Hispanic", "White", "Black", "Asian"], default="drop")
    df = df[df["group"] != "drop"]

    # variables for the model (income is in thousands)
    df["denied"] = (df["action_taken"] == "3").astype(int)
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df = df[(df["income"] > 0) & (df["loan_amount"] > 0)]
    df["log_income"] = np.log(df["income"] * 1000)
    df["log_loan"] = np.log(df["loan_amount"])
    df["dti"] = df["debt_to_income_ratio"].map(dtibucket)
    df["ltv"] = df["loan_to_value_ratio"].map(ltvbucket)
    df = df.dropna(subset=["dti", "ltv", "county_code"])
    df["group"] = pd.Categorical(df["group"], categories=groups)
    return df


# regression num 3
def fitmodel(df):
    formula = ("denied ~ C(group, Treatment('White')) + log_income + log_loan"
               " + C(dti, Treatment('30%-<36%')) + C(ltv, Treatment('<=80'))"
               " + C(loan_type) + C(county_code)")
    return smf.logit(formula, data=df).fit(disp=False, cov_type="HC1")


# adjusted rates num 4
# pretend every applicant is in each group but keep their own income, loan, dti, ltv,
# loan type, and county, then average the predicted denial chance
def adjustedrates(model, df):
    out = {}
    for g in groups:
        cf = df.copy()
        cf["group"] = pd.Categorical([g] * len(cf), categories=groups)
        out[g] = model.predict(cf).mean()
    return pd.Series(out)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="MD")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--input", type=Path, help="Use an existing HMDA CSV instead of downloading")
    args = ap.parse_args()

    path = args.input or getdata(args.state, args.year)
    df = cleandata(pd.read_csv(path, dtype=str, low_memory=False))
    outdir.mkdir(exist_ok=True)

    # raw vs adjusted gaps
    raw = df.groupby("group", observed=True)["denied"].agg(applications="size", denial_rate="mean")
    model = fitmodel(df)
    adj = adjustedrates(model, df)

    summary = raw.assign(adjusted_rate=adj)
    summary["raw_gap_pp"] = (summary["denial_rate"] - summary.loc["White", "denial_rate"]) * 100
    summary["adjusted_gap_pp"] = (summary["adjusted_rate"] - summary.loc["White", "adjusted_rate"]) * 100
    summary["share_of_gap_unexplained"] = (summary["adjusted_gap_pp"] / summary["raw_gap_pp"]).where(summary.index != "White")

    # odds ratios
    terms = [t for t in model.params.index if t.startswith("C(group")]
    ci = model.conf_int().loc[terms]
    odds = pd.DataFrame({"odds_ratio": np.exp(model.params[terms]),
                         "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                         "p_value": model.pvalues[terms]})
    odds.index = [t.split("T.")[-1].rstrip("]") for t in terms]

    # denial reasons
    denied = df[df["denied"] == 1]
    reasons = (pd.crosstab(denied["denial_reason-1"].map(denialreasons), denied["group"], normalize="columns") * 100).round(1)

    # chart
    plot = summary.drop("White")[["raw_gap_pp", "adjusted_gap_pp"]]
    ax = plot.plot.bar(rot=0, figsize=(7, 4), color=["#9aa5b1", "#1f4e79"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Denial-rate gap vs. White applicants (pp)")
    ax.set_title(f"Home-purchase mortgage denial gaps, {args.state} {args.year}")
    ax.legend(["Raw gap", "After controls"])
    plt.tight_layout()
    plt.savefig(outdir / "denial_gaps.png", dpi=200)

    # write results
    with open(outdir / "results.md", "w") as f:
        f.write(f"# Results: {args.state} {args.year}\n\n")
        f.write(f"Sample: {len(df):,} owner-occupied, first-lien, site-built home-purchase applications with a lender decision.\n\n")
        f.write("## Denial rates and gaps\n\n" + summary.round(3).to_markdown() + "\n\n")
        f.write("## Logit odds ratios (reference: White)\n\n" + odds.round(3).to_markdown() + "\n\n")
        f.write("## Primary denial reason, share of denials (%)\n\n" + reasons.to_markdown() + "\n\n")
        f.write(f"Pseudo R-squared: {model.prsquared:.3f}\n")
    print(summary.round(3).to_string())
    print(f"\nWrote {outdir / 'results.md'} and {outdir / 'denial_gaps.png'}")


if __name__ == "__main__":
    main()
