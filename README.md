# hmda-lending-analysis
hmda lending model for maryland home-purchase mortgage applications from the CFPB's HMDA API. It then compares denial rates for white, black, hispanic, and asian applicants.
# Mortgage Denial Gaps in Maryland: An HMDA Analysis

How much of the racial and ethnic gap in mortgage denial rates survives once you account for what lenders actually see on the application?

This project uses public loan-level data reported under the Home Mortgage Disclosure Act (HMDA) to compare home-purchase denial rates for White, Black, Hispanic, and Asian applicants in Maryland, then estimates how much of each gap remains after controlling for income, loan size, debt-to-income ratio, loan-to-value ratio, loan type, and county.

## Data

Application-level records from the [FFIEC/CFPB HMDA Data Browser API](https://ffiec.cfpb.gov/documentation/api/data-browser/). The script downloads them directly.

**Sample:** owner-occupied, first-lien, site-built, home-purchase applications where the lender made a decision (originated, approved but not accepted, or denied). Business-purpose loans, reverse mortgages, and open-end lines of credit are excluded.

**Groups:** Hispanic (any race), and non-Hispanic White, Black, and Asian applicants, using HMDA's derived race and ethnicity fields.

## Method

1. **Raw gaps:** denial rate for each group minus the White denial rate.
2. **Logistic regression:** probability of denial on group, log income, log loan amount, DTI category, LTV category, loan type, and county fixed effects, with robust standard errors.
3. **Adjusted gaps:** the model predicts each applicant's denial probability as if they belonged to each group while holding their own financial profile fixed. The difference in average predictions is the gap the controls don't explain.
4. **Denial reasons:** distribution of lenders' primary stated denial reason by group.

## Run it

```bash
pip install -r requirements.txt
python hmda_analysis.py                 # Maryland, 2024
python hmda_analysis.py --year 2023     # another year
python hmda_analysis.py --state VA      # another state
```

Outputs go to `output/`: `results.md` (tables) and `denial_gaps.png` (chart).

## Findings

*To be added after the full run.*

## Limitations

- **Public HMDA data does not include credit scores**, a major underwriting factor. A remaining gap after controls is a disparity worth explaining, not proof of discrimination.
- Income, loan amount, and some ratios are reported in bins or rounded, which limits precision.
- The sample only covers applications that reached a decision, so it misses applicants who were discouraged from applying or withdrew.

## Next steps

- Add lender fixed effects to separate differences *between* lenders from differences *within* the same lender.
- Compare years (2018 to present) to see whether gaps are narrowing.
- Join census-tract characteristics to examine neighborhood-level patterns.
