# Maryland Mortgage Denial Gaps (HMDA)

Black and Hispanic applicants get denied for mortgages more often than White applicants. What I wanted to find out is how much of that gap is left once you account for the things lenders actually look at, like income and debt.

I built this as a personal project to get hands-on experience with a large public dataset and regression analysis. It uses Maryland home-purchase mortgage applications from the CFPB's HMDA data.

**Status:** code is done, results coming soon.

## Data

The script pulls loan-level data straight from the [CFPB's HMDA Data Browser API](https://ffiec.cfpb.gov/documentation/api/data-browser/), so you don't have to download anything by hand.

I kept owner-occupied, first-lien, home-purchase applications where the lender made a decision (approved or denied). I dropped business loans, reverse mortgages, and home equity lines of credit since they work differently.

I compare four groups: White, Black, Asian, and Hispanic applicants.

## What the script does

1. Calculates the denial rate for each group and how far each is from the White denial rate (the "raw gap").
2. Runs a logistic regression predicting whether an application was denied, controlling for income, loan amount, debt-to-income ratio, loan-to-value ratio, loan type, and county.
3. Uses that model to estimate what each group's denial rate would be if applicants kept their own financial profile. The gap that's left is the part the controls can't explain (the "adjusted gap").
4. Breaks down the main reason lenders gave for denials in each group.

## How to run it

```bash
pip install -r requirements.txt
python hmda_analysis.py                 # Maryland, 2024
python hmda_analysis.py --year 2023     # a different year
python hmda_analysis.py --state VA      # a different state
```

Results save to the `output/` folder as a table (`results.md`) and a chart (`denial_gaps.png`).

## Limitations

- HMDA doesn't include credit scores, which matter a lot in lending decisions. So a gap that's left over after my controls shows a disparity, but it doesn't prove discrimination.
- Some variables, like income and loan amount, are rounded or reported in ranges, which makes the estimates less precise.
- The data only covers people who applied and got a decision. It misses anyone who withdrew or never applied at all.

## What I'd add next

- Lender fixed effects, to see whether gaps come from differences between lenders or within the same lender.
- Multiple years, to see if the gaps are shrinking over time.
