How to Run

This project uses a Python virtual environment to keep dependencies isolated.

bash
# 1. Create and activate a virtual environment (from inside this folder)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install pandas scikit-learn statsmodels

# 3. Run the program
python3 linear_regression_with_category.py

Every time you open a new terminal session, re-run source venv/bin/activate before running the script.

Description of Implementation

The program uses the scikit-learn diabetes dataset (load_diabetes(as_frame=True)). The dataset's sex column is a pre-centered numerical variable, so it is first converted back into a readable categorical variable, sex_category, using:

python
df["sex_category"] = df["sex"].apply(
    lambda x: "Moonbeam" if x < 0 else "Thunderpaws"
)

This produces 235 "Moonbeam" observations and 207 "Thunderpaws" observations (442 total). The program then compares three ways of incorporating this categorical predictor into a regression of target on bmi and sex_category.

Part 1 — Automatic dummy coding (statsmodels): Fits target ~ bmi + sex_category using the statsmodels formula interface, which automatically dummy-codes the categorical variable. statsmodels selects a reference level alphabetically, so "Moonbeam" is the reference (coded 0) and the model includes a coefficient for sex_category[T.Thunderpaws] (coded 1).

Part 2 — Manual dummy coding (scikit-learn): Since scikit-learn's LinearRegression cannot handle categorical variables directly, a dummy column is created manually: sex_dummy = 1 if Thunderpaws, 0 if Moonbeam (matching statsmodels' choice of reference level). This dummy column and bmi are then used as ordinary numeric predictors in LinearRegression.

Part 3 — Separate regression models per group: The dataset is split into two subsets by sex_category, and a separate univariate regression (target ~ bmi) is fit independently within each subset, allowing both the intercept and the BMI slope to differ by group.

SSE (sum of squared errors) is reported for all three parts — directly from model.ssr in Part 1, and computed manually (((y - y_pred) ** 2).sum()) in Parts 2 and 3.

Results
Part 1: Automatic Dummy Coding (statsmodels)
Term	Coefficient
Intercept	152.7628
BMI	950.6781
sex_category[T.Thunderpaws]	-1.3438 (p = 0.823, not significant)

SSE: 1,719,384.61

Part 2: Manual Dummy Coding (scikit-learn)
Term	Coefficient
Intercept	152.7628
BMI	950.6781
Dummy (sex_dummy)	-1.3438

SSE: 1,719,384.61

Part 3: Separate Regression Models
Group	n	Intercept	BMI slope	SSE
Thunderpaws	207	150.6339	1126.4042	729,696.39
Moonbeam	235	152.2464	819.4475	966,806.81

Total SSE across both groups: 1,696,503.20

Answers to Questions
Part 2 Questions

1–2. Compare the manually coded regression with the model from Part 1. Are the coefficients approximately the same?

Yes. The intercept (152.7628), BMI coefficient (950.6781), and categorical/dummy coefficient (-1.3438) are identical between the two approaches, matching to roughly 10 decimal places (the tiny remaining difference, e.g. 950.6781385437343 vs. 950.6781385437346, is ordinary floating-point rounding, not a real difference).

3. Is the SSE approximately the same?

Yes. Both approaches give an SSE of 1,719,384.6087101665 — identical.

4. Why should the two approaches produce the same predictions even though the dummy variable was created manually in Part 2?

Both approaches ultimately build the exact same design matrix: the same bmi column and the same 0/1 encoding of the category (Thunderpaws = 1, Moonbeam = 0 in both cases). Whether that 0/1 column is generated internally by statsmodels' formula parser or explicitly by .astype(int) in Part 2 doesn't matter to the underlying math — ordinary least squares has a unique closed-form solution for a given design matrix and target vector. Since the inputs to the regression are numerically identical in both cases, the fitted coefficients, predictions, and SSE must be identical too.

Part 3 Questions

1. Are the BMI slopes for the two groups the same?

No. The Thunderpaws slope (1126.40) is noticeably steeper than the Moonbeam slope (819.45) — a difference of about 307, or roughly 37% higher for Thunderpaws. This is a meaningful difference, not just noise.

2. Are the intercepts the same?

They are much closer than the slopes: 150.63 for Thunderpaws vs. 152.25 for Moonbeam, a difference of only about 1.6. This is consistent with Part 1's near-zero, statistically insignificant category coefficient (-1.34, p = 0.823), which only captures a shift in intercept and found no meaningful difference.

3. How do the two regression lines differ?

The two lines start from nearly the same point near bmi = 0 (their intercepts are close) but diverge in steepness. Because the Thunderpaws line has a much larger slope, the two lines are not parallel — they separate more and more as bmi moves away from its (centered) average in either direction.

4. How does this approach differ from Parts 1 and 2?

Parts 1 and 2 fit a single shared BMI slope across both groups, and only allow the intercept to shift between categories (an additive, "parallel lines" model). This assumes the relationship between BMI and target is the same for both groups. Part 3 makes no such assumption — it lets both the slope and the intercept vary freely per group, which is what reveals the slope difference that Parts 1 and 2 could not detect, since they structurally cannot represent a different slope per category.

5. What is the tradeoff between one regression model with a categorical predictor and fitting separate regression models for each group?

A single shared model (Parts 1–2) is simpler to report and interpret, and it estimates the BMI slope using the full sample (442 rows), which generally makes that estimate more stable and precise. However, it can mask real differences in how a predictor relates to the outcome across groups — which happened here, since the true BMI slopes differ by around 300 between groups. Separate models (Part 3) can capture that kind of difference and achieve a lower total SSE (1,696,503 vs. 1,719,385, about 1.3% lower), but each group's slope and intercept are estimated from a smaller subsample (207 or 235 rows instead of 442), making each estimate noisier and more sensitive to sampling variation, especially for smaller groups. Given that Part 1's category coefficient was not statistically significant (p = 0.823) while Part 3's slopes differ substantially, this dataset suggests that if there is a real effect of sex_category here, it looks more like a difference in the BMI–target relationship (interaction/slope effect) than a simple additive shift — something only the separate-models approach (or an explicit interaction term) can capture.

AI Usage

I used Claude.ai throughout this assignment for: better understanding of the assignment topic and any confusion I have.

