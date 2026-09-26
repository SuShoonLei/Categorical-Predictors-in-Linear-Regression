from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

df["sex_category"] = df["sex"].map(lambda x: "Moonbeam" if x < 0 else "Thunderpaws")

print(df["sex_category"].value_counts())

# Part 1: OLS with a categorical predictor
import statsmodels.formula.api as smf

model = smf.ols("target ~ bmi + sex_category", data=df).fit()
print(model.summary())

print(model.params)
print(model.ssr)

# statsmodels uses Treatment coding and picks the first level alphabetically
# as the reference. Moonbeam comes before Thunderpaws, so Moonbeam is the
# reference level (the summary shows sex_category[T.Thunderpaws]).

# Part 2: same model with an explicit dummy variable and sklearn
from sklearn.linear_model import LinearRegression

df["sex_dummy"] = (df["sex_category"] == "Thunderpaws").astype(int)

X = df[["bmi", "sex_dummy"]]
y = df["target"]
model_dummy = LinearRegression().fit(X, y)

print(model_dummy.intercept_)
print(model_dummy.coef_)

y_pred = model_dummy.predict(X)
sse = ((y - y_pred) ** 2).sum()
print(sse)

print("Part 1 vs Part 2 comparison:")
print("  intercept:", model.params["Intercept"], "vs", model_dummy.intercept_)
print("  bmi coef:", model.params["bmi"], "vs", model_dummy.coef_[0])
print("  categorical/dummy coef:", model.params["sex_category[T.Thunderpaws]"], "vs", model_dummy.coef_[1])
print("  SSE:", model.ssr, "vs", sse)

# Part 3: separate BMI-only regressions by group
thunderpaws = df[df["sex_category"] == "Thunderpaws"]
moonbeam = df[df["sex_category"] != "Thunderpaws"]

total_sse = 0
for name, subset in [("Thunderpaws", thunderpaws), ("Moonbeam", moonbeam)]:
    X_group = subset[["bmi"]]
    y_group = subset["target"]
    model_group = LinearRegression().fit(X_group, y_group)
    y_pred_group = model_group.predict(X_group)
    sse_group = ((y_group - y_pred_group) ** 2).sum()
    total_sse += sse_group
    print(name)
    print("  number of rows:", len(subset))
    print("  intercept:", model_group.intercept_)
    print("  bmi coefficient:", model_group.coef_[0])
    print("  SSE:", sse_group)

print("total SSE across both groups.", total_sse)
