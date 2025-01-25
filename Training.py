# %%
pip install pandas numpy scikit-learn xgboost textblob matplotlib seaborn


# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\bjeev\\Downloads\\AI.xlsx"  # Update this path as per your file location
df = pd.read_excel(file_path)


# %%
# Drop unnecessary columns
df = df.drop(columns=["ID", "Name", "Reason for decision"])

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Resume and Job Description Similarity
resume_jd_tfidf = tfidf_vectorizer.fit_transform(df["Resume"] + " " + df["Job Description"])
df["resume_jd_similarity"] = [cosine_similarity(resume_jd_tfidf[i], resume_jd_tfidf[i])[0, 0] for i in range(len(df))]

# Resume and Transcript Similarity
resume_transcript_tfidf = tfidf_vectorizer.fit_transform(df["Resume"] + " " + df["Transcript"])
df["resume_transcript_similarity"] = [cosine_similarity(resume_transcript_tfidf[i], resume_transcript_tfidf[i])[0, 0] for i in range(len(df))]

# Sentiment Analysis on Transcript
df["transcript_sentiment"] = df["Transcript"].apply(lambda text: TextBlob(text).sentiment.polarity)
df["transcript_subjectivity"] = df["Transcript"].apply(lambda text: TextBlob(text).sentiment.subjectivity)

# Length of Transcript
df["transcript_length"] = df["Transcript"].apply(len)

# Encode Target Variable
df["decision_encoded"] = df["decision"].map({"selected": 1, "rejected": 0})


# %%
# Define feature matrix (X) and target vector (y)
X = df[["resume_jd_similarity", "resume_transcript_similarity", "transcript_sentiment",
        "transcript_subjectivity", "transcript_length"]]
y = df["decision_encoded"]

# Split the data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train models and evaluate
results = []
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
    results.append({"Model": name, "Accuracy": accuracy, "ROC AUC": roc_auc})

# Create a DataFrame for results
results_df = pd.DataFrame(results)
print(results_df)


# %%
# Visualization of results
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.show()


# %%
from xgboost import XGBClassifier

# Initialize XGBoost without `use_label_encoder`
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("XGBoost Accuracy:", accuracy)
print("XGBoost ROC AUC:", roc_auc)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier

# Train an XGBoost model
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Display the feature with the highest importance
print("Feature with the highest importance:", feature_importance.iloc[0])

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="Importance", y="Feature", hue="Feature", dodge=False, legend=False)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




# %%


# %%


# %%


# %%


# %%


# %%


# %%
pip install shap matplotlib scikit-learn


# %%


# %%
pip install --upgrade scikit-learn


# %%
import sklearn
print(sklearn.__version__)  


# %%


# %%
import shap
import matplotlib.pyplot as plt
import numpy as np

# Assuming `xgb_model` is trained and `X_train` is available
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_train)

# --- SHAP Beeswarm Plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train)
plt.title("SHAP Beeswarm Plot")
plt.show()

# --- SHAP Waterfall Plots ---
# Low prediction
low_pred_idx = np.argmin(xgb_model.predict_proba(X_test)[:, 1])
shap.waterfall_plot(shap_values[low_pred_idx], max_display=10)
plt.title("Waterfall Plot: Low Prediction")
plt.show()

# High prediction
high_pred_idx = np.argmax(xgb_model.predict_proba(X_test)[:, 1])
shap.waterfall_plot(shap_values[high_pred_idx], max_display=10)
plt.title("Waterfall Plot: High Prediction")
plt.show()

# Medium prediction
medium_pred_idx = np.argsort(xgb_model.predict_proba(X_test)[:, 1])[len(X_test) // 2]
shap.waterfall_plot(shap_values[medium_pred_idx], max_display=10)
plt.title("Waterfall Plot: Medium Prediction")
plt.show()

# --- SHAP Dependence Plots ---
# Dependence plots for top 3 features
top_features = shap_values.abs.mean(0).values.argsort()[-3:][::-1]
for feature_idx in top_features:
    shap.dependence_plot(feature_idx, shap_values.values, X_train)


# %%
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# --- 1D Partial Dependence Plots ---
# Top 3 features for PDP
features_for_pdp = [X_train.columns[idx] for idx in top_features]
fig, ax = plt.subplots(figsize=(15, 8))
PartialDependenceDisplay.from_estimator(
    xgb_model, X_train, features=features_for_pdp, grid_resolution=50, ax=ax
)
plt.show()

# --- 2D Partial Dependence Plot ---
# Interaction of top 2 features
top_two_features = [X_train.columns[idx] for idx in top_features[:2]]
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    xgb_model, X_train, features=top_two_features, kind="both", ax=ax
)
plt.show()


# %%
feature_name = "transcript_sentiment"  # Replace with the name of the feature
shap.plots.scatter(shap_values[:, feature_name], color=None)


# %%
feature_name = "transcript_length"  # Replace with the name of the feature
shap.plots.scatter(shap_values[:, feature_name], color=None)


# %%


# %%


# %%


# %%


# %%


# %%
# Check consistency between SHAP and input data
assert shap_values.shape == X_train.shape, "SHAP values and input data do not align!"


# %%
import pandas as pd

if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train, columns=["Feature1", "Feature2", "Feature3", ...])  # Replace with actual column names


# %%
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_train)


# %%
zero_variance_features = [col for col in X_train.columns if X_train[col].nunique() == 1]
X_train = X_train.drop(columns=zero_variance_features)


# %%
import matplotlib
matplotlib.use("Agg")  # For headless environments, or try "TkAgg", "Qt5Agg", etc.


# %%
print("SHAP Values Shape:", shap_values.values.shape)
print("First SHAP Value:", shap_values[0])
print("Feature Names:", X_train.columns)


# %%
# Save SHAP Beeswarm Plot
shap.summary_plot(shap_values, X_train, show=False)
plt.savefig("shap_beeswarm_plot.png", dpi=300)

# Save SHAP Waterfall Plot
shap.waterfall_plot(shap_values[0], max_display=10)
plt.savefig("shap_waterfall_plot.png", dpi=300)


# %%
print(xgb_model.predict_proba(X_test)[low_pred_idx, 1])  # Print the probability of the positive class


# %%
print(shap_values[low_pred_idx])


# %%
import shap
shap_values = shap.TreeExplainer(xgb_model).shap_values(X_test)


# %%
print(xgb_model.predict_proba(X_test)[low_pred_idx])


# %%
shap.summary_plot(shap_values, X_test)


# %%
pip install ipympl


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



