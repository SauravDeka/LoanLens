from pathlib import Path
import pickle
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "loan_approval_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

print(df.isnull().sum())

# Label Encoding
education_encoder = LabelEncoder()
df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])

loan_encoder = LabelEncoder()
df["Loan_Approved"] = loan_encoder.fit_transform(df["Loan_Approved"])

# One-Hot Encoding
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cols),
    index=df.index
)

# Combine One-Hot Encoded Columns
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# Remove Applicant ID
df = df.drop("Applicant_ID", axis=1)

# Add or Transform Features
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

# Separate Features and Target
X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================
model = RandomForestClassifier(n_estimators=500, random_state=42)

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Calculate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ============================================================
# SAVE TRAINED MODEL
# ============================================================
MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model saved to:", MODEL_PATH)

# Save Scaler
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

with open(SCALER_PATH, "wb") as file:
    pickle.dump(scaler, file)

print("Scaler saved to:", SCALER_PATH)

# Save One-Hot Encoder
OHE_PATH = BASE_DIR / "model" / "one_hot_encoder.pkl"

with open(OHE_PATH, "wb") as file:
    pickle.dump(ohe, file)

print("One-Hot Encoder saved to:", OHE_PATH)

# Save Education Label Encoder
EDUCATION_ENCODER_PATH = BASE_DIR / "model" / "education_encoder.pkl"

with open(EDUCATION_ENCODER_PATH, "wb") as file:
    pickle.dump(education_encoder, file)

print("Education Encoder saved to:", EDUCATION_ENCODER_PATH)


# Save Loan Label Encoder
LOAN_ENCODER_PATH = BASE_DIR / "model" / "loan_encoder.pkl"

with open(LOAN_ENCODER_PATH, "wb") as file:
    pickle.dump(loan_encoder, file)

print("Loan Encoder saved to:", LOAN_ENCODER_PATH)