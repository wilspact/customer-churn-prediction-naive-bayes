import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\dataset\\project\\bayes theorem pro\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Drop 'customerID' column
df.drop('customerID', axis=1, inplace=True)

# 3. Handle missing values
df.replace(" ", pd.NA, inplace=True)  # Blank space → NA
df.dropna(inplace=True)               # Drop rows with any missing values

# 4. Convert TotalCharges to float
df['TotalCharges'] = df['TotalCharges'].astype(float)

# 5. Encode categorical columns
label_encoder = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoder[column] = le  # save the encoder in dictionary

# 6. Split features and target
x = df.drop("Churn", axis=1)
y = df["Churn"]

# 7. Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 8. Train Naive Bayes model
model = GaussianNB()
model.fit(x_train, y_train)

# 9. Make predictions
y_pred = model.predict(x_test)

# 10. Evaluation
print("Model Evaluation")
print("======================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Predict for a new customer
new_customer = [[0, 0, 1, 0, 5, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 2, 75.35, 370.75]]
prediction = model.predict(new_customer)[0]
churn = label_encoder['Churn'].inverse_transform([prediction])[0]

print("\nNew Customer Prediction")
print("=======================")
print("Prediction (0/1):", prediction)
print("Churn (Yes/No):", churn)
