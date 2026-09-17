from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor



app = Flask(__name__)

# ---------------------------------
# Load Dataset
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR,"Hyderabad_Property_Data.csv"))

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Create Average Price Per Sq Ft
df["AvgPricePerSqFt"] = (
    df["Min_Price_per_Sq_Ft"] +
    df["Max_Price_per_Sq_Ft"]
) / 2

# Encode Locality
encoder = LabelEncoder()
df["Locality_Encoded"] = encoder.fit_transform(df["Locality"])

# Features
X = df[["Locality_Encoded"]]

# Target
y = df["AvgPricePerSqFt"]

# Train Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

# Accuracy
accuracy = round(model.score(X, y) * 100, 2)

print("Model Accuracy:", accuracy, "%")

# ---------------------------------
# Home Page
# ---------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=encoder.classes_,
        accuracy=accuracy
    )

# ---------------------------------
# Prediction
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    locality = request.form["location"]
    area = float(request.form["area"])

    locality_encoded = encoder.transform([locality])[0]

    price_per_sqft = model.predict(
        [[locality_encoded]]
    )[0]

    predicted_price = area * price_per_sqft

    return render_template(
        "index.html",
        locations=encoder.classes_,
        accuracy=accuracy,
        prediction_text=f"Estimated House Price: ₹ {predicted_price:,.0f}"
    )

# ---------------------------------
# Run App
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
