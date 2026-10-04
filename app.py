from pathlib import Path
import math
import pickle

from flask import Flask, render_template, request
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "svm_model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

with (BASE_DIR / "scaler.pkl").open("rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

FEATURES = (
    {"key": "sl", "label": "Sepal length", "minimum": 4.3, "maximum": 7.9},
    {"key": "sw", "label": "Sepal width", "minimum": 2.0, "maximum": 4.4},
    {"key": "pl", "label": "Petal length", "minimum": 1.0, "maximum": 6.9},
    {"key": "pw", "label": "Petal width", "minimum": 0.1, "maximum": 2.5},
)

SPECIES_NAMES = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
}


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, values={})


@app.route("/predict", methods=["POST"])
def predict():
    values = {feature["key"]: request.form.get(feature["key"], "").strip() for feature in FEATURES}
    parsed_values = {}
    errors = []

    for feature in FEATURES:
        raw_value = values[feature["key"]]
        label = feature["label"]

        if not raw_value:
            errors.append(f"Enter a value for {label.lower()}.")
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue

        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number.")
            continue

        if not feature["minimum"] <= value <= feature["maximum"]:
            errors.append(
                f"{label} must be between {feature['minimum']} and "
                f"{feature['maximum']} cm."
            )
            continue

        parsed_values[feature["key"]] = value

    if errors:
        return (
            render_template(
                "index.html",
                features=FEATURES,
                values=values,
                errors=errors,
            ),
            400,
        )

    model_input = pd.DataFrame(
        [[parsed_values[feature["key"]] for feature in FEATURES]],
        columns=scaler.feature_names_in_,
    )
    predicted_species = str(model.predict(scaler.transform(model_input))[0])

    return render_template(
        "index.html",
        features=FEATURES,
        values=values,
        prediction=SPECIES_NAMES.get(predicted_species, predicted_species),
        predicted_label=predicted_species,
    )


if __name__ == "__main__":
    app.run(debug=False)
