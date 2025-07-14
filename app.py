import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar modelo y preprocesadores
with open("xgb_titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("sex_encoder.pkl", "rb") as f:
    sex_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            pclass = int(request.form["Pclass"])
            age = float(request.form["Age"])
            sibsp = int(request.form["SibSp"])
            parch = int(request.form["Parch"])
            fare = float(request.form["Fare"])
            embarked = int(request.form["Embarked"])
            deck = int(request.form["Deck"])
            sex = request.form["Sex"]

            # Encode Sex
            sex_encoded = sex_encoder.transform([[sex]])[0]

            # Calcular FamilySize
            family_size = sibsp + parch + 1

            # Escalar
            input_data = np.array([[pclass, age, sibsp, parch, fare, family_size]])
            input_scaled = scaler.transform(input_data)

            # Concatenar features completas
            full_input = np.hstack([
                input_scaled,
                [[embarked]],
                [sex_encoded],
                [[deck]]
            ])

            prediction = model.predict(full_input)[0]
            resultado = "Sobrevivió" if prediction == 1 else "No sobrevivió"

            return render_template("form.html", prediction=resultado)

        except Exception as e:
            return f"Error en entrada: {e}"

    return render_template("form.html", prediction=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)