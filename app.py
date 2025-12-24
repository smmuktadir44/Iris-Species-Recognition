import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="Template")
model = pickle.load(open("Model/iris.pkl","rb"))

@app.route('/')
def index():
    return render_template('iris_species_identifier.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])

        probabilities = model.predict_proba(features)[0]
        prediction_idx = np.argmax(probabilities)

        target_names = ['Setosa', 'Versicolor', 'Virginica']

        return jsonify({
            "species": target_names[prediction_idx],
            "confidence": round(probabilities[prediction_idx] * 100, 1),
            "probabilities": {
                "setosa": round(probabilities[0] * 100, 1),
                "versicolor": round(probabilities[1] * 100, 1),
                "virginica": round(probabilities[2] * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
