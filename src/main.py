from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load the TensorFlow model and the scaler fitted during training
model = tf.keras.models.load_model('my_model.keras')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return "Welcome to the Titanic Survival Predictor API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            pclass = int(data['pclass'])
            sex = 1 if data['sex'] == 'female' else 0
            age = float(data['age'])
            fare = float(data['fare'])

            # Scale the input the same way as in training, then predict
            input_data = scaler.transform(np.array([[pclass, sex, age, fare]]))
            probability = float(model.predict(input_data)[0][0])
            predicted_class = 'Survived' if probability >= 0.5 else 'Did not survive'

            return jsonify({"predicted_class": predicted_class,
                            "survival_probability": round(probability, 3)})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
