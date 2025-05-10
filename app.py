from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # Normalize input and compute distances
        scaled_input = scaler.transform(input_array)
        distances = euclidean_distances(scaled_input, scaled_features)[0]

        # Get sorted indices (excluding same prediction)
        df['distance'] = distances
        similar = df[df['label'] != prediction].sort_values(by='distance')

        slightly_recommended = similar['label'].unique()[:3]
        not_recommended = similar['label'].unique()[-3:]

        return render_template(
            'index.html',
            prediction_text=prediction,
            slightly_recommended=slightly_recommended,
            not_recommended=not_recommended
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
