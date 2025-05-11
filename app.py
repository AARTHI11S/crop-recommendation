from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

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
        # Get input values from the form
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)

        # Get the predicted crop from the model
        prediction = model.predict(input_array)[0]

        # Normalize the input and compute distances
        scaled_input = scaler.transform(input_array)
        distances = euclidean_distances(scaled_input, scaled_features)[0]

        # Add the calculated distances to the dataframe
        df['distance'] = distances

        # Sort crops based on distance, excluding the predicted crop
        similar = df[df['label'] != prediction].sort_values(by='distance')

        # Get the top 3 slightly recommended and bottom 3 not recommended crops
        slightly_recommended = similar['label'].head(3).tolist()  # Top 3 similar crops
        not_recommended = similar['label'].tail(3).tolist()  # Last 3 crops

        # Render the template with the results
        return render_template(
            'index.html',
            prediction_text=prediction,
            slightly_recommended=slightly_recommended,
            not_recommended=not_recommended
        )

    except Exception as e:
        return render_template('index.html', error_message="An error occurred. Please check the input and try again.")

if __name__ == '__main__':
    app.run(debug=True)
