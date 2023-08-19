from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved model
try:
    with open('model_pickle.pkl', 'rb') as f:
        mp = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found.")
    exit(1)
except Exception as e:
    print("Error loading model:", e)
    exit(1)

@app.route('/')
def hello():
    return 'Hello, Its D.J'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('input_data')

        if input_data is None:
            raise ValueError("Input data not provided.")

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = mp.predict(input_data_reshaped)

        if prediction[0] == 0:
            result = 0
        else:
            result = 1

        return jsonify({'result': result})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400  # Bad Request
    except KeyError as ke:
        return jsonify({'error': 'Invalid JSON field: ' + str(ke)}), 400
    except Exception as e:
        # Log the error for debugging purposes
        print("An error occurred:", e)
        return jsonify({'error': 'An internal error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)