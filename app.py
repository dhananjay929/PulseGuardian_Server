from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model_pickle.pkl', 'rb') as f:
    mp = pickle.load(f)
@app.route('/')
def hello():
    return 'Hello, Flask World!'
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('input_data')
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = mp.predict(input_data_reshaped)

        if prediction[0] == 0:
            result = 'The Person does not have a Heart Disease'
        else:
            result = 'The Person has Heart Disease'

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
