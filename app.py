import pickle
from flask import Flask, request, render_template, jsonify
import os
import pandas as pd


os.environ['PATH']

app = Flask(__name__)

model_file_name = r'D:\Mobile price prediction\opmodel.pkl'
# Load the trained model from a file
with open(model_file_name, 'rb') as f:
  model = pickle.load(f)

@app.route('/result', methods=['POST'])
def predict():
    # return "succcess"
    # Get the input data from the form
    args = request.form.to_dict()
    data = pd.DataFrame(args, index=[0])

    # Use the model to make a prediction
    prediction = model.predict(data)

    # Return the prediction as a JSON object
    return jsonify({"Prediction" : f"{prediction[0]}"})

@app.route('/')
def index():
  return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


    