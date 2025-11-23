import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle["model"]
model_columns = bundle["columns"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()

        for key, value in input_data.items():
            try:
                input_data[key] = float(value)
            except ValueError:
                pass # Keep as string if it's text (like 'GP' or 'F')

        df = pd.DataFrame([input_data])

        df_encoded = pd.get_dummies(df)

        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df_encoded)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Final Grade (G3): {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)