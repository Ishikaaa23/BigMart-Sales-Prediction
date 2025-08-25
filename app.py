from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    try:
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        # Create the input array
        X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                       outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

        # Load the scaler and model
        scaler_path = r'C:\Users\Ishika\Desktop\projects\BigMart sales prediction (using LR)\models\sc.sav'
        sc = joblib.load(scaler_path)

        # Scale the input features
        X_std = sc.transform(X)

        model_path = r'C:\Users\Ishika\Desktop\projects\BigMart sales prediction (using LR)\models\rf.sav'
        model = joblib.load(model_path)

        # Make the prediction using the model
        Y_pred = model.predict(X_std)

        return jsonify({'Prediction': float(Y_pred[0])})  # Ensure to return the first element

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)
