from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    # Load models from the Models directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
    ridge_model = pickle.load(open(os.path.join(model_dir, 'ridge.pkl'), 'rb'))
    scaler_model = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    ridge_model = None
    scaler_model = None

def calculate_bui(dmc, dc):
    """Calculate Buildup Index (BUI) based on DMC and DC"""
    try:
        if dmc <= 0.4 * dc:
            bui = (0.8 * dmc * dc) / (dmc + 0.4 * dc)
        else:
            bui = dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
        return max(0, bui)
    except Exception as e:
        logger.error(f"Error calculating BUI: {str(e)}")
        return 0

def calculate_fwi(bui, isi):
    """Calculate Fire Weather Index (FWI) based on BUI and ISI"""
    try:
        if bui <= 80:
            fwi = 0.1 * isi * bui
        else:
            fwi = 0.1 * isi * (1000 / (25 + 108.64 / np.exp(0.023 * bui)))
        return max(0, fwi)
    except Exception as e:
        logger.error(f"Error calculating FWI: {str(e)}")
        return 0

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            if ridge_model is None or scaler_model is None:
                raise Exception("Models not loaded properly")

            # Get form data
            Temperature = float(request.form.get('temperature'))
            RH = float(request.form.get('rh'))
            Ws = float(request.form.get('ws'))
            Rain = float(request.form.get('rain'))
            FFMC = float(request.form.get('ffmc'))
            DMC = float(request.form.get('dmc'))
            DC = float(request.form.get('dc'))
            ISI = float(request.form.get('isi'))
            
            # Calculate indices
            BUI = calculate_bui(DMC, DC)
            FWI = calculate_fwi(BUI, ISI)
            
            # Make prediction
            new_data_scaled = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI]])
            result = ridge_model.predict(new_data_scaled)
            
            logger.info(f"Prediction made successfully: {result[0]}")
            
            return render_template('home.html',
                                prediction=result[0],
                                input_data={
                                    'Temperature': Temperature,
                                    'Relative Humidity': RH,
                                    'Wind Speed': Ws,
                                    'Rain': Rain,
                                    'FFMC Index': FFMC,
                                    'DMC Index': DMC,
                                    'DC Index': DC,
                                    'ISI Index': ISI,
                                    'BUI Index': BUI,
                                    'FWI Index': FWI
                                })
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return render_template('home.html', error=str(e))
    return render_template('home.html')

@app.route('/health')
def health_check():
    status = "healthy" if ridge_model is not None and scaler_model is not None else "unhealthy"
    return jsonify({"status": status}), 200 if status == "healthy" else 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('home.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('home.html', error="Internal server error"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)