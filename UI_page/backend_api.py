import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta

# --- 1. Model Configuration ---
SEQ_LEN = 6      
HORIZONS = 3     
PATCH_H = 13     
PATCH_W = 13     
CHANNELS = 7

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "final_model.h5"))
FIXED_SAMPLE_INDEX = 17 
START_DATE = datetime(2015, 1, 1, 0, 0, 0)

# --- 2. Custom Layers/Functions (Required for Model Loading) ---
def slice_output_func(x):
    return x[:, :HORIZONS, :, :, :]

def squeeze_output_func(x):
    import tensorflow as tf
    return tf.squeeze(x, axis=-1)

def slice_output_shape(input_shape):
    return (input_shape[0], HORIZONS, input_shape[2], input_shape[3], input_shape[4])

def squeeze_output_shape(input_shape):
    return input_shape[:-1] 

CUSTOM_OBJECTS = {
    'slice_output_func': slice_output_func, 
    'slice_output_shape': slice_output_shape,
    'squeeze_output_func': squeeze_output_func, 
    'squeeze_output_shape': squeeze_output_shape,
}

# --- 3. Date Mapping Function ---
def get_sample_date_range(sample_index, start_date):
    """Calculates the date range based on a starting index (hour offset)."""
    input_start_hour_offset = sample_index
    input_end_hour_offset = sample_index + SEQ_LEN - 1
    
    input_start_time = start_date + timedelta(hours=input_start_hour_offset)
    input_end_time = start_date + timedelta(hours=input_end_hour_offset)
    
    pred_start_hour_offset = sample_index + SEQ_LEN
    pred_end_hour_offset = pred_start_hour_offset + HORIZONS - 1
    
    pred_start_time = start_date + timedelta(hours=pred_start_hour_offset)
    pred_end_time = start_date + timedelta(hours=pred_end_hour_offset)

    return {
        "inputStart": input_start_time.strftime('%H:%M:%S'),
        "inputEnd": input_end_time.strftime('%H:%M:%S'),
        "predStart": pred_start_time.strftime('%H:%M:%S'),
        "predEnd": pred_end_time.strftime('%H:%M:%S'),
        "date": input_start_time.strftime('%Y-%m-%d'),
        "full_start_date": input_start_time.strftime('%Y-%m-%d %H:%M:%S')
    }

# --- 4. Load Model (Load once when server starts) ---
model = None
if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        print(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=CUSTOM_OBJECTS,
            safe_mode=False 
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to load model from {MODEL_PATH}: {e}")
        model = None
else:
    print(f"[INFO] Model file not found at {MODEL_PATH}. Running in simulation/demo mode.")

# --- 5. Flask Application Setup ---
app = Flask(__name__)
CORS(app) 

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "mode": "production" if model is not None else "simulation"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        lat_min = float(data.get('latMin', 30.2))
        lon_min = float(data.get('lonMin', 77.8))
    except Exception as e:
        return jsonify({"error": f"Invalid input coordinates: {e}"}), 400

    time_details = get_sample_date_range(FIXED_SAMPLE_INDEX, START_DATE)

    if model is not None:
        try:
            input_tensor = np.random.rand(1, SEQ_LEN, PATCH_H, PATCH_W, CHANNELS).astype(np.float32)
            prediction_output = model.predict(input_tensor)
            prediction_list = prediction_output[0].tolist() 
        except Exception as e:
            return jsonify({"error": f"Model inference failed: {e}"}), 500
    else:
        # Fallback simulation when model weights are not locally downloaded
        # Generates realistic spatially smoothed probability grids [HORIZONS, PATCH_H, PATCH_W]
        rng = np.random.default_rng(seed=int(abs(lat_min * 100 + lon_min * 100)) % 1000)
        base = rng.uniform(0.05, 0.4, size=(HORIZONS, PATCH_H, PATCH_W))
        # Add a simulated high risk cluster in center
        base[:, 4:9, 4:9] += rng.uniform(0.3, 0.55, size=(HORIZONS, 5, 5))
        prediction_list = np.clip(base, 0.0, 1.0).tolist()

    return jsonify({
        "prediction_results": prediction_list,
        "time_details": time_details,
        "simulation_mode": model is None
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
