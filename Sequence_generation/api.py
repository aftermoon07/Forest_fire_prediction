import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# --- 1. Model Configuration ---
SEQ_LEN = 6      
HORIZONS = 3     
PATCH_H = 13     
PATCH_W = 13     
CHANNELS = 7     

def slice_output_func(x): 
    return x[:, :HORIZONS, :, :, :]

def squeeze_output_func(x): 
    import tensorflow as tf
    return tf.squeeze(x, axis=-1)

CUSTOM_OBJECTS = {
    'slice_output_func': slice_output_func, 
    'squeeze_output_func': squeeze_output_func,
}

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "final_model.h5"))
MODEL = None

# --- 2. Flask Setup ---
app = Flask(__name__)
CORS(app) 

def load_model():
    """Load the trained model into memory when the server starts."""
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Model file not found at {MODEL_PATH}. Prediction endpoint will run in mock/simulation mode.")
        MODEL = None
        return

    print(f"Loading model from: {MODEL_PATH}")
    try:
        import tensorflow as tf
        MODEL = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=CUSTOM_OBJECTS, 
            safe_mode=False
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Failed loading model: {e}")
        MODEL = None
        
load_model()

# --- 3. Health Endpoint ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "mode": "production" if MODEL is not None else "simulation"
    })

# --- 4. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        input_data = data.get('input_tensor')
        
        if MODEL is not None and input_data is not None:
            tensor = np.array(input_data, dtype=np.float32)
            if tensor.ndim == 4:
                tensor = np.expand_dims(tensor, axis=0)
            preds = MODEL.predict(tensor)
            return jsonify({"status": "success", "prediction": preds[0].tolist()})
        else:
            # Fallback mock prediction
            mock_preds = np.random.uniform(0.1, 0.9, size=(HORIZONS, PATCH_H, PATCH_W)).tolist()
            return jsonify({
                "status": "success", 
                "prediction": mock_preds, 
                "simulation_mode": MODEL is None
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
