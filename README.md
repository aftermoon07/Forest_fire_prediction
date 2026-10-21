# FirePredict: Spatiotemporal Wildfire Risk Forecasting

![Dashboard Preview](https://github.com/user-attachments/assets/411ac497-1ef9-42d2-ab3e-5ca5db3fded3) ![Map Preview](https://github.com/user-attachments/assets/3be94aa0-eae5-4c14-b314-b93b6f1ab261)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com)
[![Architecture](https://img.shields.io/badge/Architecture-CNN--LSTM--UNet-blue)](https://github.com)
[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-61dafb)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%20API-lightgrey)](https://flask.palletsprojects.com/)
[![Datasets](https://img.shields.io/badge/Datasets-ERA5%20%7C%20DEM%20%7C%20LULC%20%7C%20VIIRS-green)](https://github.com)

A deep learning and geospatial intelligence pipeline designed to forecast forest fire risk probability grids for the Uttarakhand region. By integrating dynamic atmospheric variables with static topographical and fuel characteristics, the system generates high-resolution, multi-horizon fire probability risk maps.

---

## 🛠️ Architecture Overview

Wildfire forecasting is framed as a spatiotemporal image-to-image sequence prediction task:
- **Spatial Feature Extraction (CNN):** Extracts localized features from raster layers including terrain elevation, slope, and atmospheric gradients.
- **Temporal Sequence Modeling (LSTM / ConvLSTM):** Captures multi-hour temporal dynamics over historical 6-hour observation windows.
- **Spatial Preservation (U-Net Decoder):** Restores high-resolution spatial details via skip-connections to output precise future $13 \times 13$ risk probability grids across 3 forecast horizons (+1h, +2h, +3h).

---

## 📊 Data Pipeline & Preprocessing

The model ingests 7 input channels comprising both dynamic meteorological and static topographical features:

1. **ERA5 Atmospheric Reanalysis:** Hourly 2m temperature (`T2M`), 2m dew point (`D2M`), total precipitation (`TP`), and 10m wind vectors (`U10`, `V10`).
2. **DEM (Digital Elevation Model):** Provides topographic elevation and slope angles.
3. **LULC (Land Use / Land Cover):** Categorized fuel classification (dense forest, shrubland, agricultural, water bodies) converted to binary spatial rasters.
4. **VIIRS Active Fire Data:** Used as ground-truth spatial fire occurrence labels.

### Sliding Window Sequence Generator
- **Input Window:** 6 consecutive hours of geospatial rasters ($T_{-5}$ to $T_0$).
- **Output Horizon:** 3 future hours ($T_{+1}, T_{+2}, T_{+3}$) of fire risk probability grids.
- **Patch Resolution:** $13 \times 13$ localized geospatial patches.

---

## 🖥️ System Modules

### 1. Geospatial Analysis Dashboard (React + Vite)
- Interactive upload interface for 7 geospatial channels (32 sequence files).
- Multi-horizon tab switching (Horizon 1, Horizon 2, Horizon 3).
- Spatial risk cell inspector with coordinate resolution and probability metrics.
- Built-in simulation fallback mode for instant demonstration when running without local weights.

### 2. High-Performance Flask API (`backend_api.py`)
- REST endpoint `/api/predict` delivering real-time tensor inference and date mapping.
- Health inspection endpoint `/health`.
- Cross-platform configuration with dynamic `MODEL_PATH` environment variables.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Node.js**: v18+ and `npm`
- **Python**: 3.9+ with `pip`

### 2. Setup Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask API server
python UI_page/backend_api.py
```
Backend will start on `http://localhost:5000`.

### 3. Setup Frontend Dashboard

```bash
# Navigate to UI directory
cd UI_page

# Install dependencies
npm install

# Start development server
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` or set the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port for Flask API service |
| `MODEL_PATH` | `./final_model.h5` | Path to trained Keras/TensorFlow model weights |
| `FLASK_ENV` | `development` | Flask runtime environment (`development` / `production`) |

---

## 📂 Repository Structure

```text
├── Data_preprocessing/      # Tabular to binary raster conversion & resamplers
│   ├── Alignment/           # Spatial alignment across DEM, ERA5, LULC, VIIRS
│   ├── data_cleaning/       # ERA5 and VIIRS fire data cleaning scripts
│   ├── dem_lulc_extent/     # Geospatial extent alignment
│   └── Nearest_neighbourLULC/# Nearest neighbor interpolation
├── LULC_categorical/        # Land use categorization logic
├── Plotting/                # Correlation and comparison visualizations
├── Sequence_generation/     # Sliding window sequence generators and model training
│   ├── api.py               # Standalone prediction API service
│   ├── main.py              # Training pipeline execution
│   └── model_cls.ipynb      # Classification model notebook
├── Time_bands/              # Time-series normalization notebooks
├── UI_page/                 # React + Vite interactive dashboard
│   ├── App.jsx              # Main dashboard component
│   ├── backend_api.py       # Flask backend API
│   ├── package.json         # Frontend configuration
│   └── vite.config.js       # Vite bundler configuration
├── requirements.txt         # Python project dependencies
├── package.json             # Root workspace management
└── README.md                # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
