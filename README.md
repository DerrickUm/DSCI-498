# Harnessing Generative AI and Remote Sensing for Urban Heat Island Analysis

## Project Description
This repository contains all code and supporting scripts used to fuse Sentinel-2 optical imagery, Landsat-8 thermal data, and ground-based UHI traverses into a machine-learning pipeline that:
1. Computes spectral indices and water masks  
2. Learns 32-dim latent features with a shallow TensorFlow/Keras autoencoder  
3. Trains and tunes an XGBoost regressor (via Optuna) to predict fine-scale Urban Heat Island (UHI) index in NYC on July 24, 2021  

### `data/readme_data.txt`
Describes how to download or access the raw GeoTIFF and CSV files, and where to place them under `data/` before running the code.

## Data Sources
- **Ground traverses**: CSV of 11,229 lat/lon points with near-surface air temperature UHI Index (3–4 pm, July 24 2021, Bronx & Manhattan)  
  - `Dataset/Training_data_uhi_index_2025-02-18.csv`  
- **Sentinel-2 optical**: 10 m median mosaic, low-cloud scenes  
- **Landsat-8 thermal (TIRS)**: 100 m Land Surface Temperature  

> See `data/readme_data.txt` for download URLs and placement instructions.

## Requirements
Create a Python 3.9+ environment and install:

nginx
numpy
pandas
scikit-learn
optuna
tensorflow
xgboost
xarray
rioxarray
geopandas
rasterio
pyproj
pystac-client
planetary-computer
odc-stac
stackstac
tqdm
shap
### How to Run
Unzip & navigate
unzip ShortTitle_LnameFname.zip
cd LnameFname
Populate data/
Follow data/readme_data.txt to download:
UHI CSV
Sentinel-2 GeoTIFF
Landsat-8 GeoTIFF
Place them under data/ exactly as named.
Install dependencies
pip install -r requirements.txt
Run the main pipeline
main_submission.ipynb
This will:
Load and preprocess raw data
Compute spectral indices & masks
Train autoencoder and extract latent features
Tune and train XGBoost with Optuna
Output uhi_data.csv, model artifacts, and evaluation metrics
Inspect results
Check output/ (created by main.py) for logs, plots, and the final submission.csv
Review feature importance plots and R² scores printed to console
