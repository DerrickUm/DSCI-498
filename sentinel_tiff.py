# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from rasterio.transform import from_bounds
import pystac_client
import planetary_computer
from odc.stac import stac_load

# Define the bounding box (NYC region for UHI data)
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

# Define the time window
time_window = "2021-06-01/2021-09-01"

# Query Sentinel-2 Data
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = stac.search(
    bbox=bounds, 
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)

items = list(search.get_items())
print(f'Number of Sentinel-2 scenes available: {len(items)}')

# Define the pixel resolution and scale for CRS 4326 (Lat/Lon)
resolution = 10  # meters per pixel 
scale = resolution / 111320.0  # degrees per pixel

# Load Sentinel-2 Bands into Xarray
data = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326",
    resolution=scale,
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)

# Compute the median composite to remove cloud contamination
median = data.median(dim="time").compute()

# Function to Compute Spectral Indices
def compute_indices(median):
    """
    Compute various spectral indices to improve UHI modeling.
    """
    # ARVI - Atmospherically Resistant Vegetation Index
    arvi = (median.B08 - (2 * median.B04) + median.B02) / (median.B08 + (2 * median.B04) + median.B02)

    # ABEI - automated built-up extraction index
    abei = (0.312 * median.B01 +
            0.513 * median.B02 -
            0.086 * median.B03 -
            0.441 * median.B04 +
            0.052 * median.B08 -
            0.198 * median.B11 +
            0.278 * median.B12)

    # AWEI - Automated Water Extraction Index (No Shadow Correction)
    awei_nsh = 4 * (median.B03 - median.B11) - (0.25 * median.B08 + 2.75 * median.B12)

    # AWEI - Automated Water Extraction Index (Shadow Correction)
    awei_sh = median.B02 + 2.5 * median.B03 - 1.5 * (median.B08 + median.B11) - 0.25 * median.B12

    # EVI - Enhanced Vegetation Index
    evi = 2.5 * ((median.B08 - median.B04) / (median.B08 + 6 * median.B04 - 7.5 * median.B02 + 1))

    # SAVI - Soil-Adjusted Vegetation Index (L = 0.5)
    savi = ((median.B08 - median.B04) / (median.B08 + median.B04 + 0.5)) * 1.5

    # BI - Bare Soil Index
    bi = ((median.B11 + median.B04) - (median.B08 + median.B02)) / ((median.B11 + median.B04) + (median.B08 + median.B02))

    # Return computed indices as a dictionary
    return {"ARVI": arvi, "ABEI": abei, "AWEI_nsh": awei_nsh, "AWEI_sh": awei_sh, "EVI": evi, "SAVI": savi, "BI": bi}

# Compute Spectral Indices
indices = compute_indices(median)

# Classification Tree Logic
def classify_surface(awei_nsh, awei_sh):
    """
    Classify surfaces using a decision tree based on AWEI indices.
    """
    classification = np.zeros_like(awei_nsh)
    
    # High albedo surfaces
    high_albedo = awei_nsh > 0.5
    classification[high_albedo] = 1  # Class 1: High albedo surfaces
    
    # Shadow/dark surfaces
    shadow_dark = (awei_nsh <= 0.5) & (awei_sh > 0.5)
    classification[shadow_dark] = 2  # Class 2: Shadow/dark surfaces
    
    # Other surfaces
    other_surfaces = (awei_nsh <= 0.5) & (awei_sh <= 0.5)
    classification[other_surfaces] = 3  # Class 3: Other surfaces
    
    return classification

# Apply Classification Tree
classification = classify_surface(indices["AWEI_nsh"], indices["AWEI_sh"])

# Save Output as GeoTIFF
filename = "Sentinel_ABEI_AWEI_Classification.tiff"

# Get image dimensions
height, width = median.dims["latitude"], median.dims["longitude"]

# Define transformation for geo-referencing
gt = from_bounds(lower_left[1], lower_left[0], upper_right[1], upper_right[0], width, height)

# Save bands, indices, and classification to GeoTIFF
with rasterio.open(
    filename, 
    'w',
    driver='GTiff',
    width=width,
    height=height,
    count=11,  # 7 bands + ABEI + AWEI_nsh + AWEI_sh + classification
    crs='EPSG:4326',
    transform=gt,
    compress='lzw',
    dtype='float32'
) as dst:
    dst.write(median.B01, 1)
    dst.write(median.B02, 2)
    dst.write(median.B03, 3)
    dst.write(median.B04, 4)
    dst.write(median.B08, 5)
    dst.write(median.B11, 6)
    dst.write(median.B12, 7)
    dst.write(indices["ARVI"], 8)
    dst.write(indices["ABEI"], 9)
    dst.write(indices["AWEI_nsh"], 10)
    dst.write(indices["AWEI_sh"], 11)
    dst.write(classification, 12)

print("Saved GeoTIFF with Sentinel-2 bands, indices, and classification: Sentinel_ABEI_AWEI_Classification.tiff")
