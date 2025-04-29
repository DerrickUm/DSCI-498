# Import necessary libraries
from pystac_client import Client  # For STAC queries
import planetary_computer         # For signing Sentinel-2 data requests
from odc.stac import stac_load    # For loading STAC items into an Xarray dataset

import rasterio                    # For writing GeoTIFFs
from rasterio.transform import from_bounds

def define_bounds():
    """Define the bounding box and time window."""
    lower_left = (40.75, -74.01)
    upper_right = (40.88, -73.86)
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
    time_window = "2021-06-01/2021-09-01"
    return bounds, time_window

def query_sentinel_data(bounds, time_window):
    """Query Sentinel-2 data using Planetary Computer API."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bounds, 
        datetime=time_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.get_items())
    print(f'Number of Sentinel-2 scenes available: {len(items)}')
    return items

def load_sentinel_data(items, bounds):
    """Load Sentinel-2 bands into an Xarray dataset and resample lower-resolution bands to 10m resolution."""
    resolution = 10  # meters per pixel
    scale = resolution / 111320.0  # degrees per pixel

    try:
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
    except Exception as e:
        print(f"Error loading Sentinel-2 data: {e}")
        return None
    
    # Resample lower-resolution bands (20m and 60m) to 10m resolution using bilinear interpolation
    resample_bands = {"B05", "B06", "B07", "B8A", "B11", "B12", "B01"}  # 20m and 60m bands
    for band in resample_bands:
        if band in data:
            data[band] = data[band].interp_like(data.B02, method="linear")
    
    return data.median(dim="time").compute()

def save_as_geotiff(filename, median, bounds):
    """Save bands as a GeoTIFF file with float64 precision."""
    if median is None:
        print("No data available to save.")
        return

    height, width = median.dims["latitude"], median.dims["longitude"]
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    bands = ["B01", "B02", "B03", "B04", "B08", "B11", "B12"]
    available_bands = [b for b in bands if b in median]

    with rasterio.open(
        filename, 'w', driver='GTiff', width=width, height=height, count=len(available_bands),
        crs='EPSG:4326', transform=transform, compress='lzw', dtype='float64'
    ) as dst:
        for i, band in enumerate(available_bands, start=1):
            dst.write(median[band].astype("float64"), i)

    print(f"Saved GeoTIFF: {filename}")

def main():
    """Main execution function."""
    bounds, time_window = define_bounds()
    items = query_sentinel_data(bounds, time_window)
    if not items:
        print("No Sentinel-2 data found for the specified time and region.")
        return
    
    median = load_sentinel_data(items, bounds)
    save_as_geotiff("Sentinel_Data.tiff", median, bounds)

if __name__ == "__main__":
    main()
