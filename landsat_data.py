# Import necessary libraries
from pystac_client import Client  # For STAC queries
import planetary_computer         # For signing Landsat data requests
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

def query_landsat_data(bounds, time_window):
    """Query Landsat 8 data using Planetary Computer API."""
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = stac.search(
        bbox=bounds, 
        datetime=time_window,
        collections=["landsat-c2-l2"],
        query={"eo:cloud_cover": {"lt": 20}, "platform": {"in": ["landsat-8"]}},
    )
    items = list(search.get_items())
    print(f'Number of Landsat-8 scenes available: {len(items)}')
    return items

def load_landsat_data(items, bounds):
    """Load Landsat 8 bands into an Xarray dataset, keeping thermal bands separate."""
    resolution = 30  # meters per pixel
    scale = resolution / 111320.0  # degrees per pixel

    try:
        # Load non-thermal bands
        data_non_thermal = stac_load(
            items,
            bands=["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"],
            crs="EPSG:4326",
            resolution=scale,
            chunks={"x": 2048, "y": 2048},
            dtype="uint16",
            patch_url=planetary_computer.sign,
            bbox=bounds
        )

        # Load thermal bands separately
        data_thermal = stac_load(
            items,
            bands=["lwir11"],  # TIRS 1 & TIRS 2
            crs="EPSG:4326",
            resolution=scale,
            chunks={"x": 2048, "y": 2048},
            dtype="uint16",
            patch_url=planetary_computer.sign,
            bbox=bounds
        )
    except Exception as e:
        print(f"Error loading Landsat 8 data: {e}")
        return None, None

    # Apply scaling for non-thermal bands (Reflectance Bands)
    scale1 = 0.0000275 
    offset1 = -0.2 
    data_non_thermal = (data_non_thermal.astype(float) * scale1) + offset1

    # Apply scaling for thermal bands (Surface Temperature in Kelvin)
    scale2 = 0.00341802 
    offset2 = 149.0 
    kelvin_celsius = 273.15  # Convert from Kelvin to Celsius
    data_thermal = ((data_thermal.astype(float) * scale2) + offset2) - kelvin_celsius

    return data_non_thermal.median(dim="time").compute(), data_thermal.median(dim="time").compute()

def save_as_geotiff(filename, median, bounds, bands):
    """Save bands as a GeoTIFF file with float64 precision."""
    if median is None:
        print(f"No data available to save for {filename}.")
        return

    height, width = median.dims["latitude"], median.dims["longitude"]
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

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
    items = query_landsat_data(bounds, time_window)
    if not items:
        print("No Landsat-8 data found for the specified time and region.")
        return
    
    median_non_thermal, median_thermal = load_landsat_data(items, bounds)
    
    save_as_geotiff("Landsat_Non_Thermal.tiff", median_non_thermal, bounds, 
                    ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"])
    save_as_geotiff("Landsat_Thermal.tiff", median_thermal, bounds, ["lwir11"])

if __name__ == "__main__":
    main()
