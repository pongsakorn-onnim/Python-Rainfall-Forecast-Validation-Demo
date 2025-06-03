import pathlib
import geopandas
import rasterio
import numpy as np
from shapely.geometry import Polygon
from rasterio.transform import from_bounds
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Configuration Constants (should align with config_demo.json) ---
DEMO_PROJECT_ROOT = pathlib.Path(__file__).parent.parent # Assumes script is in demo_root/scripts/
CRS = "EPSG:32647"
NODATA_VALUE = -999.0
RASTER_WIDTH = 10  # pixels
RASTER_HEIGHT = 10 # pixels

# Define a sample extent (e.g., 1km x 1km in UTM Zone 47N)
# These values should result in a pixel size of 100m x 100m for a 10x10 raster
MIN_X, MIN_Y = 600000, 1500000
MAX_X, MAX_Y = MIN_X + (RASTER_WIDTH * 100), MIN_Y + (RASTER_HEIGHT * 100) # 1km x 1km area for 100m pixels
RASTER_BOUNDS = (MIN_X, MIN_Y, MAX_X, MAX_Y)
RASTER_TRANSFORM = from_bounds(MIN_X, MIN_Y, MAX_X, MAX_Y, RASTER_WIDTH, RASTER_HEIGHT)

# Shapefile config
SHAPEFILE_DIR = DEMO_PROJECT_ROOT / "data" / "shapefiles_demo"
SHAPEFILE_NAME = "sample_boundary.shp"
AREA_ID_COL = "AREA_ID"
AREA_NAME_COL = "AREA_NAME"

# Iteration config (from config_demo.json)
YEARS_TO_PROCESS = ["2022"]
MONTHS_TO_PROCESS = ["01", "02"]  # Initialization months
LEAD_TIME_OFFSETS_MONTHS = [0, 1, 2, 3]

# Forecast model config
MODEL_A_DATA_FOLDER_TEMPLATE = DEMO_PROJECT_ROOT / "data" / "forecast_generic_model_A" / "{year}_{month}"
MODEL_A_FILENAME_PATTERN = "model_A_fcst_{target_ym}.tif"

MODEL_B_DATA_FOLDER_TEMPLATE = DEMO_PROJECT_ROOT / "data" / "forecast_generic_model_B" / "{year}_{month}"
MODEL_B_FILENAME_PATTERN = "model_B_fcst_{target_ym}.tif"

# Observed data config
OBSERVED_DATA_FOLDER = DEMO_PROJECT_ROOT / "data" / "observed_rainfall"
OBSERVED_FILENAME_PATTERN = "observed_rain_{year}{month}.tif"

def create_sample_shapefile():
    """Creates a sample shapefile with three polygonal areas."""
    SHAPEFILE_DIR.mkdir(parents=True, exist_ok=True)
    shapefile_path = SHAPEFILE_DIR / SHAPEFILE_NAME

    # Define three more organic-looking polygons within the RASTER_BOUNDS
    # Polygon 1: Lower-left area
    poly1_coords = [
        (MIN_X + 50, MIN_Y + 50), (MIN_X + 150, MIN_Y + 20), (MIN_X + 300, MIN_Y + 70),
        (MIN_X + 350, MIN_Y + 150), (MIN_X + 300, MIN_Y + 250), (MIN_X + 200, MIN_Y + 350),
        (MIN_X + 100, MIN_Y + 300), (MIN_X + 50, MIN_Y + 150),
        (MIN_X + 50, MIN_Y + 50) # Close polygon
    ]
    # Polygon 2: Upper-middle area, somewhat elongated
    poly2_coords = [
        (MIN_X + 400, MIN_Y + 600), (MIN_X + 500, MIN_Y + 550), (MIN_X + 650, MIN_Y + 600),
        (MIN_X + 700, MIN_Y + 700), (MIN_X + 600, MIN_Y + 850), (MIN_X + 500, MIN_Y + 900),
        (MIN_X + 400, MIN_Y + 800), (MIN_X + 350, MIN_Y + 700),
        (MIN_X + 400, MIN_Y + 600) # Close polygon
    ]
    # Polygon 3: Right-side area, with some indentations
    poly3_coords = [
        (MIN_X + 750, MIN_Y + 100), (MIN_X + 900, MIN_Y + 50), (MIN_X + 950, MIN_Y + 200),
        (MIN_X + 850, MIN_Y + 350), (MIN_X + 900, MIN_Y + 450), (MIN_X + 800, MIN_Y + 500),
        (MIN_X + 700, MIN_Y + 400), (MIN_X + 650, MIN_Y + 250),
        (MIN_X + 750, MIN_Y + 100) # Close polygon
    ]

    polygons = [Polygon(poly1_coords), Polygon(poly2_coords), Polygon(poly3_coords)]
    area_ids = [1, 2, 3]
    area_names = ["Demo Area 1", "Demo Area 2", "Demo Area 3"]

    gdf = geopandas.GeoDataFrame(
        data={AREA_ID_COL: area_ids, AREA_NAME_COL: area_names},
        geometry=polygons,
        crs=CRS
    )
    gdf.to_file(shapefile_path)
    print(f"Created shapefile: {shapefile_path}")

def create_sample_raster(file_path: pathlib.Path, value_min=0, value_max=100):
    """Creates a sample TIFF raster with random data."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate random float data
    data = np.random.uniform(value_min, value_max, (RASTER_HEIGHT, RASTER_WIDTH)).astype(rasterio.float32)

    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=RASTER_HEIGHT,
        width=RASTER_WIDTH,
        count=1,
        dtype=rasterio.float32,
        crs=CRS,
        transform=RASTER_TRANSFORM,
        nodata=NODATA_VALUE
    ) as dst:
        dst.write(data, 1)
    print(f"Created raster: {file_path}")

def generate_forecast_rasters():
    """Generates sample forecast rasters for Model A and Model B."""
    print("\nGenerating forecast rasters...")
    for year_str in YEARS_TO_PROCESS:
        for month_str in MONTHS_TO_PROCESS:
            init_date = datetime.strptime(f"{year_str}{month_str}01", "%Y%m%d")
            
            for lead_offset in LEAD_TIME_OFFSETS_MONTHS:
                target_date = init_date + relativedelta(months=lead_offset)
                target_ym_str = target_date.strftime("%Y%m")
                target_year_str = target_date.strftime("%Y")
                target_month_str = target_date.strftime("%m")

                # Model A
                model_a_init_folder = MODEL_A_DATA_FOLDER_TEMPLATE.parent / f"{year_str}_{month_str}" # Corrected path construction
                model_a_filename = MODEL_A_FILENAME_PATTERN.format(target_ym=target_ym_str)
                model_a_filepath = model_a_init_folder / model_a_filename
                create_sample_raster(model_a_filepath, value_min=0, value_max=200)

                # Model B
                model_b_init_folder = MODEL_B_DATA_FOLDER_TEMPLATE.parent / f"{year_str}_{month_str}" # Corrected path construction
                model_b_filename = MODEL_B_FILENAME_PATTERN.format(target_ym=target_ym_str)
                model_b_filepath = model_b_init_folder / model_b_filename
                create_sample_raster(model_b_filepath, value_min=5, value_max=250) # Slightly different values for Model B

def generate_observed_rasters():
    """Generates sample observed rasters for the target months."""
    print("\nGenerating observed rasters...")
    unique_target_months = set()
    for year_str in YEARS_TO_PROCESS:
        for month_str in MONTHS_TO_PROCESS:
            init_date = datetime.strptime(f"{year_str}{month_str}01", "%Y%m%d")
            for lead_offset in LEAD_TIME_OFFSETS_MONTHS:
                target_date = init_date + relativedelta(months=lead_offset)
                unique_target_months.add(target_date.strftime("%Y%m"))
    
    for target_ym_str in sorted(list(unique_target_months)):
        year_obs = target_ym_str[:4]
        month_obs = target_ym_str[4:]
        
        observed_filename = OBSERVED_FILENAME_PATTERN.format(year=year_obs, month=month_obs)
        observed_filepath = OBSERVED_DATA_FOLDER / observed_filename
        create_sample_raster(observed_filepath, value_min=0, value_max=150)

def main():
    """Main function to generate all sample data."""
    print(f"Starting sample data generation in: {DEMO_PROJECT_ROOT.resolve()}")
    
    create_sample_shapefile()
    generate_forecast_rasters()
    generate_observed_rasters()
    
    print("\n--- Sample Data Generation Complete ---")
    print(f"Please check the '{DEMO_PROJECT_ROOT / 'data'}' directory.")
    print("Ensure that GDAL and other geo-libraries are correctly installed if you run this script.")
    print("This script is intended to pre-generate data for the demo.")

if __name__ == "__main__":
    main() 