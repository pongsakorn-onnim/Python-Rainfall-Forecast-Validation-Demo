import json
import logging
import os
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta # For lead time calculations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterstats import zonal_stats

class DemoValidationProcessor:
    def __init__(self, config_path):
        """
        Initializes the processor with a configuration file.
        All paths in the config are expected to be relative to the config's base_path,
        which for the demo is the project root.
        """
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent # Assuming config_demo.json is in the project root

        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Using print for early errors as logger might not be set up.
            print(f"ERROR: Configuration file not found at {self.config_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {self.config_path}")
            raise

        # Path configurations from config_demo.json
        self.base_path = self.project_root / self.config.get("base_path", ".")
        
        # Output configuration
        output_cfg = self.config.get("output_config", {})
        self.output_folder = self.base_path / output_cfg.get("folder", "output")
        self.results_csv = self.output_folder / output_cfg.get("main_results_filename", "demo_results.csv")
        self.detailed_means_csv = self.output_folder / output_cfg.get("detailed_means_filename", "demo_means.csv")
        self.validation_matrix_excel = self.output_folder / output_cfg.get("validation_matrix_excel", "demo_matrix.xlsx")
        self.skipped_log_file = self.output_folder / output_cfg.get("skipped_log_filename", "demo_skipped.log")

        # Observed data configuration
        obs_cfg = self.config.get("observed_data_config", {})
        self.observed_data_folder = self.base_path / obs_cfg.get("folder", "data/observed_rainfall")
        self.observed_filename_pattern = obs_cfg.get("filename_pattern", "observed_rain_{year}{month}.tif")
        self.observed_file_type = obs_cfg.get("file_type", "tif")

        # Geospatial configuration
        geo_cfg = self.config.get("geospatial_config", {})
        self.shapefile_path = self.base_path / geo_cfg.get("shapefile_path", "data/shapefiles_demo/sample_boundary.shp")
        self.area_id_column = geo_cfg.get("area_id_column", "AREA_ID")
        self.area_name_column = geo_cfg.get("area_name_column", "AREA_NAME")

        # Forecast models configuration
        self.forecast_models_config = self.config.get("forecast_models_config", {})

        # Raster processing configuration
        raster_cfg = self.config.get("raster_processing_config", {})
        self.default_crs = raster_cfg.get("default_crs", "EPSG:32647")
        self.nodata_value = float(raster_cfg.get("nodata_value", -999.0)) # Ensure float
        resampling_str = raster_cfg.get("resampling_method", "bilinear")
        self.resampling_method = getattr(Resampling, resampling_str, Resampling.bilinear)

        # Analysis iterations configuration
        iter_cfg = self.config.get("analysis_iterations_config", {})
        self.years_to_process = iter_cfg.get("years_to_process", ["2022"])
        self.months_to_process = iter_cfg.get("months_to_process", ["01"]) # Init months
        self.lead_time_offsets = iter_cfg.get("lead_time_offsets_months", [0])

        # Logging configuration
        log_cfg = self.config.get("logging_config", {})
        self.log_file_name = self.output_folder / log_cfg.get("log_file_name", "demo_processing.log")
        self.log_level_str = log_cfg.get("log_level", "INFO").upper()
        self.log_level = getattr(logging, self.log_level_str, logging.INFO)

        self._setup_logging()
        self._validate_initial_paths()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Successfully loaded configuration from: {self.config_path}")
        self.logger.info(f"Project root set to: {self.project_root}")
        self.logger.info(f"Base path for data/output set to: {self.base_path}")

        try:
            self.logger.info(f"Loading shapefile: {self.shapefile_path}")
            self.watershed_gdf = gpd.read_file(self.shapefile_path)
            if self.area_id_column not in self.watershed_gdf.columns:
                msg = f"Area ID column '{self.area_id_column}' not found in shapefile {self.shapefile_path}. Available: {self.watershed_gdf.columns.tolist()}"
                self.logger.error(msg)
                raise ValueError(msg)
            if not pd.api.types.is_numeric_dtype(self.watershed_gdf[self.area_id_column]):
                 self.logger.warning(f"Area ID column '{self.area_id_column}' is not numeric. Attempting conversion.")
                 try:
                     self.watershed_gdf[self.area_id_column] = pd.to_numeric(self.watershed_gdf[self.area_id_column])
                 except ValueError as e:
                    msg = f"Could not convert Area ID column '{self.area_id_column}' to numeric: {e}"
                    self.logger.error(msg)
                    raise ValueError(msg)
            self.all_area_ids = sorted(self.watershed_gdf[self.area_id_column].dropna().astype(int).unique())
            self.logger.info(f"Shapefile loaded successfully. Area IDs: {self.all_area_ids}")
        except Exception as e:
            self.logger.error(f"Failed to load or process shapefile {self.shapefile_path}: {e}", exc_info=True)
            raise

    def _setup_logging(self):
        """Sets up logging for the processor."""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing handlers on the root logger if any
        if logging.root.handlers:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_name, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_initial_paths(self):
        """Validates essential paths found in the configuration."""
        if not self.shapefile_path.exists():
            msg = f"Shapefile not found: {self.shapefile_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        if not self.observed_data_folder.is_dir():
            msg = f"Observed data folder not found: {self.observed_data_folder}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        
        for model_name, model_config in self.forecast_models_config.items():
            try:
                if not self.years_to_process or not self.months_to_process:
                    self.logger.warning(f"Skipping path validation for model '{model_name}' due to empty year/month processing list.")
                    continue
                sample_year = self.years_to_process[0]
                sample_month = self.months_to_process[0]
                folder_template = model_config.get("data_folder_template")
                if folder_template:
                    # Check if the parent of a sample instance of the folder exists
                    sample_specific_folder = self.base_path / folder_template.format(year=sample_year, month=sample_month)
                    model_base_dir = sample_specific_folder.parent 
                    if not model_base_dir.is_dir():
                       self.logger.warning(f"Base directory for model '{model_name}' may not exist: {model_base_dir} (derived from template: {folder_template})")
                else:
                    self.logger.warning(f"'data_folder_template' not defined for model '{model_name}'. Cannot validate path.")
            except KeyError as e:
                self.logger.warning(f"Missing key '{e}' in config for model '{model_name}' during path validation.")
            except Exception as e:
                self.logger.error(f"Unexpected error during path validation for model '{model_name}': {e}", exc_info=True)

    def _find_forecast_file_path(self, model_name: str, init_year_str: str, init_month_str: str, target_year_str: str, target_month_str: str) -> Path | None:
        """
        Finds the forecast file path for a given model, initialization, and target period.

        Args:
            model_name: The name of the forecast model (e.g., "model_A").
            init_year_str: Initialization year string (e.g., "2022").
            init_month_str: Initialization month string (e.g., "01").
            target_year_str: Target year string (e.g., "2022").
            target_month_str: Target month string (e.g., "01").

        Returns:
            A Path object to the forecast file if found, otherwise None.
        """
        model_cfg = self.forecast_models_config.get(model_name)
        if not model_cfg:
            self.logger.error(f"Configuration for model '{model_name}' not found.")
            return None

        data_folder_template = model_cfg.get("data_folder_template")
        filename_pattern_options = model_cfg.get("filename_pattern_options")

        if not data_folder_template or not filename_pattern_options:
            self.logger.error(f"Incomplete configuration for model '{model_name}'. Missing data_folder_template or filename_pattern_options.")
            return None

        try:
            # Construct the specific directory for this model's initialization period
            # Example: data/forecast_generic_model_A/2022_01
            forecast_init_dir_str = data_folder_template.format(year=init_year_str, month=init_month_str)
            forecast_init_dir = self.base_path / forecast_init_dir_str

            if not forecast_init_dir.is_dir():
                self.logger.warning(f"Forecast directory not found for model '{model_name}', init '{init_year_str}-{init_month_str}': {forecast_init_dir}")
                return None

            target_ym_str = f"{target_year_str}{target_month_str}"
            
            for pattern in filename_pattern_options:
                # Format the filename pattern with the target year and month
                # Example pattern: "model_A_fcst_{target_ym}.tif"
                # Needs target_ym, target_year, target_month based on pattern needs.
                # Current demo config uses {target_ym}
                try:
                    filename = pattern.format(
                        target_ym=target_ym_str,
                        target_year=target_year_str, 
                        target_month=target_month_str,
                        init_year=init_year_str, # some patterns might need init year/month too
                        init_month=init_month_str
                    )
                    file_path = forecast_init_dir / filename
                    if file_path.exists() and file_path.is_file():
                        self.logger.debug(f"Found forecast file for model '{model_name}', init '{init_year_str}-{init_month_str}', target '{target_year_str}-{target_month_str}': {file_path}")
                        return file_path
                except KeyError as e:
                    self.logger.warning(f"Filename pattern '{pattern}' for model '{model_name}' has a missing key: {e}. Check config and available format vars.")
                    continue # Try next pattern
            
            self.logger.warning(f"No forecast file found for model '{model_name}', init '{init_year_str}-{init_month_str}', target '{target_year_str}-{target_month_str}' in {forecast_init_dir} using patterns: {filename_pattern_options}")
            return None

        except KeyError as e: # Handles missing keys in data_folder_template.format()
            self.logger.error(f"Error formatting data_folder_template '{data_folder_template}' for model '{model_name}'. Missing key: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in _find_forecast_file_path for model '{model_name}': {e}", exc_info=True)
            return None

    def _get_observed_file_path(self, target_year_str: str, target_month_str: str) -> Path | None:
        """
        Finds the observed rainfall file path for a given target year and month.

        Args:
            target_year_str: The target year string (e.g., "2022").
            target_month_str: The target month string (e.g., "01").

        Returns:
            A Path object to the observed file if found, otherwise None.
        """
        try:
            filename = self.observed_filename_pattern.format(year=target_year_str, month=target_month_str)
            file_path = self.observed_data_folder / filename
            if file_path.exists() and file_path.is_file():
                self.logger.debug(f"Found observed file for target '{target_year_str}-{target_month_str}': {file_path}")
                return file_path
            else:
                self.logger.warning(f"Observed file not found for target '{target_year_str}-{target_month_str}' at: {file_path}")
                return None
        except KeyError as e:
            self.logger.error(f"Error formatting observed_filename_pattern '{self.observed_filename_pattern}'. Missing key: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in _get_observed_file_path for target '{target_year_str}-{target_month_str}': {e}", exc_info=True)
            return None

    def run_validation(self, model_filter: list[str] | None = None, init_months_filter: list[str] | None = None):
        """
        Runs the core validation process.

        Args:
            model_filter: Optional list of model names to process. If None, all configured models are processed.
            init_months_filter: Optional list of initialization months (e.g. ["01", "02"]) to process. If None, uses config.
        """
        self.logger.info("Starting validation process...")
        
        all_results_data = [] # To store dictionaries of row data for final CSV
        skipped_combinations = [] # To log combinations that couldn't be processed

        models_to_process = model_filter if model_filter else self.forecast_models_config.keys()
        
        actual_init_months_to_process = init_months_filter if init_months_filter else self.months_to_process

        for model_name in models_to_process:
            if model_name not in self.forecast_models_config:
                self.logger.warning(f"Model '{model_name}' specified in filter but not found in configuration. Skipping.")
                continue
            
            self.logger.info(f"Processing model: {model_name}")

            for year_str in self.years_to_process:
                for init_month_str in actual_init_months_to_process:
                    try:
                        init_date = datetime.strptime(f"{year_str}{init_month_str}01", "%Y%m%d")
                    except ValueError:
                        self.logger.error(f"Invalid year/month combination for init_date: {year_str}-{init_month_str}. Skipping.")
                        skipped_combinations.append({
                            "model": model_name, "init_period": f"{year_str}-{init_month_str}", "lead_time_months": "N/A",
                            "target_period": "N/A", "reason": f"Invalid init date {year_str}-{init_month_str}"
                        })
                        continue

                    for lead_offset in self.lead_time_offsets:
                        target_date = init_date + relativedelta(months=lead_offset)
                        target_year_str = target_date.strftime("%Y")
                        target_month_str = target_date.strftime("%m")

                        self.logger.info(f"Processing: Model={model_name}, Init={year_str}-{init_month_str}, Lead={lead_offset}m, Target={target_year_str}-{target_month_str}")

                        forecast_file = self._find_forecast_file_path(
                            model_name, year_str, init_month_str, target_year_str, target_month_str
                        )
                        observed_file = self._get_observed_file_path(target_year_str, target_month_str)

                        if forecast_file and observed_file:
                            self.logger.info(f"  Forecast file: {forecast_file}")
                            self.logger.info(f"  Observed file: {observed_file}")
                            # Placeholder for actual processing:
                            mean_forecast_values = self._process_raster_to_zonal_means(forecast_file, self.watershed_gdf, self.area_id_column)
                            mean_observed_values = self._process_raster_to_zonal_means(observed_file, self.watershed_gdf, self.area_id_column)
                            
                            if mean_forecast_values is not None and mean_observed_values is not None:
                                self.logger.debug(f"    Forecast Means: {mean_forecast_values}")
                                self.logger.debug(f"    Observed Means: {mean_observed_values}")
                                metrics_data = self._calculate_and_store_metrics(
                                    model_name, year_str, init_month_str, 
                                    lead_offset, 
                                    target_year_str, target_month_str,
                                    mean_forecast_values, mean_observed_values, 
                                    self.all_area_ids
                                )
                                all_results_data.extend(metrics_data)
                            else:
                                reason_processing = "Raster processing failed: "
                                if mean_forecast_values is None: reason_processing += "Forecast raster error. "
                                if mean_observed_values is None: reason_processing += "Observed raster error."
                                self.logger.warning(f"Skipping combination due to raster processing failure: Model={model_name}, Init={year_str}-{init_month_str}, Lead={lead_offset}m. Reason: {reason_processing.strip()}")
                                skipped_combinations.append({
                                    "model": model_name, "init_period": f"{year_str}-{init_month_str}", 
                                    "lead_time_months": lead_offset, "target_period": f"{target_year_str}-{target_month_str}",
                                    "reason": reason_processing.strip()
                                })
                        else:
                            reason = ""
                            if not forecast_file:
                                reason += "Forecast file not found. "
                            if not observed_file:
                                reason += "Observed file not found."
                            self.logger.warning(f"Skipping combination: Model={model_name}, Init={year_str}-{init_month_str}, Lead={lead_offset}m. Reason: {reason.strip()}")
                            skipped_combinations.append({
                                "model": model_name,
                                "init_period": f"{year_str}-{init_month_str}",
                                "lead_time_months": lead_offset,
                                "target_period": f"{target_year_str}-{target_month_str}",
                                "reason": reason.strip()
                            })
        
        if not all_results_data:
            self.logger.warning("No results were generated. Check input data and configuration.")
            return

        # Save detailed per-area results
        self._save_results(all_results_data)
        self._save_skipped_log(skipped_combinations)

        # Calculate and save aggregated metrics
        aggregated_df = self._calculate_and_save_aggregated_metrics(all_results_data)

        # Create Excel validation matrix
        if aggregated_df is not None and not aggregated_df.empty:
            self._create_validation_matrix_excel(aggregated_df)
        else:
            self.logger.warning("Aggregated data is empty. Skipping Excel matrix creation.")

        # --- Generate Plots ---
        if aggregated_df is not None and not aggregated_df.empty:
            self._plot_metrics_vs_lead_time(aggregated_df)
        
        self._plot_validation_areas()
        # --- End of Plotting ---

        self.logger.info(f"Validation process finished.")

    def _process_raster_to_zonal_means(self, raster_file_path: Path, zone_gdf: gpd.GeoDataFrame, zone_id_col: str) -> dict[int, float] | None:
        """
        Processes a raster file to extract zonal mean statistics for given zones.

        Args:
            raster_file_path: Path to the input raster file.
            zone_gdf: GeoDataFrame containing the zones.
            zone_id_col: Name of the column in zone_gdf that contains unique zone identifiers.

        Returns:
            A dictionary mapping zone IDs to mean raster values, or None if processing fails.
        """
        self.logger.debug(f"Processing raster for zonal means: {raster_file_path}")
        try:
            with rasterio.open(raster_file_path) as src:
                src_crs = src.crs
                src_transform = src.transform
                src_nodata = src.nodata if src.nodata is not None else self.nodata_value # Use self.nodata_value if raster has no nodata
                src_array = src.read(1, masked=True) # Read as a masked array

                # Use the NoData value from the raster if available, otherwise from config.
                # This ensures that if a raster has a specific NoData, it's respected.
                # If we reproject, the reprojected array will use self.nodata_value.

                target_crs = rasterio.CRS.from_string(self.default_crs)
                reprojected_array = src_array
                reprojected_transform = src_transform

                if src_crs != target_crs:
                    self.logger.debug(f"Reprojecting {raster_file_path.name} from {src_crs} to {target_crs}")
                    
                    # Use bounds of the zone_gdf for the destination transform calculation
                    dst_bounds = zone_gdf.to_crs(target_crs).total_bounds #left, bottom, right, top
                    
                    # For demo purposes, let's fix the output resolution to be small, e.g., 100x100 pixels over the extent of the shapefile
                    # This keeps processing light and consistent for potentially varied source rasters.
                    # More sophisticated handling might try to match source resolution or use a configurable target resolution.
                    dst_width = 100 
                    dst_height = 100

                    dst_transform, _, _ = calculate_default_transform(
                        src_crs, target_crs, src.width, src.height, *src.bounds,
                        dst_width=dst_width, dst_height=dst_height, 
                        dst_bounds=dst_bounds # Pass calculated bounds for destination
                    )
                    
                    destination_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)

                    reproject(
                        source=src_array,
                        destination=destination_array,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=self.resampling_method,
                        src_nodata=src_nodata, # Nodata from source raster
                        dst_nodata=self.nodata_value # Nodata for destination array
                    )
                    reprojected_array = np.ma.masked_equal(destination_array, self.nodata_value)
                    reprojected_transform = dst_transform
                else:
                    self.logger.debug(f"Raster {raster_file_path.name} already in target CRS {target_crs}. No reprojection needed.")
                    # Ensure consistent nodata masking if not reprojected
                    reprojected_array = np.ma.masked_equal(src_array, src_nodata)


                # Perform zonal stats
                # rasterstats expects a non-masked array for `affine` and uses `nodata` param.
                # We pass the filled array and specify nodata value.
                stats = zonal_stats(
                    zone_gdf.to_crs(target_crs), # Ensure zones are in the same CRS as the (potentially reprojected) raster
                    reprojected_array.filled(fill_value=self.nodata_value), # Use filled array
                    affine=reprojected_transform,
                    stats=["mean"],
                    nodata=self.nodata_value, # Explicitly tell zonal_stats the nodata value
                    geojson_out=False
                )

                zonal_means = {}
                for i, stat_result in enumerate(stats):
                    area_id = zone_gdf[zone_id_col].iloc[i]
                    mean_val = stat_result.get('mean')
                    if mean_val is not None:
                        zonal_means[area_id] = float(mean_val)
                    else:
                        zonal_means[area_id] = np.nan # Or some other indicator for no valid data
                        self.logger.warning(f"No valid data found for area ID {area_id} in {raster_file_path.name}. Mean is None.")
                
                self.logger.debug(f"Successfully calculated zonal means for {raster_file_path.name}: {zonal_means}")
                return zonal_means

        except Exception as e:
            self.logger.error(f"Error processing raster {raster_file_path}: {e}", exc_info=True)
            return None

    def _calculate_and_store_metrics(self, model_name: str, 
                                     year_str: str, init_month_str: str, 
                                     lead_offset: int, 
                                     target_year_str: str, target_month_str: str, 
                                     forecast_means: dict, observed_means: dict, 
                                     area_ids: list) -> list[dict]:
        """
        Calculates basic validation metrics (error, squared error) for each area ID 
        and prepares the data for CSV output.
        """
        results_for_this_combination = []
        init_period_str = f"{year_str}-{init_month_str}"
        target_period_str = f"{target_year_str}-{target_month_str}"

        # Removed accuracy_metric_config loading as categorical/binary metrics are removed

        for area_id in area_ids:
            area_name_series = self.watershed_gdf[self.watershed_gdf[self.area_id_column] == area_id][self.area_name_column]
            area_name = area_name_series.iloc[0] if not area_name_series.empty else f"Unknown Area {area_id}"

            fcst_val = forecast_means.get(area_id, np.nan)
            obs_val = observed_means.get(area_id, np.nan)

            row = {
                "model_name": model_name,
                "init_period": init_period_str,
                "target_period": target_period_str,
                "lead_time_months": lead_offset,
                "area_id": area_id,
                "area_name": area_name,
                "forecast_value": fcst_val,
                "observed_value": obs_val,
                "error": np.nan,
                "squared_error": np.nan
            }

            if not np.isnan(fcst_val) and not np.isnan(obs_val):
                row["error"] = fcst_val - obs_val
                row["squared_error"] = row["error"] ** 2
            else:
                self.logger.debug(f"Missing forecast or observed data for area {area_id} (Model: {model_name}, Target: {target_period_str}). Fcst: {fcst_val}, Obs: {obs_val}. Error metrics will be NaN.")
            
            results_for_this_combination.append(row)
        
        self.logger.debug(f"Calculated basic error metrics for {len(results_for_this_combination)} areas for Model '{model_name}', Init '{init_period_str}', Lead '{lead_offset}m'.")
        return results_for_this_combination

    def _create_validation_matrix_excel(self, aggregated_metrics_df: pd.DataFrame):
        """
        Creates an Excel file with validation metrics formatted in a matrix style.

        Args:
            aggregated_metrics_df: DataFrame containing the aggregated metrics.
        """
        if aggregated_metrics_df.empty:
            self.logger.info("No aggregated metrics data to create Excel validation matrix.")
            return

        self.logger.info(f"Creating Excel validation matrix: {self.validation_matrix_excel}")
        try:
            with pd.ExcelWriter(self.validation_matrix_excel, engine='xlsxwriter') as writer:
                # --- Sheet 1: Continuous Metrics (RMSE, Bias, Pearson R, N_Pairs) ---
                self.logger.debug("Preparing 'Validation Metrics' sheet for Excel output.")
                sheet_name = "Validation Metrics"
                
                # Define relevant columns for the simplified metrics
                metric_cols_to_pivot = ['rmse', 'mean_error_bias', 'pearson_r', 'count_valid_pairs']
                pivoted_dfs = []

                for metric_col in metric_cols_to_pivot:
                    if metric_col in aggregated_metrics_df.columns:
                        pivot_df = aggregated_metrics_df.pivot(index='model_name', columns='lead_time_months', values=metric_col)
                        # Clarify column headers, e.g., RMSE_L0, Bias_L0, PearsonR_L0, N_Pairs_L0
                        col_prefix = ""
                        if metric_col == 'rmse': col_prefix = "RMSE"
                        elif metric_col == 'mean_error_bias': col_prefix = "Bias"
                        elif metric_col == 'pearson_r': col_prefix = "PearsonR"
                        elif metric_col == 'count_valid_pairs': col_prefix = "N_Pairs"
                        else: col_prefix = metric_col # Fallback, though unlikely with defined list
                        
                        pivot_df.columns = [f"{col_prefix}_L{col}" for col in pivot_df.columns]
                        pivoted_dfs.append(pivot_df)
                    else:
                        self.logger.warning(f"Metric column '{metric_col}' not found in aggregated data. Skipping its pivot for Excel.")
                
                if not pivoted_dfs:
                    self.logger.warning("No data could be pivoted for the Excel sheet. Skipping Excel file creation.")
                    return

                combined_metrics_df = pd.concat(pivoted_dfs, axis=1)
                combined_metrics_df.reset_index().to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]
                title_format = workbook.add_format({'bold': True, 'font_size': 14, 'align': 'center'})
                worksheet.merge_range(0, 0, 0, len(combined_metrics_df.columns), "Key Validation Metrics (RMSE, Bias, Pearson R, N_Pairs)", title_format)
                worksheet.autofit()

                self.logger.info(f"'{sheet_name}' sheet written to {self.validation_matrix_excel}")

            self.logger.info(f"Excel validation matrix saved successfully: {self.validation_matrix_excel}")

        except Exception as e:
            self.logger.error(f"Failed to create Excel validation matrix: {e}", exc_info=True)

    def _calculate_and_save_aggregated_metrics(self, all_results_data: list[dict]):
        """
        Calculates aggregated metrics (RMSE, Bias, Pearson R) 
        per model and lead time, and saves them to a CSV file.
        """
        if not all_results_data:
            self.logger.info("No results data available to calculate aggregated metrics.")
            return

        try:
            results_df = pd.DataFrame(all_results_data)
            self.logger.debug(f"Calculating aggregated metrics from {len(results_df)} total area-specific results.")

            # Drop rows where forecast_value or observed_value is NaN, as these pairs cannot be used for R, RMSE, or Bias.
            valid_pairs_df = results_df.dropna(subset=['forecast_value', 'observed_value'])
            self.logger.debug(f"Using {len(valid_pairs_df)} valid (non-NaN forecast/observed) pairs for aggregation.")

            if valid_pairs_df.empty:
                self.logger.info("No valid data pairs with non-NaN forecast/observed values to calculate aggregated metrics.")
                # Create an empty DataFrame with expected columns if needed for consistency, or just return.
                # For now, if no valid pairs, no aggregated metrics file will be produced for these types.
                return

            # Define a function to calculate Pearson R, handling cases with too few data points or zero variance.
            def calculate_pearson_r(group):
                if len(group['forecast_value']) < 2:
                    return np.nan # Pearson R not defined for less than 2 points
                # Check for zero variance in either forecast or observed values
                if group['forecast_value'].var() == 0 or group['observed_value'].var() == 0:
                    return np.nan 
                return group['forecast_value'].corr(group['observed_value'])

            # Aggregate RMSE, Bias, Count
            aggregated_df = valid_pairs_df.groupby(['model_name', 'lead_time_months']).agg(
                mean_error_bias=('error', 'mean'),
                rmse=('squared_error', lambda x: np.sqrt(x.mean())),
                count_valid_pairs=('error', 'count')
            ).reset_index()

            # Calculate Pearson R separately and merge
            pearson_r_series = valid_pairs_df.groupby(['model_name', 'lead_time_months']).apply(
                calculate_pearson_r, include_groups=False
            ).rename('pearson_r')
            
            aggregated_df = aggregated_df.merge(pearson_r_series, on=['model_name', 'lead_time_months'], how='left')
            
            aggregated_df['mean_error_bias'] = aggregated_df['mean_error_bias'].round(4)
            aggregated_df['rmse'] = aggregated_df['rmse'].round(4)
            aggregated_df['pearson_r'] = aggregated_df['pearson_r'].round(4)
            
            if aggregated_df.empty:
                self.logger.info("Aggregated metrics DataFrame is empty after calculations.")
                return

            self.output_folder.mkdir(parents=True, exist_ok=True)
            aggregated_df.to_csv(self.detailed_means_csv, index=False)
            self.logger.info(f"Aggregated metrics (RMSE, Bias, Pearson R) saved to: {self.detailed_means_csv}")
            self.logger.debug(f"Aggregated metrics content:\n{aggregated_df.to_string()}")
            
            # Call to create Excel after aggregated_df is ready
            if not aggregated_df.empty:
                self._create_validation_matrix_excel(aggregated_df)

            return aggregated_df

        except Exception as e:
            self.logger.error(f"Failed to calculate or save aggregated metrics: {e}", exc_info=True)
            return None # Ensure a DataFrame or None is returned

    def _save_results(self, results_data: list):
        """Saves the aggregated results to a CSV file."""
        if not results_data:
            self.logger.info("No results data to save.")
            return
        
        try:
            results_df = pd.DataFrame(results_data)
            # Ensure output_folder exists (it should by now, but good practice)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(self.results_csv, index=False)
            self.logger.info(f"Main validation results saved to: {self.results_csv}")
        except Exception as e:
            self.logger.error(f"Failed to save main results CSV: {e}", exc_info=True)

    def _save_skipped_log(self, skipped_data: list):
        """Saves the log of skipped combinations to a CSV file."""
        if not skipped_data:
            self.logger.info("No skipped combinations to log.")
            return
        
        try:
            skipped_df = pd.DataFrame(skipped_data)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            skipped_df.to_csv(self.skipped_log_file, index=False)
            self.logger.info(f"Skipped combinations log saved to: {self.skipped_log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save skipped combinations log: {e}", exc_info=True)

    # --- New Plotting Methods ---
    def _plot_metrics_vs_lead_time(self, aggregated_df):
        """Generates and saves line plots for each metric vs. lead time."""
        if aggregated_df is None or aggregated_df.empty:
            self.logger.warning("Aggregated data is empty. Skipping metrics plots.")
            return

        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = {
                'rmse': 'RMSE',
                'mean_error_bias': 'Bias (Mean Error)',
                'pearson_r': 'Pearson Correlation (R)'
            }
            models = aggregated_df['model_name'].unique()

            for metric_col, pretty_name in metrics_to_plot.items():
                plt.figure(figsize=(10, 6))
                for model in models:
                    model_data = aggregated_df[aggregated_df['model_name'] == model]
                    plt.plot(model_data['lead_time_months'], model_data[metric_col], marker='o', linestyle='-', label=model)
                
                plt.xlabel("Lead Time (Months)")
                plt.ylabel(pretty_name)
                plt.title(f"{pretty_name} vs. Lead Time by Model")
                plt.legend()
                plt.grid(True)
                plt.xticks(aggregated_df['lead_time_months'].unique()) # Ensure all lead times are marked

                plot_filename = self.output_folder / f"plot_{metric_col}_vs_lead_time.png"
                plt.savefig(plot_filename)
                plt.close() # Close the figure to free memory
                self.logger.info(f"Saved plot: {plot_filename}")

        except ImportError:
            self.logger.error("Matplotlib is not installed. Skipping metrics plots. Please install it (e.g., pip install matplotlib).")
        except Exception as e:
            self.logger.error(f"Failed to generate metrics plots: {e}", exc_info=True)

    def _plot_validation_areas(self):
        """Generates and saves a plot of the validation areas from the shapefile."""
        if self.watershed_gdf is None or self.watershed_gdf.empty:
            self.logger.warning("Watershed GeoDataFrame is not loaded. Skipping validation areas plot.")
            return
        
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            self.watershed_gdf.plot(ax=ax, facecolor='lightblue', edgecolor='black', alpha=0.7)
            
            # Add labels for area IDs if the column exists and is not too crowded
            area_id_col = self.config.get("shapefile_config", {}).get("area_id_column", "AREA_ID")
            if area_id_col in self.watershed_gdf.columns and len(self.watershed_gdf) < 20: # Avoid clutter for many polygons
                self.watershed_gdf.apply(lambda x: ax.annotate(text=x[area_id_col], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
            
            plt.title("Validation Areas")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True)
            
            plot_filename = self.output_folder / "plot_validation_areas.png"
            plt.savefig(plot_filename)
            plt.close() # Close the figure to free memory
            self.logger.info(f"Saved plot: {plot_filename}")

        except ImportError:
            self.logger.error("Matplotlib or GeoPandas might not be installed correctly. Skipping validation areas plot.")
        except Exception as e:
            self.logger.error(f"Failed to generate validation areas plot: {e}", exc_info=True)

if __name__ == '__main__':
    print(f"Class {DemoValidationProcessor.__name__} defined. Run via a dedicated runner script.") 