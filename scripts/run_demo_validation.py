import argparse
import sys
import os

# Add the parent directory of 'scripts' to sys.path to allow importing DemoValidationProcessor
# This assumes run_demo_validation.py is in the 'scripts' directory
# and demo_validation_processor.py is also in 'scripts'
# and config_demo.json is in the parent of 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.demo_validation_processor import DemoValidationProcessor

def main():
    parser = argparse.ArgumentParser(description="Run the Demo Rainfall Forecast Validation Process.")
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="config_demo.json", # Assumes config_demo.json is in the parent directory of 'scripts'
        help="Path to the configuration JSON file (relative to the project root, e.g., RainfallForecastValidation_Demo)."
    )
    parser.add_argument(
        "--models",
        nargs='+',
        help="Optional: List of specific models to process (e.g., model_A model_B). Processes all if not specified."
    )
    parser.add_argument(
        "--init_months",
        nargs='+',
        type=int,
        help="Optional: List of specific initialization months (1-12) to process. Processes all specified in config if not specified."
    )
    
    args = parser.parse_args()

    # The config file path needs to be relative to the project root (RainfallForecastValidation_Demo)
    # __file__ gives the path to this script (run_demo_validation.py)
    # os.path.dirname(__file__) gives 'RainfallForecastValidation_Demo/scripts'
    # os.path.join(os.path.dirname(__file__), '..', args.config_file) constructs the correct path
    # to 'RainfallForecastValidation_Demo/config_demo.json'
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, args.config_file)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script path: {__file__}")
        print(f"Calculated project root: {project_root}")
        return

    processor = DemoValidationProcessor(config_path=config_path)
    processor.run_validation(model_filter=args.models, init_months_filter=args.init_months)

if __name__ == "__main__":
    main() 