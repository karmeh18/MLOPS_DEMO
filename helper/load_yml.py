import yaml
from src.logger import logging
from src.exception import Custom_Exception


def load_params(params_path: str) -> dict:
    """Loading parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters Loaded from %s', params_path)
        return params
    except Exception as e:
        logging.error('Failed to load parameters from %s', params_path)
        print(f"Error loading parameters: {e}")