import yaml


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): The path to the YAML file.

    Returns:
        dict: The loaded parameters as a dictionary.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        print(f"Error loading parameters: {e}")

params = load_params("D:\Vikash_Dash_Demo_Spam_MLOps\params.yml")
print(params["data_import"]['random_state'])
print(params['data_explorer']['target_column'][1])


