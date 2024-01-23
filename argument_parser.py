import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model using the specified configuration file.')
    parser.add_argument('-config', '--config_file', type=str, required=True,
                        help='Path to the YAML configuration file.')
    return parser.parse_args()