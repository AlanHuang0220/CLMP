def extract_config_components(config, component_keys):
    """
    Extracts specified components and their arguments from the configuration dictionary.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
        component_keys (list): List of keys to extract from the configuration.

    Returns:
        dict: A dictionary containing the extracted components and their arguments.
    """
    components = {}
    for key in component_keys:
        if key in config:
            component_info = config[key]
            component_type = component_info.get('type', None)
            component_args = component_info.get('args', None)

            components[key] = {
                'type': component_type,
                'args': component_args
            }
        else:
            # Handle missing components in the configuration
            components[key] = {'type': None, 'args': None}
    
    return components

def initialize_component(component_info, module, additional_args=None):
    """General function to initialize a component based on its configuration."""
    component_type = component_info.get('type')
    if component_type is None:
        raise ValueError("Component type is not specified in the configuration.")

    if hasattr(module, component_type):
        component_class = getattr(module, component_type)
        args = component_info.get('args', {})
        if additional_args:
            args.update(additional_args)
        return component_class(**args)
    else:
        raise ValueError(f"Component type '{component_type}' not found in the specified module.")