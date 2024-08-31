import numpy as np
import argparse
import os
import re
import utils
import ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.explicit_start = True

def custom_parser(value):
    # individual element parser
    def parse_element(element):
        # Check for None, null, or empty
        if element is None or element.lower() in ['none', 'null', '']:
            return None
        # Detect floats, scientific notations, ints
        if re.match(r'^-?[\d.]+(?:e-?\d+)?$', element, re.IGNORECASE) or re.match(r'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$', element):
            return int(float(element)) if float(element) == int(float(element)) else float(element)
        # Detect booleans
        elif element.lower() in ['true', 'false']:
            return element.lower() == 'true'
        # Return as string if none of the above
        return element.strip("'\"")

    def split_and_parse(element_str):
        if not element_str:
            return []
        # Split by comma and strip whitespace, then parse each element
        return [parse_element(e.strip()) for e in element_str.split(',')]

    # Detect tuples
    if value.startswith('(') and value.endswith(')'):
        return tuple(split_and_parse(value.strip('()')))
    # Detect lists
    elif value.startswith('[') and value.endswith(']'):
        return split_and_parse(value.strip('[]'))
    # For other cases, pass the value to the recursive function
    else:
        return parse_element(value)

def object_to_dict(obj):
    result = {}
    for key, value in obj.__dict__.items():
        if not key.startswith('_'):
            result[key] = value
    return result

def cfg2dic(config):
    return object_to_dict(config)

class Config(object):

    def __init__(self, cfg_id, date=None, overrides=None):
        self.id = cfg_id
        self._keys_seen = set()
        self._paths = {}

        if date is not None:
            cfg_name = './cfg/%s/%s.yaml' % (date, cfg_id)
        else:
            cfg_name = './cfg/%s.yaml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)

        config_data = yaml.load(open(cfg_name, 'r'))
        self._original_config_data = config_data
        self._reverse_path = self.reverse_nested_dict(self._original_config_data)
        self._build_config(config_data)

        if overrides:
            overrides = self.convert_overrides_to_dict(overrides)
            self._apply_overrides(overrides)

    def _build_config(self, config_data):
        self._flatten_and_set_attrs(config_data)
    
    def _flatten_and_set_attrs(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                self._flatten_and_set_attrs(value)
            else:
                # Just use the key as the attribute name
                if key in self._keys_seen:
                    raise ValueError(f'Duplicate key found: {key}')
                self._keys_seen.add(key)
                setattr(self, key, custom_parser(str(value)))


    def _apply_overrides(self, overrides):
        for key, value in overrides.items():
            # Update the attribute if it exists
            if hasattr(self, key):
                setattr(self, key, custom_parser(str(value)))
                # Update the original config data by following the path
                if key in self._reverse_path:
                    parent_key = self._reverse_path[key]
                    if parent_key:  # If parent key exists, it's a nested dictionary
                        self._original_config_data[parent_key][key] = custom_parser(str(value))
                    else:  # The key exists at the top level
                        self._original_config_data[key] = custom_parser(str(value))
            # Otherwise, add a new attribute and add to "additional" key
            else:
                setattr(self, key, custom_parser(str(value)))
                if 'additional' not in self._original_config_data:
                    self._original_config_data['additional'] = {}
                self._original_config_data['additional'][key] = value

    def write_yaml(self, output_path):
        full_path = os.path.join(output_path, self.id + '_updated.yaml')
        with open(full_path, 'w') as output_file:
            yaml.dump(self._original_config_data, output_file)

        
    @staticmethod
    def from_args(args):
        # Extract only named arguments without prefix '--' as overrides
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        
        # Remove recognized arguments to leave only overrides
        recognized_args = ['cfg', 'd', 'gpu', 'seed']  # Add more if needed
        for arg in recognized_args:
            overrides.pop(arg, None)
        
        return Config(args.cfg, args.d, overrides)


    @staticmethod
    def convert_overrides_to_dict(overrides_list):
        overrides_dict = {}
        iterator = iter(overrides_list)
        for key in iterator:
            # Check that the key is formatted correctly
            if not key.startswith('--'):
                raise ValueError(f"Unexpected format for key: {key}")
            key = key[2:]  # Remove the '--' prefix
            try:
                value = next(iterator)
                # If the value also looks like a key, raise an error
                if value.startswith('--'):
                    raise ValueError(f"Missing value for key: --{key}")
            except StopIteration:
                # If we run out of items, use None for the value
                value = None
            overrides_dict[key] = value
        return overrides_dict


    def reverse_nested_dict(self, orig_dict):
        reversed_dict = {}
        for outer_key, value in orig_dict.items():
            if isinstance(value, dict):  # Check if the value is a dictionary
                for inner_key in value.keys():
                    reversed_dict[inner_key] = outer_key
            else:
                reversed_dict[outer_key] = None
        return reversed_dict