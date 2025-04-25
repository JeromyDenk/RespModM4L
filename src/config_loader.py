# src/config_loader.py
# Handles loading and merging of configuration files.

import json
import os
import cv2 # Needed for TERM_CRITERIA constants if converting here
import traceback

# --- Default values (used if keys are missing in JSON) ---
# Keep a minimal set of essential defaults here
MINIMAL_DEFAULTS = {
    'VIDEO_SOURCE': 0,
    'DEFAULT_MOTION_METHOD': 'ROI_Average',
    'SIGNAL_BUFFER_SIZE': 300,
    'LOWPASS_FILTER_CUTOFF_HZ': 2.0,
    'LOWPASS_FILTER_ORDER': 4,
    'MAX_EXPECTED_BPM': 120,
    'PEAK_FINDING_DISTANCE_FACTOR': 0.5,
    'PEAK_FINDING_PROMINENCE_FACTOR': 0.1,
    'OPTICAL_FLOW_PARAMS': { # Need defaults for nested dicts too
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': [15, 15], 'maxLevel': 2, 'criteria': [3, 10, 0.03]} # Use list/int for JSON compatibility
    },
    # Add other critical defaults if necessary
    'CALIBRATION_METHOD': 'MotionVariance',
    'CALIBRATION_DURATION_FRAMES': 150,
    'ROI_AVG_PREFILTER': 'Median',
    'ROI_AVG_MEDIAN_KSIZE': 5,
    'ROI_AVG_GAUSSIAN_KERNEL': [3, 3], # Use list for JSON compatibility
    'ROI_REFINE_SIZE': False,
    'USE_GAUSSIAN_FIT': False,
    'GAUSSIAN_FIT_WINDOW_FACTOR': 3,
    'GAUSSIAN_FIT_MIN_STDDEV': 1.0,
    'GAUSSIAN_FIT_MAX_STDDEV': 50.0,
    'USE_OPENCL': False,
    'USE_ADAPTIVE_CONTROL': False,
    'ADAPTIVE_FPS_THRESHOLD': 15,
    'ADAPTIVE_BPM_STABILITY_THRESHOLD': 15,
    'ADAPTIVE_HYSTERESIS_FRAMES': 60,
    'DEBUG_PRINT_VALUES': False,
    'DEBUG_CALIBRATION_VIZ': False,
    'EVM_PYRAMID_LEVEL': 2,
    'EVM_LOW_FREQ_HZ': 0.4,
    'EVM_HIGH_FREQ_HZ': 2.0,
}


# --- Helper function to load and merge config ---
def load_config(filepath):
    """Loads config from JSON file and merges with minimal defaults."""
    config = MINIMAL_DEFAULTS.copy() # Start with minimal defaults
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)

            # --- Deep Merge Logic ---
            # Simple config.update() doesn't handle nested dictionaries well.
            # We need to merge nested dicts like OPTICAL_FLOW_PARAMS properly.
            def merge_dicts(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        merge_dicts(base[key], value)
                    else:
                        base[key] = value
                return base

            config = merge_dicts(config, loaded_config)
            print(f"Loaded and merged configuration from: {filepath}")

            # --- Post-Load Type Conversions (if needed by OpenCV/other libs) ---
            # Example: Convert list from JSON back to tuple for specific OpenCV params
            # Note: It's often better to do these conversions right before calling the
            # specific function that needs the tuple, rather than globally here.
            # However, if frequently needed, conversion can happen here.

            # Example: winSize for lk_params
            if 'lk_params' in config.get('OPTICAL_FLOW_PARAMS', {}) and \
               'winSize' in config['OPTICAL_FLOW_PARAMS']['lk_params'] and \
               isinstance(config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'], list):
                try:
                    config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'] = tuple(config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'])
                except TypeError:
                     print(f"Warning: Could not convert lk_params.winSize to tuple. Keeping list.")


            # Example: Gaussian Kernel
            if 'ROI_AVG_GAUSSIAN_KERNEL' in config and isinstance(config['ROI_AVG_GAUSSIAN_KERNEL'], list):
                 try:
                    config['ROI_AVG_GAUSSIAN_KERNEL'] = tuple(config['ROI_AVG_GAUSSIAN_KERNEL'])
                 except TypeError:
                      print(f"Warning: Could not convert ROI_AVG_GAUSSIAN_KERNEL to tuple. Keeping list.")

            # Example: LK criteria (OpenCV needs tuple: (type, maxCount, epsilon))
            # JSON stores [type_int, maxCount, epsilon]
            if 'lk_params' in config.get('OPTICAL_FLOW_PARAMS', {}) and \
               'criteria' in config['OPTICAL_FLOW_PARAMS']['lk_params'] and \
               isinstance(config['OPTICAL_FLOW_PARAMS']['lk_params']['criteria'], list) and \
               len(config['OPTICAL_FLOW_PARAMS']['lk_params']['criteria']) == 3:
                try:
                    # Combine flags correctly for OpenCV criteria tuple
                    # Type flags: cv2.TERM_CRITERIA_EPS=2, cv2.TERM_CRITERIA_COUNT=1
                    # Default type is usually EPS | COUNT = 3
                    crit_list = config['OPTICAL_FLOW_PARAMS']['lk_params']['criteria']
                    type_flags = int(crit_list[0]) # Should be 1, 2, or 3 from JSON
                    max_iter = int(crit_list[1])
                    epsilon = float(crit_list[2])
                    config['OPTICAL_FLOW_PARAMS']['lk_params']['criteria'] = (type_flags, max_iter, epsilon)
                except (ValueError, TypeError, IndexError) as e_crit:
                     print(f"Warning: Could not convert lk_params.criteria to tuple ({e_crit}). Keeping list or default.")
                     # Optionally fall back to a hardcoded default tuple if conversion fails
                     # config['OPTICAL_FLOW_PARAMS']['lk_params']['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


        else:
            print(f"Warning: Config file not found at '{filepath}'. Using minimal defaults.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}. Using minimal defaults.")
    except Exception as e:
        print(f"Error loading config file '{filepath}': {e}. Using minimal defaults.")
        traceback.print_exc() # Print traceback for unexpected loading errors
    return config

# Example usage (for testing this module directly)
if __name__ == '__main__':
    # Assume configs folder is one level up from src
    script_dir = os.path.dirname(__file__)
    config_dir = os.path.join(script_dir, '..', 'configs')

    test_file = os.path.join(config_dir, 'config_default.json')
    print(f"Attempting to load: {test_file}")
    cfg = load_config(test_file)
    print("\nLoaded Config:")
    import pprint
    pprint.pprint(cfg)

    print("-" * 20)

    test_file_missing = os.path.join(config_dir, 'non_existent_config.json')
    print(f"Attempting to load missing file: {test_file_missing}")
    cfg_missing = load_config(test_file_missing)
    print("\nConfig from missing file (should be defaults):")
    pprint.pprint(cfg_missing)
