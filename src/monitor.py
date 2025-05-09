# src/monitor.py
# Contains the main RespirationMonitor class that orchestrates the process.
# MODIFIED: Correctly handles calibration results and variance map passing.
# MODIFIED: Added helper methods for UI data retrieval.

import time
import traceback # For error printing
import cv2
import numpy as np # Import numpy

# Import necessary components from the src package
try:
    # Use relative imports assuming standard package structure
    from .config_loader import load_config, MINIMAL_DEFAULTS
    from .video_input import VideoInput
    from .calibration import Calibration
    from .motion_detection import MotionDetector
    from .signal_processing import SignalProcessor
    from .adaptive_control import AdaptiveController
except ImportError as e:
    print(f"Error importing modules within src package: {e}")
    print("Ensure all .py files (config_loader, video_input, etc.) exist in the 'src' directory.")
    raise # Re-raise the error to stop execution if imports fail

class RespirationMonitor:
    """Encapsulates the core respiration monitoring pipeline."""

    def __init__(self, config_filepath=None):
        """Initializes the monitor, loading config from the specified filepath.

        Args:
            config_filepath (str, optional): Path to the JSON configuration file.
                                             If None, minimal defaults are used.
        """
        # Load config from file or use minimal defaults
        if config_filepath:
            self.config = load_config(config_filepath)
        else:
            print("Warning: No config filepath provided. Using minimal defaults.")
            self.config = MINIMAL_DEFAULTS.copy() # Use copy of defaults

        # Initialize component placeholders
        self.video_input = None
        self.calibration = None
        self.motion_detector = None
        self.signal_processor = None
        self.adaptive_controller = None
        self.roi = None
        # Add variance_map placeholder
        self.variance_map = None
        self.is_initialized = False
        self.last_frame = None # Store last frame for display/debugging if needed

    def initialize(self, manual_roi=None):
        """Initializes all components of the monitor.
           Skips automatic calibration if manual_roi is provided.

        Args:
            manual_roi (tuple, optional): A pre-defined ROI (x, y, w, h) to use,
                                          skipping automatic calibration. Defaults to None.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        print("Initializing Respiration Monitor...")
        try:
            # 1. Initialize Video Input
            self.video_input = VideoInput(self.config)
            if not self.video_input.initialize():
                print("Monitor initialization failed: VideoInput failed.")
                return False

            # 2. Determine ROI and Variance Map
            self.roi = None
            self.variance_map = None
            self.calibration = None # Ensure calibration object is reset

            # Check for valid manual ROI
            if manual_roi is not None and isinstance(manual_roi, tuple) and len(manual_roi) == 4:
                x, y, w, h = manual_roi
                if w > 0 and h > 0:
                     print(f"Using manually provided ROI: {manual_roi}. Skipping automatic calibration.")
                     self.roi = manual_roi
                     # No variance map when using manual ROI
                     self.variance_map = None
                else:
                     print("Warning: Manual ROI has invalid dimensions. Proceeding with calibration.")
                     manual_roi = None # Force calibration if dims invalid

            # Run calibration if no valid manual ROI was provided
            if self.roi is None:
                print("Attempting automatic calibration...")
                # --- Add print statement to confirm execution path ---
                print("--- monitor.py: About to create Calibration object ---")
                self.calibration = Calibration(self.video_input, self.config)
                # --- Correctly capture both return values ---
                calibration_result_roi, calibration_result_map = self.calibration.run_calibration()

                # --- Check results ---
                if calibration_result_roi is not None:
                    self.roi = calibration_result_roi
                    self.variance_map = calibration_result_map # Store map (could be None if map calc failed)
                    print(f"Calibration successful. ROI: {self.roi}")
                    if self.variance_map is not None:
                         # Ensure map is float32 if it exists
                         if isinstance(self.variance_map, np.ndarray) and self.variance_map.dtype != np.float32:
                              try:
                                   self.variance_map = self.variance_map.astype(np.float32)
                                   print(f"  Variance Map Shape: {self.variance_map.shape}, Dtype: {self.variance_map.dtype} (Converted to float32)")
                              except Exception as e_cast:
                                   print(f"  Warning: Failed to cast variance map to float32: {e_cast}. Setting map to None.")
                                   self.variance_map = None
                         elif isinstance(self.variance_map, np.ndarray):
                              print(f"  Variance Map Shape: {self.variance_map.shape}, Dtype: {self.variance_map.dtype}")
                         else: # Handle UMat case if necessary, though calibration should return NumPy
                              print(f"  Variance Map Type: {type(self.variance_map)}. Check calibration return type.")

                    else:
                         print("  Warning: Calibration succeeded for ROI, but failed to generate variance map.")
                else:
                    # --- This is where the original failure occurred ---
                    print("Monitor initialization failed: Calibration unsuccessful (run_calibration returned None for ROI).")
                    self.video_input.release() # Release video if calibration fails
                    return False

            # --- At this point, self.roi should be valid, self.variance_map might be None ---

            # 3. Initialize Motion Detector with the determined ROI and variance map
            # MotionDetector.__init__ needs to handle variance_map being None
            try:
                self.motion_detector = MotionDetector(self.roi, self.config, self.variance_map)
            except Exception as e_md:
                 print(f"Error initializing MotionDetector: {e_md}")
                 traceback.print_exc()
                 self.video_input.release()
                 return False

            # 4. Initialize Signal Processor
            fps = self.video_input.get_fps()
            if fps <= 0:
                 print("Error: Cannot initialize SignalProcessor with invalid FPS.")
                 self.video_input.release()
                 return False
            try:
                self.signal_processor = SignalProcessor(fps, self.config)
            except Exception as e_sp:
                 print(f"Error initializing SignalProcessor: {e_sp}")
                 traceback.print_exc()
                 self.video_input.release()
                 return False

            # 5. Initialize Adaptive Controller (if enabled)
            if self.config.get('USE_ADAPTIVE_CONTROL', False):
                try:
                    self.adaptive_controller = AdaptiveController(self.config)
                    # Sync controller's initial state with the detector's state
                    self.adaptive_controller.current_method = self.motion_detector.method
                except Exception as e_ac:
                     print(f"Error initializing AdaptiveController: {e_ac}")
                     # Continue initialization even if adaptive control fails? Or return False?
                     # For now, print warning and continue.
                     print("Warning: AdaptiveController failed to initialize, adaptive control disabled.")
                     self.adaptive_controller = None


            self.is_initialized = True
            print("Respiration Monitor initialized successfully.")
            return True

        except Exception as e:
             print(f"Unexpected error during monitor initialization: {e}")
             traceback.print_exc()
             # Attempt cleanup if partially initialized
             if self.video_input: self.video_input.release()
             self.is_initialized = False
             return False

    def run_cycle(self):
        """Runs one cycle of frame capture and processing.

        Returns:
            dict: A dictionary containing results like:
                  {'success': bool, 'bpm': float, 'normalized_signal': float,
                   'method': str, 'frame': np.ndarray/None, 'roi': tuple/None,
                   'peaks': list, 'error': str (optional)}
                  Returns {'success': False} if not initialized or error occurs.
        """
        if not self.is_initialized or self.video_input is None:
            return {'success': False, 'error': 'Monitor not initialized'}

        try:
            # 1. Get Frame
            success, frame = self.video_input.get_frame()
            if not success:
                # print("Debug: End of video source reached or error reading frame.") # Reduce noise
                return {'success': False, 'error': 'Video ended or failed'}

            # Store the frame (convert UMat for storage/return if needed)
            try:
                 # Store as NumPy for easier handling outside this class
                 self.last_frame = frame.get() if isinstance(frame, cv2.UMat) else frame
            except Exception as e_get:
                 print(f"Warning: Could not get frame data for storage: {e_get}")
                 self.last_frame = None # Set to None if conversion fails

            # Ensure frame is usable before motion detection
            if self.last_frame is None:
                 return {'success': False, 'error': 'Failed to get valid frame data'}


            # 2. Detect Motion
            # Ensure motion_detector exists (should always if initialized)
            if self.motion_detector is None:
                 return {'success': False, 'error': 'Motion detector not initialized'}
            # Pass the original frame (UMat or NumPy) to process_frame
            motion_value = self.motion_detector.process_frame(frame)

            # 3. Process Signal
            # Ensure signal_processor exists
            if self.signal_processor is None:
                 return {'success': False, 'error': 'Signal processor not initialized'}
            self.signal_processor.process_signal(motion_value)
            # --- Assume analyze_buffer now stores peaks in self.last_peaks ---
            filtered_signal, bpm, peaks, normalized_signal_value = self.signal_processor.analyze_buffer()

            # 4. Prepare Results
            # Note: Adaptation logic is triggered externally via trigger_adaptation()
            results = {
                'success': True,
                'bpm': bpm,
                'normalized_signal': normalized_signal_value if normalized_signal_value is not None else 0.0, # Default to 0 if None
                'method': self.motion_detector.method,
                'frame': self.last_frame, # Return the processed frame (NumPy)
                'roi': self.roi,
                'peaks': peaks, # Pass peaks info for UI
            }
            return results

        except Exception as e:
             print(f"Unexpected error during run_cycle: {e}")
             traceback.print_exc()
             return {'success': False, 'error': f'Runtime error: {e}'}


    def trigger_adaptation(self, fps):
         """Externally trigger the adaptive controller logic, passing current FPS."""
         if self.adaptive_controller and self.motion_detector and self.signal_processor:
              # Get necessary metrics
              bpm_stability = self.signal_processor.get_bpm_stability()
              tracked_points = self.motion_detector.get_tracked_points_count()

              # Update controller's monitor state
              self.adaptive_controller.monitor(fps, bpm_stability, tracked_points)
              # Trigger adaptation logic
              self.adaptive_controller.adapt(self.motion_detector)

    def get_current_method(self):
        """Returns the currently active motion detection method."""
        return self.motion_detector.method if self.motion_detector else "N/A"

    # --- Add method to get signal history for UI ---
    def get_filtered_signal_history(self):
        """Returns the last valid filtered signal buffer for plotting."""
        if self.signal_processor:
            # Return the stored last valid signal (could be raw if filtering failed)
            # Assumes SignalProcessor has 'last_valid_filtered_signal' attribute
            return getattr(self.signal_processor, 'last_valid_filtered_signal', None)
        return None

    def get_last_detected_peaks(self):
         """Returns the indices of the last detected peaks relative to the signal history."""
         if self.signal_processor:
              # Assumes SignalProcessor stores the last peaks found in 'last_peaks'
              return getattr(self.signal_processor, 'last_peaks', [])
         return []


    def stop(self):
        """Releases resources, particularly the video input."""
        print("Stopping Respiration Monitor...")
        if self.video_input:
            self.video_input.release()
        self.is_initialized = False
        print("Respiration Monitor stopped.")

# (No test block needed here, main.py handles execution)
