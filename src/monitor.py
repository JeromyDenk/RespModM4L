# src/monitor.py
# Contains the main RespirationMonitor class that orchestrates the process.

import time
import traceback # For error printing
import cv2

# Import necessary components from the src package
try:
    from .config_loader import load_config, MINIMAL_DEFAULTS # Use relative import
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
                return False # Cannot proceed without video

            # 2. Determine ROI (Manual or Automatic Calibration)
            if manual_roi is not None and len(manual_roi) == 4:
                x, y, w, h = manual_roi
                if w > 0 and h > 0:
                     print(f"Using manually provided ROI: {manual_roi}. Skipping automatic calibration.")
                     self.roi = manual_roi
                else:
                     print("Warning: Manual ROI has invalid dimensions. Falling back to calibration.")
                     manual_roi = None # Force calibration
            else:
                 # print("No valid manual ROI provided.") # Less verbose
                 manual_roi = None # Ensure it's None if invalid or not provided

            if self.roi is None: # If no valid manual ROI was set, run automatic calibration
                self.calibration = Calibration(self.video_input, self.config)
                print("Running automatic calibration...")
                self.roi = self.calibration.run_calibration()
                if self.roi is None:
                    print("Monitor initialization failed: Calibration unsuccessful.")
                    self.video_input.release() # Release video if calibration fails
                    return False

            # 3. Initialize Motion Detector with the determined ROI
            self.motion_detector = MotionDetector(self.roi, self.config)

            # 4. Initialize Signal Processor
            self.signal_processor = SignalProcessor(self.video_input.get_fps(), self.config)

            # 5. Initialize Adaptive Controller (if enabled)
            if self.config.get('USE_ADAPTIVE_CONTROL', False):
                self.adaptive_controller = AdaptiveController(self.config)
                # Sync controller's initial state with the detector's state
                self.adaptive_controller.current_method = self.motion_detector.method

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
                   'error': str (optional)}
                  Returns {'success': False} if not initialized or error occurs.
        """
        if not self.is_initialized or self.video_input is None:
            return {'success': False, 'error': 'Monitor not initialized'}

        try:
            # 1. Get Frame
            success, frame = self.video_input.get_frame()
            if not success:
                return {'success': False, 'error': 'Video ended or failed'}

            # Store the frame (convert UMat for storage/return if needed)
            try:
                 # Store as NumPy for easier handling outside this class
                 self.last_frame = frame.get() if isinstance(frame, cv2.UMat) else frame
            except Exception as e_get:
                 print(f"Warning: Could not get frame data for storage: {e_get}")
                 self.last_frame = None # Set to None if conversion fails

            # 2. Detect Motion
            # Ensure motion_detector exists (should always if initialized)
            if self.motion_detector is None:
                 return {'success': False, 'error': 'Motion detector not initialized'}
            motion_value = self.motion_detector.process_frame(frame)

            # 3. Process Signal
            # Ensure signal_processor exists
            if self.signal_processor is None:
                 return {'success': False, 'error': 'Signal processor not initialized'}
            self.signal_processor.process_signal(motion_value)
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
                # Optionally return more detailed data if needed by caller
                # 'filtered_signal_buffer': filtered_signal,
                # 'peaks': peaks,
            }
            return results

        except Exception as e:
             print(f"Unexpected error during run_cycle: {e}")
             traceback.print_exc()
             return {'success': False, 'error': f'Runtime error: {e}'}


    def trigger_adaptation(self, fps):
         """Externally trigger the adaptive controller logic, passing current FPS."""
         if self.adaptive_controller and self.motion_detector:
              # Get necessary metrics (some might need to be stored if not readily available)
              bpm_stability = self.signal_processor.get_bpm_stability() if self.signal_processor else 0.0
              tracked_points = self.motion_detector.get_tracked_points_count()

              # Update controller's monitor state
              self.adaptive_controller.monitor(fps, bpm_stability, tracked_points)
              # Trigger adaptation logic
              self.adaptive_controller.adapt(self.motion_detector)

    def get_current_method(self):
        """Returns the currently active motion detection method."""
        return self.motion_detector.method if self.motion_detector else "N/A"

    def stop(self):
        """Releases resources, particularly the video input."""
        print("Stopping Respiration Monitor...")
        if self.video_input:
            self.video_input.release()
        self.is_initialized = False
        print("Respiration Monitor stopped.")

# (No test block needed here, main.py handles execution)

