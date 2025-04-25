# src/adaptive_control.py
# Contains the AdaptiveController class for dynamic parameter/method adjustments.

import time # Potentially useful for time-based stability checks later
import numpy as np # Needed for np.std if stability metrics evolve

# Note: This class interacts with a MotionDetector object.
# Note: Uses config dictionary passed during initialization.

class AdaptiveController:
    """Dynamically adjusts parameters based on performance and quality.
       Currently implements basic method switching logic.
    """
    def __init__(self, config):
        """Initializes AdaptiveController.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.frames_since_last_switch = 0
        # Get initial method safely from config
        self.current_method = config.get('DEFAULT_MOTION_METHOD', 'ROI_Average')
        # Get adaptive parameters safely from config
        self.hysteresis_frames = config.get('ADAPTIVE_HYSTERESIS_FRAMES', 60)
        self.fps_threshold = config.get('ADAPTIVE_FPS_THRESHOLD', 15)
        self.stability_threshold = config.get('ADAPTIVE_BPM_STABILITY_THRESHOLD', 15)
        self.use_adaptive_control = config.get('USE_ADAPTIVE_CONTROL', False)

        # Internal state for monitoring (can be expanded)
        self.current_fps = 0.0
        self.current_bpm_stability = 0.0
        self.current_tracked_points = 0

        if self.use_adaptive_control:
            print("AdaptiveController initialized and enabled.")
        else:
            print("AdaptiveController initialized but disabled in config.")


    def monitor(self, fps, bpm_stability, tracked_points):
        """Stores current performance metrics for decision making."""
        self.current_fps = fps
        self.current_bpm_stability = bpm_stability if np.isfinite(bpm_stability) else 0.0 # Ensure stability is finite
        self.current_tracked_points = tracked_points # Relevant for OpticalFlow
        self.frames_since_last_switch += 1

    def adapt(self, motion_detector):
        """Decides whether to switch methods based on monitored metrics
           and applies the change via the motion_detector object.
        """
        # Do nothing if disabled in config or if within hysteresis period
        if not self.use_adaptive_control or self.frames_since_last_switch < self.hysteresis_frames:
            return

        switched = False

        # --- Logic to switch TO ROI_Average (if currently using OpticalFlow) ---
        if self.current_method == 'OpticalFlow':
            # Reason 1: FPS is too low
            if self.current_fps < self.fps_threshold:
                print(f"Adaptive Switch Trigger: FPS ({self.current_fps:.1f}) < Threshold ({self.fps_threshold}). Switching to ROI_Average.")
                motion_detector.set_method('ROI_Average')
                self.current_method = 'ROI_Average'
                switched = True
            # Add other potential reasons to switch away from OpticalFlow here
            # Example: Consistently low tracked points (might need averaging over time)
            # elif self.current_tracked_points < config.get('ADAPTIVE_MIN_TRACKED_POINTS', 10):
            #    print(f"Adaptive Switch Trigger: Low tracked points...")
            #    motion_detector.set_method('ROI_Average')
            #    self.current_method = 'ROI_Average'
            #    switched = True

        # --- Logic to switch TO OpticalFlow (if currently using ROI_Average) ---
        elif self.current_method == 'ROI_Average':
            # Reason 1: Signal quality (BPM stability) is poor AND FPS allows it
            # Check if stability is valid (not 0) before comparing
            if self.current_bpm_stability > 0 and self.current_bpm_stability > self.stability_threshold:
                # Check if FPS has enough headroom (e.g., 20% above threshold)
                if self.current_fps > self.fps_threshold * 1.2:
                     print(f"Adaptive Switch Trigger: BPM Stability ({self.current_bpm_stability:.1f}) > Threshold ({self.stability_threshold}) and FPS ({self.current_fps:.1f}) sufficient. Switching to OpticalFlow.")
                     motion_detector.set_method('OpticalFlow')
                     self.current_method = 'OpticalFlow'
                     switched = True
                # else: # Optional print if stability is bad but FPS is too low to switch
                #    print(f"Info: BPM Stability ({self.current_bpm_stability:.1f}) > Threshold but FPS ({self.current_fps:.1f}) too low to switch to OpticalFlow.")

        # Reset hysteresis counter if a switch occurred
        if switched:
            self.frames_since_last_switch = 0

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing AdaptiveController module...")

    # Mock config
    test_config = {
        'DEFAULT_MOTION_METHOD': 'ROI_Average',
        'USE_ADAPTIVE_CONTROL': True,
        'ADAPTIVE_FPS_THRESHOLD': 15,
        'ADAPTIVE_BPM_STABILITY_THRESHOLD': 10,
        'ADAPTIVE_HYSTERESIS_FRAMES': 5 # Short hysteresis for testing
    }

    # Mock MotionDetector with a set_method function
    class MockMotionDetector:
        def __init__(self, initial_method):
            self.method = initial_method
            print(f"MockDetector created with method: {self.method}")
        def set_method(self, method_name):
            print(f"MockDetector: Method changed to {method_name}")
            self.method = method_name

    controller = AdaptiveController(test_config)
    detector = MockMotionDetector(controller.current_method)

    print("\n--- Test Scenario 1: Low FPS using OpticalFlow ---")
    controller.current_method = 'OpticalFlow' # Simulate starting with OF
    detector.method = 'OpticalFlow'
    controller.frames_since_last_switch = 10 # Assume past hysteresis
    controller.monitor(fps=10, bpm_stability=5, tracked_points=50)
    controller.adapt(detector)
    print(f"Detector method after adapt: {detector.method}")

    print("\n--- Test Scenario 2: High BPM Stability using ROI_Average (FPS OK) ---")
    controller.current_method = 'ROI_Average' # Simulate switching back
    detector.method = 'ROI_Average'
    controller.frames_since_last_switch = 10 # Assume past hysteresis
    controller.monitor(fps=25, bpm_stability=15, tracked_points=0)
    controller.adapt(detector)
    print(f"Detector method after adapt: {detector.method}")

    print("\n--- Test Scenario 3: High BPM Stability using ROI_Average (FPS too low) ---")
    controller.current_method = 'ROI_Average' # Simulate staying ROI_Avg
    detector.method = 'ROI_Average'
    controller.frames_since_last_switch = 10 # Assume past hysteresis
    controller.monitor(fps=12, bpm_stability=15, tracked_points=0)
    controller.adapt(detector)
    print(f"Detector method after adapt: {detector.method}")

    print("\n--- Test Scenario 4: Inside Hysteresis Period ---")
    controller.current_method = 'ROI_Average' # Simulate staying ROI_Avg
    detector.method = 'ROI_Average'
    controller.frames_since_last_switch = 3 # Inside hysteresis (config is 5)
    controller.monitor(fps=25, bpm_stability=15, tracked_points=0)
    controller.adapt(detector)
    print(f"Detector method after adapt (should not change): {detector.method}")

