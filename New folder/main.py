# main.py
# Runs the RespirationMonitor with automatic calibration, displays UI, and sends data via OSC.
# v10: Final version for modular structure with automatic calibration.
# MODIFIED: Fixed UnboundLocalError for start_loop_time

import time
import numpy as np
import cv2 # Needed for UI display frame conversion
import matplotlib.pyplot as plt # Needed for UI
import matplotlib.patches as patches # Needed for UI
import traceback # For printing full tracebacks
import os # Import os for path joining

# --- OSC Library Import ---
try:
    from pythonosc import udp_client
    from pythonosc import osc_message_builder
    osc_available = True
except ImportError:
    print("Warning: 'python-osc' library not found. OSC output will be disabled.")
    print("Install it using: pip install python-osc")
    osc_available = False
    # Define dummy classes/functions if library is missing to avoid NameErrors later
    class DummyOSCClient:
        def send_message(self, address, value): pass
    # Assign the dummy class to the name expected by the code
    udp_client.SimpleUDPClient = DummyOSCClient # Assign to the specific class used later

# --- Project Module Imports ---
try:
    # Assuming src directory is in the same folder or Python path
    from src.monitor import RespirationMonitor
    # Import the utility function if needed elsewhere, or just rely on monitor using it
    # from src.utils import calculate_weighted_roi_average
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    print("Ensure src/__init__.py exists and all src/*.py files are present.")
    exit() # Exit if the core module cannot be imported

# --- Configuration File Selection ---
# Specify the config file to load from the 'configs' folder
config_filename = "profile_1.json" # Or choose another config file

# Construct the full path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory where main.py is located
# Assume profiles are within a 'configs' folder
CONFIG_FOLDER = "profiles" # Define the config folder name
CONFIG_FILEPATH = os.path.join(script_dir, CONFIG_FOLDER, config_filename)


# --- OSC Configuration ---
OSC_CONFIG = {
    'ENABLE_OSC': True,
    'OSC_IP_ADDRESS': "127.0.0.1", # Target IP (Ableton/M4L machine)
    'OSC_PORT': 9001,              # Target Port
    'OSC_ADDRESS_SIGNAL': "/respmon/signal", # Address for normalized signal
    'OSC_ADDRESS_BPM': "/respmon/bpm",       # Address for BPM
    'OSC_ADDRESS_STATUS': "/respmon/status"  # Address for status messages
}

# --- UI Configuration ---
UI_CONFIG = {
    'ENABLE_UI': True # Set to False to disable the Matplotlib window
}


# --- UIManager Class (Keep as is from original) ---
class UIManager:
    """Handles basic UI display using Matplotlib."""
    def __init__(self):
        """Initializes the Matplotlib figure and axes."""
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        # Use a unique window title if multiple instances might run
        self.fig.canvas.manager.set_window_title(f'Respmon Enhanced (Modular) - {os.getpid()}')


        # Video display subplot
        self.ax_video = self.axs[0]
        self.video_im = self.ax_video.imshow(np.zeros((100, 100, 3), dtype=np.uint8)) # Placeholder image
        self.ax_video.set_title("Video Feed")
        self.ax_video.axis('off') # Hide axes ticks
        self.roi_rect_patch = None # Placeholder for ROI rectangle patch
        # Text overlay for status information
        self.status_text = self.ax_video.text(0.02, 0.95, 'Status: Initializing', color='white',
                                               fontsize=10, va='top', ha='left',
                                               transform=self.ax_video.transAxes, # Position relative to axes
                                               bbox=dict(facecolor='black', alpha=0.5, pad=0.2))

        # Signal plot subplot
        self.ax_plot = self.axs[1]
        self.line, = self.ax_plot.plot([], [], 'b-') # Signal line object (initially empty)
        self.peaks_scatter = self.ax_plot.scatter([], [], c='r', marker='x', s=50, zorder=5) # Peak markers (initially empty, ensure they are drawn on top)
        self.ax_plot.set_title("Motion Signal (Filtered & Normalized)") # Clarify plot content
        self.ax_plot.set_xlabel("Sample Index")
        self.ax_plot.set_ylabel("Value") # Generic label as normalization changes scale

        self.fig.tight_layout(pad=1.5) # Adjust layout
        plt.ion() # Interactive mode ON - allows updating plot without blocking
        plt.show() # Display the window
        self._is_figure_open = True
        # Connect the close event to a handler function
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
        self.last_plot_signal = None # Store the data used for the last plot

    def _handle_close(self, evt):
        """Callback function for when the plot window is closed by the user."""
        print("UI window closed by user.")
        self._is_figure_open = False

    def is_open(self):
        """Check if the UI window is still open."""
        # Check both the internal flag and if the figure actually exists
        # This handles cases where the window might be closed externally
        return self._is_figure_open and plt.fignum_exists(self.fig.number)

    def display_frame(self, frame, roi=None):
        """Displays the video frame and ROI rectangle in the top subplot."""
        if not self.is_open() or frame is None: return # Don't update if closed or no frame

        # Ensure frame is NumPy array (it should be from monitor results)
        frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame
        if frame_np is None or frame_np.size == 0: return # Check for empty frame

        try:
            # Convert BGR (OpenCV default) to RGB (Matplotlib default)
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                 frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            elif len(frame_np.shape) == 2: # Handle grayscale frames if they occur
                 frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
            else:
                 print(f"Warning: Unexpected frame shape for display: {frame_np.shape}")
                 return

            # Update the image data
            self.video_im.set_data(frame_rgb)
            # Update the image extent to match frame dimensions
            h, w = frame_rgb.shape[:2]
            self.video_im.set_extent([0, w, h, 0])

            # Draw or update ROI rectangle
            if roi:
                x, y, w_roi, h_roi = roi
                # Basic validation of ROI dimensions
                if w_roi > 0 and h_roi > 0:
                    if self.roi_rect_patch: # If patch exists, update its position/size
                        self.roi_rect_patch.set_bounds(x, y, w_roi, h_roi)
                        if not self.roi_rect_patch.axes: # Re-add if removed somehow
                             self.ax_video.add_patch(self.roi_rect_patch)
                    else: # Otherwise, create a new patch
                        self.roi_rect_patch = patches.Rectangle((x, y), w_roi, h_roi, linewidth=1, edgecolor='lime', facecolor='none')
                        self.ax_video.add_patch(self.roi_rect_patch) # Add patch to the axes
                elif self.roi_rect_patch: # Invalid ROI dims, remove patch if exists
                     self.roi_rect_patch.remove()
                     self.roi_rect_patch = None
            elif self.roi_rect_patch: # If no ROI provided but patch exists, remove it
                self.roi_rect_patch.remove()
                self.roi_rect_patch = None

            # Ensure axes limits match the frame size
            self.ax_video.set_xlim(0, w)
            self.ax_video.set_ylim(h, 0) # Flipped y-axis for image display
        except cv2.error as e: # Catch potential OpenCV errors (e.g., during color conversion)
            print(f"Warning: OpenCV error during UI frame display: {e}")
        except Exception as e: # Catch other potential errors
            print(f"Warning: Error during UI frame display: {e}")
            # traceback.print_exc() # Uncomment for debugging

    def display_plot(self, signal_data, peaks=None):
        """Updates the motion signal plot (expects filtered signal)."""
        if not self.is_open(): # Don't update if closed
            self.last_plot_signal = None; return
        if signal_data is None or len(signal_data) == 0: # If no signal data
            self.line.set_data([], []); self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear plot
            self.last_plot_signal = None; return

        try:
            # Ensure signal_data is a NumPy array for processing
            signal_array = np.asarray(signal_data)
            # Remove NaNs or Infs which can break plotting/normalization
            finite_signal = signal_array[np.isfinite(signal_array)]
            if len(finite_signal) < 2: # Need at least 2 points to normalize/plot
                 self.line.set_data([], []); self.peaks_scatter.set_offsets(np.empty((0, 2)))
                 self.last_plot_signal = None; return

            # Normalize signal for consistent plotting scale
            signal_mean = np.mean(finite_signal); signal_std = np.std(finite_signal)
            if signal_std > 1e-6: # Avoid division by zero if signal is flat
                # Normalize the original signal_array using stats from finite values
                plot_signal = (signal_array - signal_mean) / (3 * signal_std)
                # Replace non-finite values with NaN *after* normalization so they don't plot
                plot_signal[~np.isfinite(signal_array)] = np.nan
            else:
                plot_signal = np.zeros_like(signal_array) # Plot zeros if signal is flat

            self.last_plot_signal = plot_signal # Store the potentially normalized signal with NaNs
            t = np.arange(len(plot_signal)) # X-axis (sample index)
            self.line.set_data(t, plot_signal) # Update line data (NaNs create gaps)

            # Update peak markers
            if peaks is not None and len(peaks) > 0:
                peak_indices = np.asarray(peaks)
                # Ensure peaks are within current signal bounds and correspond to finite values
                valid_peak_mask = (peak_indices >= 0) & (peak_indices < len(plot_signal)) & np.isfinite(plot_signal[peak_indices])
                valid_peaks = peak_indices[valid_peak_mask]

                if len(valid_peaks) > 0:
                    peak_values_normalized = plot_signal[valid_peaks] # Get normalized values at peak locations
                    self.peaks_scatter.set_offsets(np.c_[valid_peaks, peak_values_normalized]) # Set scatter plot points
                else: self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear if no valid finite peaks
            else: self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear if no peaks detected

            # Adjust plot limits
            self.ax_plot.relim() # Recalculate limits based on new data
            self.ax_plot.autoscale_view(scalex=True, scaley=False) # Autoscale x-axis only
            # Set fixed y-limits for normalized data stability
            min_val = np.nanmin(plot_signal) if np.any(np.isfinite(plot_signal)) else -1
            max_val = np.nanmax(plot_signal) if np.any(np.isfinite(plot_signal)) else 1
            y_margin = max(0.5, (max_val - min_val) * 0.1) # Add some margin
            self.ax_plot.set_ylim(min_val - y_margin, max_val + y_margin)


        except Exception as e: # Catch errors during plotting
            print(f"Warning: Error during UI plot display: {e}")
            # traceback.print_exc() # Uncomment for debugging
            self.line.set_data([], []); self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear plot on error

    def display_status(self, bpm, fps, method):
        """Updates the status text display on the video feed."""
        if not self.is_open(): return
        bpm_str = f"{bpm:.1f}" if bpm is not None and bpm > 0 else "--" # Format BPM
        fps_str = f"{fps:.1f}" if fps is not None else "--"
        method_str = str(method) if method else "N/A"
        status = f"BPM: {bpm_str} | FPS: {fps_str} | Method: {method_str}"
        self.status_text.set_text(status) # Update text object

    def update_ui(self):
        """Redraws the UI and processes events."""
        if not self.is_open(): return False
        try:
            # self.fig.canvas.draw_idle() # Schedule a redraw efficiently
            self.fig.canvas.flush_events() # Process pending events (like close)
            plt.pause(0.001) # Very short pause to allow plot to redraw and events to process
            return self.is_open() # Return current status after processing events
        except Exception as e: # Catch errors during UI update/pause
            # Check if the error indicates the window was likely closed
            # These error messages can vary depending on the Matplotlib backend (TkAgg, QtAgg, etc.)
            if not plt.fignum_exists(self.fig.number):
                 print("UI window seems closed.")
                 self._is_figure_open = False
                 return False
            else:
                 # If the window still exists but another error occurred, log it but continue if possible
                 print(f"Warning: Error updating UI, but window still exists: {e}")
                 # traceback.print_exc() # Uncomment for debugging
                 return True # Assume we can continue if window is technically open

    def close(self):
        """Closes the Matplotlib window programmatically."""
        if self.is_open():
            plt.close(self.fig)
            self._is_figure_open = False
            print("UI closed programmatically.")

# --- Main Application Class ---
class MainApplication:
    def __init__(self):
        """Initializes the main application components."""
        self.monitor = None
        self.osc_client = None
        self.ui_manager = None
        self.last_loop_time = time.time()
        self.current_fps = 0.0
        self.running = False

    def _initialize_osc(self):
        """Initializes the OSC client based on configuration."""
        if OSC_CONFIG.get('ENABLE_OSC', False) and osc_available:
            ip = OSC_CONFIG.get('OSC_IP_ADDRESS', "127.0.0.1")
            port = OSC_CONFIG.get('OSC_PORT', 9001)
            try:
                # Ensure the correct class name is used based on import logic
                if hasattr(udp_client, 'SimpleUDPClient'):
                     self.osc_client = udp_client.SimpleUDPClient(ip, port)
                else: # Fallback if only the dummy was available
                     self.osc_client = udp_client.SimpleUDPClient() # Instantiate dummy

                # Check if it's not the dummy client before sending
                if not isinstance(self.osc_client, DummyOSCClient):
                     print(f"OSC Client initialized. Sending to {ip}:{port}")
                     # Send connection status message
                     self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "connected")
                else:
                     print("OSC Client is a dummy instance (python-osc likely missing).")
                     self.osc_client = None # Set to None if it's the dummy

            except Exception as e:
                print(f"Error initializing OSC client: {e}")
                self.osc_client = None
                print("Warning: OSC output disabled due to initialization error.")
        elif not osc_available:
            print("OSC output disabled: python-osc library missing.")
        else:
            print("OSC output disabled in configuration.")

    def _initialize_ui(self):
        """Initializes the UI Manager if enabled in configuration."""
        if UI_CONFIG.get('ENABLE_UI', False):
            print("Initializing UI...")
            try:
                self.ui_manager = UIManager()
            except Exception as e:
                print(f"Error initializing UI Manager: {e}")
                traceback.print_exc() # Print traceback for UI errors
                print("Warning: UI display disabled.")
                self.ui_manager = None # Ensure it's None if init fails
        else:
            print("UI display disabled in configuration.")

    def _send_osc_data(self, bpm, signal_value):
        """Sends BPM and normalized signal data via OSC."""
        # Check if OSC is enabled and client is valid (and not the dummy)
        if not self.osc_client or not OSC_CONFIG.get('ENABLE_OSC', False):
            return
        try:
            # Ensure values are standard Python floats for OSC library
            bpm_float = float(bpm) if bpm is not None and np.isfinite(bpm) else 0.0
            signal_float = float(signal_value) if signal_value is not None and np.isfinite(signal_value) else 0.0

            # Send messages to configured addresses
            self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_BPM', "/respmon/bpm"), bpm_float)
            self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_SIGNAL', "/respmon/signal"), signal_float)
        except Exception as e:
            print(f"Error sending OSC message: {e}")
            # Optional: Consider disabling OSC after repeated errors?
            # self.osc_client = None

    def _calculate_fps(self):
        """Calculates and updates the current frames per second."""
        current_time = time.time()
        delta_time = current_time - self.last_loop_time
        self.last_loop_time = current_time
        # Avoid division by zero and handle potential initial zero delta_time
        if delta_time > 1e-6: # Use a small threshold instead of zero
            instant_fps = 1.0 / delta_time
            # Simple low-pass filter for FPS smoothing
            self.current_fps = 0.9 * self.current_fps + 0.1 * instant_fps
        # else: FPS remains at previous value if delta_time is too small

    def run(self):
        """Starts the monitor and runs the main processing loop."""
        self.running = True
        print("Starting Main Application...")
        # --- FIX: Initialize variables used in finally block ---
        start_loop_time = None # Initialize to None
        frame_count = 0      # Initialize frame_count

        try:
            # --- Initialize Monitor ---
            print(f"Loading core configuration from: {CONFIG_FILEPATH}")
            if not os.path.exists(CONFIG_FILEPATH):
                 print(f"ERROR: Config file not found: {CONFIG_FILEPATH}")
                 self.running = False; return

            self.monitor = RespirationMonitor(config_filepath=CONFIG_FILEPATH)
            if not self.monitor.initialize():
                print("Failed to initialize Respiration Monitor. Exiting.")
                self.running = False; return

            # --- Initialize OSC and UI ---
            self._initialize_osc()
            self._initialize_ui()
            print("Starting main processing loop (Press Ctrl+C or close UI window to exit)...")
            # --- FIX: Assign start_loop_time only if initialization succeeds ---
            start_loop_time = time.time() # Assign actual start time here
            self.last_loop_time = start_loop_time

            # --- Main Loop ---
            while self.running and (self.ui_manager is None or self.ui_manager.is_open()):
                self._calculate_fps() # Update FPS calculation
                results = self.monitor.run_cycle()

                if not results['success']:
                    print(f"Monitor cycle failed: {results.get('error', 'Unknown')}")
                    if results.get('error') == 'Video ended or failed': self.running = False
                    time.sleep(0.1); continue

                bpm = results['bpm']
                normalized_signal = results['normalized_signal']
                frame = results['frame']
                roi = results['roi']
                method = results['method']

                if hasattr(self.monitor, 'adaptive_controller') and self.monitor.adaptive_controller:
                    self.monitor.trigger_adaptation(self.current_fps)

                self._send_osc_data(bpm, normalized_signal)

                if self.ui_manager:
                    self.ui_manager.display_frame(frame, roi)
                    filtered_signal_buffer = []
                    peaks = []
                    if hasattr(self.monitor, 'signal_processor'):
                         filtered_signal_buffer = self.monitor.signal_processor.get_filtered_signal_history()
                         peaks = self.monitor.signal_processor.get_last_detected_peaks()

                    self.ui_manager.display_plot(filtered_signal_buffer, peaks)
                    self.ui_manager.display_status(bpm, self.current_fps, method)
                    if not self.ui_manager.update_ui():
                        self.running = False

                frame_count += 1
                # Optional: Print status less frequently
                # if frame_count % 60 == 0:
                #     print(f"FPS: {self.current_fps:.1f} | BPM: {bpm:.1f} | NormSignal: {normalized_signal:.3f} | Method: {self.monitor.get_current_method()}")

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Stopping...")
            self.running = False
        except Exception as e: # Catch unexpected errors in the main loop
            print(f"\n--- An error occurred in main loop ---")
            traceback.print_exc()
            print(f"Error: {e}")
            self.running = False # Stop on unexpected error
        finally:
            # --- Cleanup ---
            end_loop_time = time.time()
            # --- FIX: Check if start_loop_time was set before using it ---
            if start_loop_time is not None:
                 total_time = end_loop_time - start_loop_time
                 if total_time > 0 and frame_count > 0:
                      print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds. Average FPS: {frame_count / total_time:.2f}")
                 elif frame_count > 0:
                      print(f"\nProcessed {frame_count} frames.")
                 else:
                      print("\nNo frames processed.")
            else:
                 print("\nProcessing loop did not start.")
            # --- End FIX ---

            print("Cleaning up resources...")
            if self.monitor: self.monitor.stop() # Stop the monitor (releases video)
            if self.ui_manager and self.ui_manager.is_open(): self.ui_manager.close() # Close UI window if open
            if self.osc_client and osc_available: # Send disconnect message if OSC was used
                try: self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "disconnected")
                except Exception: pass # Ignore errors on exit
            cv2.destroyAllWindows() # Close any remaining OpenCV windows
            print("Main application finished.")

# --- Entry Point ---
if __name__ == "__main__":
    # Check if config file exists before starting
    if not os.path.exists(CONFIG_FILEPATH):
         print(f"FATAL ERROR: Configuration file not found at '{CONFIG_FILEPATH}'")
         print("Please ensure the file exists and the path is correct.")
    else:
         app = MainApplication()
         app.run()
