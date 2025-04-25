# main.py
# Runs the RespirationMonitor with automatic calibration, displays UI, and sends data via OSC.
# v10: Final version for modular structure with automatic calibration.

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
    udp_client = DummyOSCClient()

# Import the core monitor class
try:
    # Assuming src directory is in the same folder or Python path
    from src.monitor import RespirationMonitor
except ImportError as e:
    print(f"ERROR: Could not import RespirationMonitor from src.monitor: {e}")
    print("Ensure src/__init__.py exists and all src/*.py files are present.")
    exit() # Exit if the core module cannot be imported

# --- Configuration File Selection ---
# Specify the config file to load from the 'configs' folder
# You could make this a command-line argument later if needed
config_filename = "profile_1.json" # Or choose another config file
# config_filename = "config_opticalflow.json"
# config_filename = "config_robust.json"
# config_filename = "config_opticalflow_alt.json"
# config_filename = "config_balanced_roi.json"

# Construct the full path relative to this script's location
# Assumes 'configs' folder is in the same directory as main.py
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory where main.py is located
CONFIG_FILEPATH = os.path.join(script_dir, "profiles", config_filename)


# --- OSC Configuration ---
OSC_CONFIG = {
    'ENABLE_OSC': True,
    'OSC_IP_ADDRESS': "127.0.0.1", # Target IP (Ableton/M4L machine)
    'OSC_PORT': 9001,             # Target Port
    'OSC_ADDRESS_SIGNAL': "/respmon/signal", # Address for normalized signal
    'OSC_ADDRESS_BPM': "/respmon/bpm",       # Address for BPM
    'OSC_ADDRESS_STATUS': "/respmon/status" # Address for status messages
}

# --- UI Configuration ---
UI_CONFIG = {
    'ENABLE_UI': True # Set to False to disable the Matplotlib window
}


# --- UIManager Class ---
class UIManager:
    """Handles basic UI display using Matplotlib."""
    def __init__(self):
        """Initializes the Matplotlib figure and axes."""
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title('Respmon Enhanced (Modular)')

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
        self.peaks_scatter = self.ax_plot.scatter([], [], c='r', marker='x', s=50) # Peak markers (initially empty)
        self.ax_plot.set_title("Motion Signal")
        self.ax_plot.set_xlabel("Sample Index")
        self.ax_plot.set_ylabel("Motion Value (Normalized)")

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
        return self._is_figure_open and plt.fignum_exists(self.fig.number)

    def display_frame(self, frame, roi=None):
        """Displays the video frame and ROI rectangle in the top subplot."""
        if not self.is_open() or frame is None: return # Don't update if closed or no frame
        frame_np = frame # Assume frame is already NumPy from monitor results
        try:
            # Convert BGR (OpenCV default) to RGB (Matplotlib default)
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            # Update the image data
            self.video_im.set_data(frame_rgb)
            # Update the image extent to match frame dimensions
            h, w = frame_rgb.shape[:2]
            self.video_im.set_extent([0, w, h, 0])

            # Draw or update ROI rectangle
            if roi:
                x, y, w_roi, h_roi = roi
                if self.roi_rect_patch: # If patch exists, update its position/size
                    self.roi_rect_patch.set_bounds(x, y, w_roi, h_roi)
                else: # Otherwise, create a new patch
                    self.roi_rect_patch = patches.Rectangle((x, y), w_roi, h_roi, linewidth=1, edgecolor='lime', facecolor='none')
                    self.ax_video.add_patch(self.roi_rect_patch) # Add patch to the axes
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

    def display_plot(self, signal_data, peaks=None):
        """Updates the motion signal plot after normalizing the data."""
        if not self.is_open(): # Don't update if closed
            self.last_plot_signal = None; return
        if signal_data is None or len(signal_data) == 0: # If no signal data
            self.line.set_data([], []); self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear plot
            self.last_plot_signal = None; return
        try:
            # Normalize signal for consistent plotting scale
            signal_mean = np.mean(signal_data); signal_std = np.std(signal_data)
            if signal_std > 1e-6: # Avoid division by zero if signal is flat
                plot_signal = (signal_data - signal_mean) / (3 * signal_std) # Normalize to approx +/- 1 range (3 std devs)
                plot_signal = np.clip(plot_signal, -2, 2) # Clip extreme values for visualization
            else:
                plot_signal = np.zeros_like(signal_data) # Plot zeros if signal is flat

            self.last_plot_signal = plot_signal # Store the normalized signal
            t = np.arange(len(plot_signal)) # X-axis (sample index)
            self.line.set_data(t, plot_signal) # Update line data

            # Update peak markers
            if peaks is not None and len(peaks) > 0:
                valid_peaks = peaks[peaks < len(plot_signal)] # Ensure peaks are within current signal bounds
                if len(valid_peaks) > 0:
                     peak_values_normalized = plot_signal[valid_peaks] # Get normalized values at peak locations
                     # Filter out non-finite values that might occur
                     finite_peak_indices = valid_peaks[np.isfinite(peak_values_normalized)]
                     finite_peak_values = peak_values_normalized[np.isfinite(peak_values_normalized)]
                     if len(finite_peak_indices) > 0:
                         self.peaks_scatter.set_offsets(np.c_[finite_peak_indices, finite_peak_values]) # Set scatter plot points
                     else: self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear if no finite peaks
                else: self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear if no valid peaks
            else: self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear if no peaks detected

            # Adjust plot limits
            self.ax_plot.relim() # Recalculate limits based on new data
            self.ax_plot.autoscale_view() # Autoscale axes
            self.ax_plot.set_ylim(-2.5, 2.5) # Set fixed y-limits for normalized data stability
        except Exception as e: # Catch errors during plotting
            print(f"Warning: Error during UI plot display: {e}")
            self.line.set_data([], []); self.peaks_scatter.set_offsets(np.empty((0, 2))) # Clear plot on error

    def display_status(self, bpm, fps, method):
        """Updates the status text display on the video feed."""
        if not self.is_open(): return
        bpm_str = f"{bpm:.1f}" if bpm > 0 else "--" # Format BPM
        status = f"BPM: {bpm_str} | FPS: {fps:.1f} | Method: {method}"
        self.status_text.set_text(status) # Update text object

    def update_ui(self):
        """Redraws the UI and processes events."""
        if not self.is_open(): return False
        try:
            self.fig.canvas.flush_events() # Process pending events (like close)
            plt.pause(0.001) # Very short pause to allow plot to redraw
            return True
        except Exception as e: # Catch errors during UI update/pause
            # Check for specific errors indicating the window was closed externally
            if "invalid command name" in str(e) or \
               "application has been destroyed" in str(e) or \
               "FigureCanvasAgg" in str(type(e)): # Common error types when window closes
                 print("UI window seems closed.")
                 self._is_figure_open = False; return False
            else: # Handle other potential errors
                 if plt.fignum_exists(self.fig.number): return True # Ignore if window still exists
                 else: print(f"Error updating UI and window closed: {e}"); self._is_figure_open = False; return False

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
                if osc_available: self.osc_client = udp_client.SimpleUDPClient(ip, port)
                else: self.osc_client = None; return # Should not happen due to check, but safety
                print(f"OSC Client initialized. Sending to {ip}:{port}")
                # Send connection status message
                self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "connected")
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
        # Check if OSC is enabled and client is valid
        if not self.osc_client or not OSC_CONFIG.get('ENABLE_OSC', False) or not osc_available:
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
        # Avoid division by zero and use a simple moving average (optional)
        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            # Simple low-pass filter for FPS smoothing
            self.current_fps = 0.9 * self.current_fps + 0.1 * instant_fps
        # else: FPS remains at previous value if delta_time is zero

    # Removed _select_roi_manually method as we reverted to automatic calibration

    def run(self):
        """Starts the monitor and runs the main processing loop."""
        self.running = True
        print("Starting Main Application...")
        try:
            # --- Initialize Monitor ---
            print(f"Loading core configuration from: {CONFIG_FILEPATH}")
            self.monitor = RespirationMonitor(config_filepath=CONFIG_FILEPATH)
            # Initialize now calls automatic calibration internally
            if not self.monitor.initialize(): # No manual_roi argument needed
                print("Failed to initialize Respiration Monitor. Exiting.")
                self.running = False; return

            # --- Initialize OSC and UI ---
            self._initialize_osc()
            self._initialize_ui()
            print("Starting main processing loop (Press Ctrl+C or close UI window to exit)...")
            frame_count = 0; start_loop_time = time.time(); self.last_loop_time = start_loop_time

            # --- Main Loop ---
            # Loop continues as long as running flag is true AND (UI is disabled OR UI window is open)
            while self.running and (self.ui_manager is None or self.ui_manager.is_open()):
                self._calculate_fps() # Update FPS calculation

                # Run one cycle of the monitor
                results = self.monitor.run_cycle()

                # Check if monitor cycle was successful
                if not results['success']:
                    print(f"Monitor cycle failed: {results.get('error', 'Unknown')}")
                    # Stop if video ended, otherwise maybe just wait briefly
                    if results.get('error') == 'Video ended or failed': self.running = False
                    time.sleep(0.1); continue # Avoid busy-looping on errors

                # Extract results
                bpm = results['bpm']
                normalized_signal = results['normalized_signal']
                frame = results['frame'] # Get frame for UI display
                roi = results['roi']     # Get ROI for UI display
                method = results['method'] # Get current method for UI status

                # Trigger Adaptive Control (if enabled in monitor's config)
                if self.monitor.adaptive_controller:
                    self.monitor.trigger_adaptation(self.current_fps)

                # Send OSC Data
                self._send_osc_data(bpm, normalized_signal)

                # --- Update UI (if enabled) ---
                if self.ui_manager:
                    self.ui_manager.display_frame(frame, roi)
                    # Get the filtered signal buffer directly from the processor for plotting
                    # This ensures the plot shows the data used for peak detection
                    filtered_signal_buffer = self.monitor.signal_processor.last_valid_filtered_signal
                    peaks = results.get('peaks') # Get peaks if monitor returns them (currently it doesn't, but could)
                    self.ui_manager.display_plot(filtered_signal_buffer, peaks)
                    self.ui_manager.display_status(bpm, self.current_fps, method)
                    # update_ui() returns False if the window was closed by the user
                    if not self.ui_manager.update_ui():
                         self.running = False # Stop main loop if UI closed

                frame_count += 1
                # Optional: Print status less frequently to avoid spamming console
                if frame_count % 60 == 0:
                    print(f"FPS: {self.current_fps:.1f} | BPM: {bpm:.1f} | NormSignal: {normalized_signal:.3f} | Method: {self.monitor.get_current_method()}")

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
            end_loop_time = time.time(); total_time = end_loop_time - start_loop_time
            if total_time > 0 and frame_count > 0: print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds. Average FPS: {frame_count / total_time:.2f}")
            print("Cleaning up resources...")
            if self.monitor: self.monitor.stop() # Stop the monitor (releases video)
            # No temp_video_input to release anymore
            if self.ui_manager and self.ui_manager.is_open(): self.ui_manager.close() # Close UI window if open
            if self.osc_client and osc_available: # Send disconnect message if OSC was used
                 try: self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "disconnected")
                 except Exception: pass # Ignore errors on exit
            cv2.destroyAllWindows() # Close any remaining OpenCV windows (like debug windows if enabled)
            print("Main application finished.")

# --- Entry Point ---
if __name__ == "__main__":
    app = MainApplication()
    app.run()
