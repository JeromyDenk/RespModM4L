# src/signal_processing.py
# Contains the SignalProcessor class for filtering, peak detection, and BPM calculation.

import numpy as np
from scipy import signal as sp_signal
from scipy.optimize import curve_fit
import collections
import warnings
import traceback # For error printing

# Suppress RuntimeWarning from mean of empty slice etc.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Gaussian Function for Fitting ---
def gaussian(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting."""
    stddev = max(abs(stddev), 1e-6) # Avoid division by zero or negative stddev
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# --- Module: SignalProcessor ---
class SignalProcessor:
    """Filters motion signal, detects peaks, and calculates BPM."""
    def __init__(self, sampling_rate, config):
        """Initializes SignalProcessor.

        Args:
            sampling_rate (float): The sampling rate (FPS) of the input signal.
            config (dict): The configuration dictionary.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        self.config = config
        self.sampling_rate = sampling_rate
        self.buffer = collections.deque(maxlen=config.get('SIGNAL_BUFFER_SIZE', 300))
        self.filter_cutoff = config.get('LOWPASS_FILTER_CUTOFF_HZ', 2.0)
        self.filter_order = config.get('LOWPASS_FILTER_ORDER', 4)
        self.use_gaussian_fit = config.get('USE_GAUSSIAN_FIT', False)
        self.min_buffer_fill = max(int(sampling_rate * 1.5), config.get('SIGNAL_BUFFER_SIZE', 300) // 3) # Need ~1.5s or 1/3 buffer
        self.last_bpms = collections.deque(maxlen=10) # For stability calculation
        self.debug_prints = config.get('DEBUG_PRINT_VALUES', False)
        self.last_valid_filtered_signal = None # Store last good signal for OSC/plot fallback
        self.filter_sos = None # Initialize filter coefficients

        # Pre-calculate filter coefficients (Rec 5a)
        if self.filter_cutoff > 0 and self.filter_cutoff < self.sampling_rate / 2.0:
            try:
                self.filter_sos = sp_signal.butter(self.filter_order,
                                                   self.filter_cutoff,
                                                   btype='low',
                                                   fs=self.sampling_rate,
                                                   output='sos') # Use second-order sections for stability
                print(f"SignalProcessor initialized. Sample Rate: {self.sampling_rate:.2f} Hz, Filter: Lowpass Butterworth Order {self.filter_order} Cutoff {self.filter_cutoff} Hz")
            except ValueError as e: # Catch specific SciPy errors
                print(f"Error creating Butterworth filter: {e}. Check parameters (cutoff vs Nyquist).")
                print("Warning: Filtering disabled.")
        else:
             print(f"Warning: Invalid filter cutoff {self.filter_cutoff} Hz for sample rate {self.sampling_rate} Hz. Filtering disabled.")

    def process_signal(self, motion_value):
        """Adds a new motion value to the buffer."""
        # Check for non-finite values before appending
        if np.isfinite(motion_value):
            self.buffer.append(motion_value)
            # Debug Print 1: Check raw motion value
            if self.debug_prints and abs(motion_value) > 1000: # Arbitrary threshold for "large"
                 print(f"DEBUG: Large raw motion value received: {motion_value:.2e}")
        else:
            print(f"Warning: Received non-finite motion value ({motion_value}). Skipping.")
            # Optional: Append a default value? (e.g., 0 or last valid)
            # if len(self.buffer) > 0: self.buffer.append(self.buffer[-1])
            # else: self.buffer.append(0)

    def _fit_gaussian_to_peak(self, signal, peak_index, estimated_width):
        """Fits a Gaussian function to a window around the peak."""
        # Determine window size based on estimated width
        window_half_width = int(max(5, estimated_width * self.config.get('GAUSSIAN_FIT_WINDOW_FACTOR', 3) / 2))
        start = max(0, peak_index - window_half_width)
        end = min(len(signal), peak_index + window_half_width + 1)
        window_x = np.arange(start, end)
        window_y = signal[start:end]

        if len(window_x) < 3: return None # Need at least 3 points to fit

        # Initial guess for parameters
        peak_val = signal[peak_index]
        baseline = np.min(window_y) # Simple baseline estimate
        amp_guess = peak_val - baseline
        mean_guess = peak_index
        # Estimate stddev from width property if available and valid, else use estimate
        if estimated_width > 0:
             stddev_guess = estimated_width / 2.355 # FWHM to stddev approx
        else:
             stddev_guess = 3.0 # Default guess if width is invalid

        # Ensure guesses are reasonable
        if amp_guess <= 0: amp_guess = 1e-6 # Amplitude must be positive
        stddev_guess = max(0.5, stddev_guess) # Minimum stddev guess

        initial_guess = [amp_guess, mean_guess, stddev_guess]
        # Define bounds for fitting parameters
        bounds = ([0, start - 1, 0.1], [np.inf, end + 1, np.inf]) # Min amplitude=0, mean within window, min stddev=0.1

        try: # Catch specific exceptions for curve_fit
            params, covariance = curve_fit(gaussian, window_x, window_y - baseline, # Fit amplitude above baseline
                                           p0=initial_guess, bounds=bounds, maxfev=5000)
            amplitude, mean, stddev = params
            # Basic validation of fit result (Rec 5c - Basic Criteria)
            min_std = self.config.get('GAUSSIAN_FIT_MIN_STDDEV', 1.0)
            max_std = self.config.get('GAUSSIAN_FIT_MAX_STDDEV', 50.0)
            # Check if fitted stddev is within configured range
            if min_std <= abs(stddev) <= max_std:
                return mean # Return refined peak position (mean of the Gaussian)
            else:
                # Optional: print reason for rejection
                # print(f"Gaussian fit rejected for peak {peak_index}: stddev {stddev:.2f} out of range [{min_std}, {max_std}]")
                return None
        except RuntimeError: # curve_fit couldn't find parameters
            # print(f"Gaussian fit failed for peak {peak_index}: Could not find optimal parameters.")
            return None
        except ValueError as e: # e.g., incompatible shapes
            print(f"ValueError during Gaussian fit for peak {peak_index}: {e}")
            return None
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error during Gaussian fit for peak {peak_index}: {e}")
            return None

    def analyze_buffer(self):
        """Analyzes the current signal buffer to find peaks and BPM.

        Returns:
            tuple: (filtered_signal, bpm, peaks, normalized_signal_value)
                   normalized_signal_value is the latest value of the normalized signal (-2 to 2)
                   Returns (None, 0.0, [], 0.0) if analysis cannot be performed.
        """
        bpm = 0.0 # Default value
        peaks = np.array([]) # Default value
        filtered_signal = None # Default value
        normalized_signal_value = 0.0 # Default value

        # Check if buffer has enough data
        if len(self.buffer) < self.min_buffer_fill:
            # Provide normalized value based on last good signal if possible
            if self.last_valid_filtered_signal is not None and len(self.last_valid_filtered_signal) > 0:
                 last_signal = self.last_valid_filtered_signal
                 sig_mean = np.mean(last_signal)
                 sig_std = np.std(last_signal)
                 if sig_std > 1e-6: normalized_signal_value = np.clip((last_signal[-1] - sig_mean) / (3 * sig_std), -2, 2)
            return filtered_signal, bpm, peaks, normalized_signal_value

        signal_arr = np.array(self.buffer)

        # --- Input Signal Validation ---
        if not np.all(np.isfinite(signal_arr)):
            print("Warning: Non-finite values in buffer PRE-filter. Attempting to use finite subset.")
            finite_signal = signal_arr[np.isfinite(signal_arr)]
            if len(finite_signal) < self.min_buffer_fill:
                 print("  Not enough finite values remaining.")
                 return None, bpm, peaks, normalized_signal_value # Return defaults
            signal_arr = finite_signal # Use only finite values

        # --- Debug Print 2: Check max abs value before filtering ---
        max_abs_raw = np.max(np.abs(signal_arr)) if len(signal_arr) > 0 else 0
        if self.debug_prints and max_abs_raw > 1e6: # Check if magnitude seems excessive
            print(f"DEBUG: Max abs raw signal before filter: {max_abs_raw:.2e}")

        # --- Apply low-pass filter ---
        if self.filter_sos is not None:
            try:
                filtered_signal = sp_signal.sosfiltfilt(self.filter_sos, signal_arr)
                # --- Output Signal Validation ---
                if not np.all(np.isfinite(filtered_signal)):
                    print("Warning: Non-finite values detected AFTER filtering. Using raw signal for peak finding.")
                    filtered_signal = signal_arr # Fallback to unfiltered signal
                else:
                    # Debug Print 3: Check max abs value after filtering
                    max_abs_filtered = np.max(np.abs(filtered_signal))
                    if self.debug_prints and max_abs_filtered > 1e6:
                        print(f"DEBUG: Max abs signal AFTER filter: {max_abs_filtered:.2e} (Raw max was {max_abs_raw:.2e})")
                    # Check for unreasonably large values specifically
                    if max_abs_filtered > 1e10:
                        print(f"Warning: Extremely large values ({max_abs_filtered:.2e}) detected AFTER filtering. Using raw signal.")
                        filtered_signal = signal_arr # Fallback
            except Exception as e:
                print(f"Error applying filter: {e}. Using raw signal.")
                filtered_signal = signal_arr # Fallback
        else:
            filtered_signal = signal_arr # No filter applied

        # --- Ensure filtered_signal is still valid before proceeding ---
        if filtered_signal is None or not np.all(np.isfinite(filtered_signal)):
             print("Warning: Signal became invalid after filter/fallback. Skipping peak finding.")
             # Provide normalized value based on last good signal if possible
             if self.last_valid_filtered_signal is not None and len(self.last_valid_filtered_signal) > 0:
                 last_signal = self.last_valid_filtered_signal; sig_mean = np.mean(last_signal); sig_std = np.std(last_signal)
                 if sig_std > 1e-6: normalized_signal_value = np.clip((last_signal[-1] - sig_mean) / (3 * sig_std), -2, 2)
             return None, bpm, peaks, normalized_signal_value # Return defaults, but with potentially updated normalized value

        # Store the last known good signal (could be raw if filtering failed)
        self.last_valid_filtered_signal = filtered_signal.copy()

        # --- Calculate Normalized Value for OSC ---
        # Do this based on the current valid filtered_signal
        sig_mean = np.mean(filtered_signal)
        sig_std = np.std(filtered_signal)
        if sig_std > 1e-6:
            # Use the *last* point in the buffer for the current OSC value
            normalized_signal_value = (filtered_signal[-1] - sig_mean) / (3 * sig_std)
            normalized_signal_value = np.clip(normalized_signal_value, -2, 2) # Clip for range
        # else: normalized_signal_value remains 0.0

        # --- Find peaks (Rec 5b) ---
        try:
            # Calculate minimum distance based on max expected BPM
            max_freq_hz = self.config.get('MAX_EXPECTED_BPM', 120) / 60.0
            min_dist_samples = int((self.sampling_rate / max_freq_hz) * self.config.get('PEAK_FINDING_DISTANCE_FACTOR', 0.5))
            min_dist_samples = max(1, min_dist_samples) # Ensure distance is at least 1

            # Calculate prominence relative to signal range
            signal_range = np.ptp(filtered_signal) # Peak-to-peak range
            signal_range = max(signal_range, 1e-9) # Avoid zero range for flat signal
            prominence_val = max(1e-6, signal_range * self.config.get('PEAK_FINDING_PROMINENCE_FACTOR', 0.1)) # Avoid zero prominence

            # Find peaks, requesting width property
            peaks, properties = sp_signal.find_peaks(filtered_signal,
                                                     distance=min_dist_samples,
                                                     prominence=prominence_val,
                                                     width=(None, None))
        except ValueError as e_peaks: # Catch specific SciPy errors
            print(f"Error during peak finding (ValueError): {e_peaks}")
            # Return current signal state, but 0 BPM and empty peaks
            return filtered_signal, bpm, peaks, normalized_signal_value
        except Exception as e_peaks_other: # Catch other unexpected errors
            print(f"Error during peak finding (Exception): {e_peaks_other}")
            return filtered_signal, bpm, peaks, normalized_signal_value

        # Proceed only if enough peaks are found
        if len(peaks) < 2:
            return filtered_signal, bpm, peaks, normalized_signal_value

        # --- Optional Gaussian Fitting (Rec 5c) ---
        if self.use_gaussian_fit:
            valid_peaks_indices = []
            peak_widths = properties.get('widths') # Get widths calculated by find_peaks
            # Estimate width if find_peaks didn't provide it
            if peak_widths is None:
                 avg_interval = np.mean(np.diff(peaks)) if len(peaks) >=2 else min_dist_samples
                 estimated_width = max(3.0, avg_interval / 4.0) # Heuristic estimate
                 peak_widths = [estimated_width] * len(peaks) # Use same estimate for all
            else:
                 peak_widths = [max(1.0, w) for w in peak_widths] # Ensure widths are positive

            # Fit Gaussian to each detected peak
            for i, p in enumerate(peaks):
                width_for_peak = peak_widths[i]
                refined_pos = self._fit_gaussian_to_peak(filtered_signal, p, width_for_peak)
                # Keep original peak index if fit is valid
                if refined_pos is not None:
                     valid_peaks_indices.append(p)

            peaks = np.array(valid_peaks_indices) # Update peaks array
            # Check again if enough peaks remain after filtering
            if len(peaks) < 2:
                 return filtered_signal, bpm, peaks, normalized_signal_value

        # --- Calculate BPM (Rec 5b result) ---
        if len(peaks) >= 2:
            avg_interval_samples = np.mean(np.diff(peaks))
            if avg_interval_samples > 0:
                bpm = (self.sampling_rate / avg_interval_samples) * 60
                self.last_bpms.append(bpm) # Store for stability calculation
            # else: bpm remains 0.0
        # else: bpm remains 0.0

        return filtered_signal, bpm, peaks, normalized_signal_value

    def get_bpm_stability(self):
        """Calculates the standard deviation of the last few BPM values."""
        if len(self.last_bpms) < 5: # Need a few values for meaningful std dev
            return 0.0 # Return 0 if not enough data, avoids warnings
        return np.std(self.last_bpms)

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing SignalProcessor module...")
    # Create dummy config and data
    test_config = {
        'SIGNAL_BUFFER_SIZE': 100,
        'LOWPASS_FILTER_CUTOFF_HZ': 3.0,
        'LOWPASS_FILTER_ORDER': 4,
        'MAX_EXPECTED_BPM': 90,
        'PEAK_FINDING_DISTANCE_FACTOR': 0.5,
        'PEAK_FINDING_PROMINENCE_FACTOR': 0.2,
        'USE_GAUSSIAN_FIT': False,
        'DEBUG_PRINT_VALUES': True
    }
    sampling_rate = 30 # Hz
    processor = SignalProcessor(sampling_rate, test_config)

    # Generate a noisy sine wave simulating breathing
    duration = 10 # seconds
    num_samples = duration * sampling_rate
    time_vec = np.linspace(0, duration, num_samples)
    breathing_freq = 0.3 # Hz (18 BPM)
    clean_signal = 5 * np.sin(2 * np.pi * breathing_freq * time_vec)
    noise = np.random.normal(0, 1.5, num_samples) # Add some noise
    test_signal = clean_signal + noise

    # Process the signal chunk by chunk (simulate real-time)
    chunk_size = 10
    results = []
    for i in range(0, num_samples, chunk_size):
        chunk = test_signal[i:min(i + chunk_size, num_samples)]
        for val in chunk:
            processor.process_signal(val)
        # Analyze buffer periodically
        if i > processor.min_buffer_fill:
             filtered, bpm, peaks, norm_val = processor.analyze_buffer()
             if filtered is not None: # Check if analysis was successful
                  results.append({'bpm': bpm, 'norm': norm_val, 'peaks_count': len(peaks)})

    print("\nAnalysis Results (Sample):")
    if results:
        for i in range(0, len(results), len(results)//5): # Print a few samples
            print(f"  Time ~{i*chunk_size/sampling_rate:.1f}s: BPM={results[i]['bpm']:.1f}, NormVal={results[i]['norm']:.3f}, Peaks={results[i]['peaks_count']}")
    else:
        print("No analysis results generated (buffer might not have filled sufficiently).")

