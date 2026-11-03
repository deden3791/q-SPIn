#file for qkit FPGA measurements
import numpy as np
import qkit

from qkit.storage.store import Data
from qkit.measure.measurement_class import Measurement
from qkit.gui.plot import plot as qviewkit
from qkit.gui.notebook.Progress_Bar import Progress_Bar

import json

import copy
import logging

class time_trace:
    """QKit measurement workflow for an RFSoC4x2 time trace."""

    def __init__(
        self,
        rfsoc,
        exp_name="",
        sample=None,
        comment="",
        open_qviewkit=True,
    ):
        self.rfsoc = rfsoc
        self.exp_name = exp_name
        self.sample = sample
        self.comment = comment
        self.open_qviewkit = open_qviewkit

        self._measurement_object = Measurement()
        self._measurement_object.measurement_type = "time_trace"
        self._measurement_object.sample = sample

        self._qvk_process = None
        self.filepath = None

        

    
    def measure(self):
        """Acquire and save one time-domain I/Q trace."""

        import logging

        data = None
        flow_started = False
        measurement_succeeded = False

        try:
            qkit.flow.start()
            flow_started = True

            # Acquire using the existing hardware driver.
            samples = np.asarray(self.rfsoc.measure()).squeeze()
            time_axis = np.asarray(
                self.rfsoc.get_time_axis(), dtype=np.float64
            ).squeeze()

            if samples.ndim != 1 or time_axis.ndim != 1:
                raise ValueError(
                    f"Expected 1D arrays; samples={samples.shape}, "
                    f"time={time_axis.shape}"
                )

            if samples.size == 0 or samples.shape != time_axis.shape:
                raise ValueError(
                    f"Time/data shape mismatch: {time_axis.shape} "
                    f"versus {samples.shape}"
                )

            if not np.iscomplexobj(samples):
                raise ValueError(
                    "Expected complex samples from rfsoc.measure()"
                )

            I = samples.real.astype(np.float64)
            Q = samples.imag.astype(np.float64)

            frequency = self.rfsoc.get_frequency()

            if frequency is None:
                raise ValueError(
                    "RFSoC frequency is unknown; set the frequency "
                    "before acquiring a trace."
                )

            # Let QKit generate the filename and directory.
            filename_suffix = "RFSoC_time_trace"
            if self.exp_name:
                filename_suffix += "_" + self.exp_name

            data = Data(name=filename_suffix, mode="a")

            # Associate measurement metadata with the HDF5 file.
            self._measurement_object.uuid = data._uuid
            self._measurement_object.hdf_relpath = data._relpath
            self._measurement_object.instruments = (
                qkit.instruments.get_instrument_names()
            )
            self._measurement_object.save()

            measurement_ds = data.add_textlist("measurement")
            measurement_ds.append(
                self._measurement_object.get_JSON()
            )

            # Frequency coordinate.
            freq_ds = data.add_coordinate(
                "frequency",
                unit="Hz",
                comment="Configured RFSoC frequency",
            )
            freq_ds.add([frequency])

            # Time coordinate.
            time_ds = data.add_coordinate(
                "time",
                unit="s",
                comment="RFSoC acquisition time axis",
            )
            time_ds.add(time_axis)

            # I/Q traces against time.
            i_ds = data.add_value_vector(
                "i",
                x=time_ds,
                unit="a.u.",
                comment="In-phase signal",
            )
            q_ds = data.add_value_vector(
                "q",
                x=time_ds,
                unit="a.u.",
                comment="Quadrature signal",
            )

            i_ds.add(I)
            q_ds.add(Q)

            config = self.rfsoc.get_config()

            if config is not None:
                config_ds = data.add_textlist("rfsoc_config")
                config_ds.append(
                    json.dumps(config, indent=2)
                )

            if self.comment:
                data.add_comment(self.comment)

            data.hf.hf.attrs["default_ds"] = [
                "data0/i",
                "data0/q",
            ]

            self.filepath = data.get_filepath()
            data.flush()
            data.save_finished()

            measurement_succeeded = True

        except Exception:
            logging.exception(
                "RFSoC time-trace measurement failed. "
                "The acquisition or file saving did not complete."
            )
            raise

        finally:
            try:
                if data is not None:
                    data.close()
            except Exception:
                logging.exception(
                    "Failed to close the RFSoC HDF5 file."
                )
            finally:
                if flow_started:
                    qkit.flow.end()

        # Only open qviewkit after successful acquisition and saving.
        if measurement_succeeded and self.open_qviewkit:
            try:
                self._qvk_process = qviewkit.plot(
                    self.filepath,
                    datasets=["data0/i", "data0/q"],
                )
            except Exception:
                logging.exception(
                    "Measurement was saved, but qviewkit "
                    "could not be opened automatically."
                )
                print("Saved to:", self.filepath)

        return self.filepath



class frequency_sweep:
    """QKIT frequency sweep for an RFSoC4x2.

    Modes
    -----
    mean_iq:
        Save mean I and Q, magnitude of mean IQ, and phase of
        mean IQ versus frequency.

    full_trace:
        Save I, Q, amplitude, and phase as frequency-by-time
        matrices.
    """

    def __init__(
        self,
        rfsoc,
        start_frequency,
        stop_frequency,
        num_points,
        mode="mean_iq",
        exp_name="",
        sample=None,
        comment="",
        open_qviewkit=True,
    ):
        self.rfsoc = rfsoc
        self.start_frequency = float(start_frequency)
        self.stop_frequency = float(stop_frequency)
        self.num_points = int(num_points)
        self.mode = mode
        self.exp_name = exp_name
        self.sample = sample
        self.comment = comment
        self.open_qviewkit = open_qviewkit

        self.progress_bar = True
        self.progress = None

        if mode not in ("mean_iq", "full_trace"):
            raise ValueError(
                "mode must be 'mean_iq' or 'full_trace'"
            )

        if self.num_points < 1:
            raise ValueError(
                "num_points must be at least 1"
            )

        if not (
            np.isfinite(self.start_frequency)
            and np.isfinite(self.stop_frequency)
        ):
            raise ValueError(
                "Frequencies must be finite numbers"
            )

        if self.start_frequency <= 0 or self.stop_frequency <= 0:
            raise ValueError(
                "Frequencies must be positive"
            )

        if self.stop_frequency < self.start_frequency:
            raise ValueError(
                "stop_frequency must be >= start_frequency"
            )

        # QKIT instrument frequencies are in Hz.
        self.frequencies = np.linspace(
            self.start_frequency,
            self.stop_frequency,
            self.num_points,
        )

        # QKIT measurement metadata.
        self._measurement_object = Measurement()
        self._measurement_object.measurement_type = (
            "frequency_sweep"
        )
        self._measurement_object.sample = sample

        self._qvk_process = None
        self.filepath = None

    def measure(self):
        """Acquire a frequency sweep and save IQ, amplitude and phase."""

        data = None
        flow_started = False

        try:
            # Start the QKIT measurement flow.
            qkit.flow.start()
            flow_started = True

            # Initialize progress reporting.
            if self.progress_bar:
                self.progress = Progress_Bar(
                    self.num_points,
                    name="RFSoC frequency sweep",
                )

            # Obtain and validate the time axis.
            time_axis = np.asarray(
                self.rfsoc.get_time_axis(),
                dtype=np.float64,
            ).squeeze()

            if time_axis.ndim != 1 or time_axis.size == 0:
                raise ValueError(
                    f"Invalid time axis: {time_axis.shape}"
                )

            # Construct the measurement filename.
            filename = "RFSoC_frequency_sweep"

            if self.exp_name:
                filename += "_" + self.exp_name

            data = Data(name=filename, mode="a")

            # Save measurement metadata.
            self._measurement_object.uuid = data._uuid
            self._measurement_object.hdf_relpath = data._relpath
            self._measurement_object.instruments = (
                qkit.instruments.get_instrument_names()
            )
            self._measurement_object.measurement_func = (
                "frequency_sweep"
            )
            self._measurement_object.x_axis = "frequency"
            self._measurement_object.y_axis = (
                "time" if self.mode == "full_trace" else ""
            )
            self._measurement_object.save()

            measurement_ds = data.add_textlist("measurement")
            measurement_ds.append(
                self._measurement_object.get_JSON()
            )

            # Create the frequency coordinate.
            freq_ds = data.add_coordinate(
                "frequency",
                unit="Hz",
                comment="RFSoC sweep frequency",
            )
            freq_ds.add(self.frequencies)

            # Save the RFSoC configuration if available.
            config = getattr(self.rfsoc, "_config", None)

            if config is not None:
                config_ds = data.add_textlist("rfsoc_config")
                config_ds.append(
                    json.dumps(
                        config,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                )

            # Create datasets before acquisition begins.
            if self.mode == "full_trace":

                time_ds = data.add_coordinate(
                    "time",
                    unit="s",
                    comment="RFSoC acquisition time",
                )
                time_ds.add(time_axis)

                i_ds = data.add_value_matrix(
                    "i",
                    x=time_ds,
                    y=freq_ds,
                    unit="a.u.",
                    comment="In-phase signal",
                )

                q_ds = data.add_value_matrix(
                    "q",
                    x=time_ds,
                    y=freq_ds,
                    unit="a.u.",
                    comment="Quadrature signal",
                )

                amplitude_ds = data.add_value_matrix(
                    "amplitude",
                    x=time_ds,
                    y=freq_ds,
                    unit="a.u.",
                    comment="IQ magnitude",
                )

                phase_ds = data.add_value_matrix(
                    "phase",
                    x=time_ds,
                    y=freq_ds,
                    unit="rad",
                    comment="IQ phase",
                )

            else:

                i_ds = data.add_value_vector(
                    "i",
                    x=freq_ds,
                    unit="a.u.",
                    comment="Mean in-phase signal",
                )

                q_ds = data.add_value_vector(
                    "q",
                    x=freq_ds,
                    unit="a.u.",
                    comment="Mean quadrature signal",
                )

                amplitude_ds = data.add_value_vector(
                    "amplitude",
                    x=freq_ds,
                    unit="a.u.",
                    comment="Magnitude of mean IQ",
                )

                phase_ds = data.add_value_vector(
                    "phase",
                    x=freq_ds,
                    unit="rad",
                    comment="Phase of mean IQ",
                )

            # Set default datasets for QViewKit.
            data.hf.hf.attrs["default_ds"] = [
                "data0/i",
                "data0/q",
                "data0/amplitude",
                "data0/phase",
            ]

            # Resolve the filepath after creating the datasets.
            self.filepath = data.get_filepath()

            # Open QViewKit once, before the sweep.
            if self.open_qviewkit:
                try:
                    self._qvk_process = qviewkit.plot(
                        self.filepath,
                        datasets=[
                            "data0/i",
                            "data0/q",
                            "data0/amplitude",
                            "data0/phase",
                        ],
                        refresh=2,
                        live=True,
                    )

                    logging.info(
                        "Started QViewKit for RFSoC sweep: %s",
                        self.filepath,
                    )

                except Exception:
                    self._qvk_process = None
                    logging.exception(
                        "Could not start QViewKit; "
                        "continuing acquisition."
                    )

            # Acquire one trace at each frequency.
            for index, frequency in enumerate(self.frequencies):

                logging.info(
                    "RFSoC sweep point %d/%d: %.9f GHz",
                    index + 1,
                    self.num_points,
                    frequency / 1e9,
                )

                # Set frequency in Hz.
                self.rfsoc.set_frequency(
                    float(frequency)
                )

                # Acquire complex IQ samples.
                samples = np.asarray(
                    self.rfsoc.measure()
                ).squeeze()

                if samples.ndim != 1 or samples.size == 0:
                    raise ValueError(
                        f"Invalid IQ trace at {frequency} Hz: "
                        f"{samples.shape}"
                    )

                if not np.iscomplexobj(samples):
                    raise ValueError(
                        "Expected complex IQ samples from RFSoC"
                    )

                if samples.size != time_axis.size:
                    raise ValueError(
                        f"Trace length {samples.size} does not "
                        f"match time axis length {time_axis.size} "
                        f"at {frequency} Hz"
                    )

                # Separate the raw I and Q components.
                I = samples.real.astype(np.float64)
                Q = samples.imag.astype(np.float64)

                # Calculate magnitude and phase for every sample.
                amplitude = np.abs(samples).astype(np.float64)
                phase = np.angle(samples).astype(np.float64)

                # Save the data for this frequency.
                if self.mode == "full_trace":

                    # Append one complete row to each matrix.
                    i_ds.append(I)
                    q_ds.append(Q)
                    amplitude_ds.append(amplitude)
                    phase_ds.append(phase)

                else:

                    # Mean I and Q.
                    mean_i = float(np.mean(I))
                    mean_q = float(np.mean(Q))

                    i_ds.append(
                        np.asarray([mean_i])
                    )
                    q_ds.append(
                        np.asarray([mean_q])
                    )

                    # Magnitude and phase of the mean complex IQ.
                    mean_iq = complex(mean_i, mean_q)

                    mean_amplitude = float(np.abs(mean_iq))
                    mean_phase = float(np.angle(mean_iq))

                    amplitude_ds.append(
                        np.asarray([mean_amplitude])
                    )
                    phase_ds.append(
                        np.asarray([mean_phase])
                    )

                # Update progress after saving this frequency.
                if self.progress_bar:
                    self.progress.iterate(
                        param=f"{frequency / 1e9:.6f} GHz"
                    )

                # Preserve the existing QKIT flow behaviour.
                qkit.flow.sleep()

            # Save the optional measurement comment.
            if self.comment:
                data.add_comment(self.comment)

            # Flush and finalize the HDF5 file.
            data.flush()
            data.save_finished()

            print(
                "Saved frequency sweep:",
                self.filepath,
            )
            # print(
            #     "Frequencies (GHz):",
            #     self.frequencies / 1e9,
            # )

        except Exception:
            if self.progress_bar and self.progress is not None:
                try:
                    self.progress.abort()
                except Exception:
                    logging.exception(
                        "Could not abort the progress bar."
                    )

            logging.exception(
                "RFSoC frequency sweep failed."
            )
            raise

        finally:
            # Close the HDF5 file and end the QKIT flow.
            try:
                if data is not None:
                    data.close()
            except Exception:
                logging.exception(
                    "Failed to close RFSoC HDF5 file."
                )
            finally:
                if flow_started:
                    qkit.flow.end()

        return self.filepath


class current_sweep:
    """RFSoC frequency sweeps controlled by an external current sweep.

    The external instrument is controlled through x_set_obj, following
    the interface of QKIT's existing VNA spectrum class.

    Modes
    -----
    mean_iq:
        Save mean I, Q, amplitude and phase versus frequency at each
        current.

    full_trace:
        Save frequency-by-time matrices for I, Q, amplitude and phase
        at each current.

    The class never turns external instruments on or off.
    """

    def __init__(
        self,
        rfsoc,
        exp_name="",
        sample=None,
        comment="",
        mode="full_trace",
        open_qviewkit=True,
    ):
        self.rfsoc = rfsoc
        self.exp_name = exp_name
        self._sample = sample
        self.comment = comment

        if mode not in ("mean_iq", "full_trace"):
            raise ValueError(
                "mode must be 'mean_iq' or 'full_trace'"
            )

        self.mode = mode
        self.open_qviewkit = bool(open_qviewkit)

        # Outer sweep parameters.
        self.x_vec = None
        self.x_coordname = None
        self.x_unit = ""
        self.x_set_obj = None

        # Inner RFSoC frequency sweep.
        self.frequency_vec = None

        # Progress reporting.
        self.progress_bar = True
        self._progress = None

        # QViewKit settings.
        self.qviewkit_refresh = 2
        self._qvk_process = None

        # Optional delay after changing current.
        self.tdx = 0.0

        # Measurement file.
        self.dirname = None
        self.filepath = None
        self._data_file = None

        # RFSoC acquisition information.
        self._time_axis = None
        self._datasets = []
        self._heatmap_datasets = {}

        # QKIT measurement metadata.
        self._measurement_object = Measurement()
        self._measurement_object.measurement_type = "spectroscopy"
        self._measurement_object.sample = sample

    def set_x_parameters(
        self,
        x_vec,
        x_coordname,
        x_set_obj,
        x_unit="",
    ):
        """Configure the outer sweep, following QKIT's VNA interface."""

        x_vec = np.asarray(x_vec, dtype=np.float64)

        if not np.all(np.isfinite(x_vec)):
            raise ValueError("x_vec must contain only finite values")
        
        if x_vec.ndim != 1 or x_vec.size == 0:
            raise ValueError(
                "x_vec must be a non-empty one-dimensional array"
            )

        if not callable(x_set_obj):
            raise TypeError("x_set_obj must be callable")

        self.x_vec = x_vec.copy()
        self.x_coordname = str(x_coordname)
        self.x_unit = str(x_unit)
        self.x_set_obj = x_set_obj

    def set_frequency_parameters(
        self,
        start_frequency,
        stop_frequency,
        num_points,
    ):
        """Configure the inner RFSoC frequency sweep, in Hz."""

        start_frequency = float(start_frequency)
        stop_frequency = float(stop_frequency)
        num_points = int(num_points)

        if start_frequency <= 0 or stop_frequency <= 0:
            raise ValueError("Frequencies must be positive")

        if stop_frequency < start_frequency:
            raise ValueError(
                "stop_frequency must be >= start_frequency"
            )

        if num_points < 1:
            raise ValueError("num_points must be at least 1")

        if not (
            np.isfinite(start_frequency)
            and np.isfinite(stop_frequency)
        ):
            raise ValueError("Frequencies must be finite")

        self.frequency_vec = np.linspace(
            start_frequency,
            stop_frequency,
            num_points,
        )

    def _validate_parameters(self):

        if self.x_vec is None or self.x_set_obj is None:
            raise RuntimeError(
                "Call set_x_parameters() before measuring"
            )

        if self.frequency_vec is None:
            raise RuntimeError(
                "Call set_frequency_parameters() before measuring"
            )

        if not np.all(np.isfinite(self.x_vec)):
            raise ValueError(
                "Current sweep coordinates must be finite"
            )

        if not np.all(np.isfinite(self.frequency_vec)):
            raise ValueError(
                "Frequency sweep coordinates must be finite"
            )

        if not self.x_coordname:
            raise ValueError(
                "Current coordinate name cannot be empty"
            )
        
    def _prepare_measurement_file(self):
        """Create measurement metadata, coordinates and datasets."""

        filename = "RFSoC_current_frequency_sweep"

        if self.exp_name:
            filename += "_" + self.exp_name

        # Create the HDF5 file.
        self._data_file = Data(name=filename, mode="a")
        data = self._data_file

        # Measurement metadata.
        self._measurement_object.uuid = data._uuid
        self._measurement_object.hdf_relpath = data._relpath

        self._measurement_object.instruments = (
            qkit.instruments.get_instrument_names()
        )

        self._measurement_object.measurement_func = "measure_2D"
        self._measurement_object.x_axis = self.x_coordname
        self._measurement_object.y_axis = "frequency"
        self._measurement_object.z_axis = (
            "time" if self.mode == "full_trace" else ""
        )

        self._measurement_object.save()

        measurement_ds = data.add_textlist("measurement")
        measurement_ds.append(
            self._measurement_object.get_JSON()
        )

        # --------------------------------------------------
        # Coordinates
        # --------------------------------------------------

        current_ds = data.add_coordinate(
            self.x_coordname,
            unit=self.x_unit,
            comment="Outer current sweep coordinate",
        )
        current_ds.add(self.x_vec)

        frequency_ds = data.add_coordinate(
            "frequency",
            unit="Hz",
            comment="RFSoC frequency sweep",
        )
        frequency_ds.add(self.frequency_vec)

        # Retrieve and validate RFSoC time axis.
        time_axis = np.asarray(
            self.rfsoc.get_time_axis(),
            dtype=np.float64,
        ).squeeze()

        if (
            time_axis.ndim != 1
            or time_axis.size == 0
            or not np.all(np.isfinite(time_axis))
        ):
            raise ValueError(
                f"Invalid RFSoC time axis: {time_axis.shape}"
            )

        time_ds = data.add_coordinate(
            "time",
            unit="s",
            comment="RFSoC acquisition time",
        )
        time_ds.add(time_axis)

        self._time_axis = time_axis

        # --------------------------------------------------
        # RFSoC configuration
        # --------------------------------------------------

        config = self.rfsoc.get_config()

        if config is not None:
            config_ds = data.add_textlist("rfsoc_config")
            config_ds.append(
                json.dumps(
                    config,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )

        # --------------------------------------------------
        # Combined heatmap datasets
        #
        # X = frequency
        # Y = current
        #
        # Matrix shape:
        # (n_current, n_frequency)
        # --------------------------------------------------

        self._heatmap_datasets = {

            "i": data.add_value_matrix(
                "i_vs_current",
                x=frequency_ds,
                y=current_ds,
                unit="a.u.",
                comment="Mean I versus frequency and current",
            ),

            "q": data.add_value_matrix(
                "q_vs_current",
                x=frequency_ds,
                y=current_ds,
                unit="a.u.",
                comment="Mean Q versus frequency and current",
            ),

            "amplitude": data.add_value_matrix(
                "amplitude_vs_current",
                x=frequency_ds,
                y=current_ds,
                unit="a.u.",
                comment="Mean amplitude versus frequency and current",
            ),

            "phase": data.add_value_matrix(
                "phase_vs_current",
                x=frequency_ds,
                y=current_ds,
                unit="deg",
                comment="Circular mean phase versus frequency and current",
            ),
        }

        # --------------------------------------------------
        # Individual datasets for each current
        # --------------------------------------------------

        self._datasets = []

        for ix, current in enumerate(self.x_vec):

            suffix = f"{ix:04d}"

            if self.mode == "full_trace":

                # X = time, Y = frequency.
                datasets = {

                    "i": data.add_value_matrix(
                        f"i_{suffix}",
                        x=time_ds,
                        y=frequency_ds,
                        unit="a.u.",
                        comment=f"I at {current:g} {self.x_unit}",
                    ),

                    "q": data.add_value_matrix(
                        f"q_{suffix}",
                        x=time_ds,
                        y=frequency_ds,
                        unit="a.u.",
                        comment=f"Q at {current:g} {self.x_unit}",
                    ),

                    "amplitude": data.add_value_matrix(
                        f"amplitude_{suffix}",
                        x=time_ds,
                        y=frequency_ds,
                        unit="a.u.",
                        comment=f"Amplitude at {current:g} {self.x_unit}",
                    ),

                    "phase": data.add_value_matrix(
                        f"phase_{suffix}",
                        x=time_ds,
                        y=frequency_ds,
                        unit="deg",
                        comment=f"Phase at {current:g} {self.x_unit}",
                    ),
                }

            else:

                datasets = {

                    "i": data.add_value_vector(
                        f"i_{suffix}",
                        x=frequency_ds,
                        unit="a.u.",
                        comment=f"Mean I at {current:g} {self.x_unit}",
                    ),

                    "q": data.add_value_vector(
                        f"q_{suffix}",
                        x=frequency_ds,
                        unit="a.u.",
                        comment=f"Mean Q at {current:g} {self.x_unit}",
                    ),

                    "amplitude": data.add_value_vector(
                        f"amplitude_{suffix}",
                        x=frequency_ds,
                        unit="a.u.",
                        comment=f"Mean amplitude at {current:g} {self.x_unit}",
                    ),

                    "phase": data.add_value_vector(
                        f"phase_{suffix}",
                        x=frequency_ds,
                        unit="deg",
                        comment=f"Mean phase at {current:g} {self.x_unit}",
                    ),
                }

            self._datasets.append(datasets)

        # --------------------------------------------------
        # Default QViewKit datasets
        # --------------------------------------------------

        data.hf.hf.attrs["default_ds"] = [
            "data0/amplitude_vs_current",
            "data0/phase_vs_current",
        ]

        self.filepath = data.get_filepath()

        data.flush()

    def measure_2D(self, web_visible=True):
        """Run the outer current sweep and inner frequency sweep."""

        self._validate_parameters()

        flow_started = False
        measurement_succeeded = False

        # Reset file reference before starting.
        self._data_file = None
        self._progress = None
        self._qvk_process = None

        try:

            # Start QKIT measurement flow.
            qkit.flow.start()
            flow_started = True

            # Prepare measurement file.
            # The file handle is assigned inside this method,
            # allowing cleanup even if preparation fails.
            self._prepare_measurement_file()

            data = self._data_file

            # --------------------------------------------------
            # Open QViewKit once, before acquisition.
            # --------------------------------------------------

            if self.open_qviewkit:

                try:
                    self._qvk_process = qviewkit.plot(
                        self.filepath,
                        datasets=[
                            "data0/amplitude_vs_current",
                            "data0/phase_vs_current",
                        ],
                        refresh=self.qviewkit_refresh,
                        live=True,
                    )

                    logging.info(
                        "Started QViewKit: %s",
                        self.filepath,
                    )

                except Exception:
                    self._qvk_process = None

                    logging.exception(
                        "Could not start QViewKit. "
                        "Continuing measurement without viewer."
                    )

            # --------------------------------------------------
            # Initialize progress bar.
            # --------------------------------------------------

            if self.progress_bar:

                total_points = (
                    len(self.x_vec) * len(self.frequency_vec)
                )

                self._progress = Progress_Bar(
                    total_points,
                    name="RFSoC current/frequency sweep",
                )

            # --------------------------------------------------
            # Run measurement.
            # --------------------------------------------------

            self._measure()

            # --------------------------------------------------
            # Finalize successful measurement.
            # --------------------------------------------------

            if self.comment:
                data.add_comment(self.comment)

            data.flush()
            data.save_finished()

            measurement_succeeded = True

            print(
                "Saved current/frequency sweep:",
                self.filepath,
            )

        except Exception:

            logging.exception(
                "RFSoC current/frequency sweep failed."
            )

            if self._progress is not None:

                try:
                    self._progress.abort()

                except Exception:
                    logging.exception(
                        "Could not abort progress bar."
                    )

            raise

        finally:

            # --------------------------------------------------
            # Close HDF5 file, including preparation failures.
            # --------------------------------------------------

            if self._data_file is not None:

                try:
                    self._data_file.close()

                except Exception:
                    logging.exception(
                        "Failed to close measurement file."
                    )

            # --------------------------------------------------
            # End QKIT flow.
            # --------------------------------------------------

            if flow_started:

                try:
                    qkit.flow.end()

                except Exception:
                    logging.exception(
                        "Failed to end QKIT measurement flow."
                    )

        if measurement_succeeded:
            return self.filepath

    def _measure(self):
        """Set each current and acquire its complete frequency sweep."""

        for ix, current in enumerate(self.x_vec):

            logging.info(
                "Current point %d/%d: %g %s",
                ix + 1,
                len(self.x_vec),
                current,
                self.x_unit,
            )

            # --------------------------------------------------
            # Set external current
            # --------------------------------------------------

            result = self.x_set_obj(float(current))

            # The callback may raise its own exception.
            # Explicit failure returns are also handled here.
            if isinstance(result, (bool, np.bool_)) and not result:
                raise RuntimeError(
                    f"Current setter failed at {current} {self.x_unit}"
                )

            if isinstance(result, str):
                raise RuntimeError(
                    f"Current setter failed at {current} {self.x_unit}: "
                    f"{result}"
                )

            # Optional settling delay.
            if self.tdx > 0:
                qkit.flow.sleep(self.tdx)

            datasets = self._datasets[ix]

            # One value per frequency for the combined heatmaps.
            mean_i_row = []
            mean_q_row = []
            mean_amplitude_row = []
            mean_phase_row = []

            # --------------------------------------------------
            # Inner RFSoC frequency sweep
            # --------------------------------------------------

            for frequency in self.frequency_vec:

                logging.info(
                    "Current %g %s, frequency %.9f GHz",
                    current,
                    self.x_unit,
                    frequency / 1e9,
                )

                # Set RFSoC frequency.
                self.rfsoc.set_frequency(float(frequency))

                # Acquire complex IQ samples.
                samples = np.asarray(
                    self.rfsoc.measure()
                ).squeeze()

                # Validate IQ data.
                if samples.ndim != 1 or samples.size == 0:
                    raise ValueError(
                        f"Invalid IQ trace at {frequency} Hz: "
                        f"{samples.shape}"
                    )

                if not np.iscomplexobj(samples):
                    raise ValueError(
                        f"Expected complex IQ samples at {frequency} Hz"
                    )

                if not np.all(np.isfinite(samples)):
                    raise ValueError(
                        f"Non-finite IQ samples at {frequency} Hz"
                    )

                if samples.size != self._time_axis.size:
                    raise ValueError(
                        f"Trace length {samples.size} does not match "
                        f"time axis length {self._time_axis.size} "
                        f"at {frequency} Hz"
                    )

                # --------------------------------------------------
                # Calculate I, Q, amplitude and phase
                # --------------------------------------------------

                I = samples.real.astype(np.float64)
                Q = samples.imag.astype(np.float64)

                amplitude = np.abs(samples).astype(np.float64)

                phase_rad = np.angle(samples)
                phase_deg = np.degrees(phase_rad)

                # Mean values for combined heatmaps.
                mean_i = float(np.mean(I))
                mean_q = float(np.mean(Q))
                mean_amplitude = float(np.mean(amplitude))

                # Circular mean phase.
                mean_phase = float(
                    np.degrees(
                        np.angle(
                            np.mean(np.exp(1j * phase_rad))
                        )
                    )
                )

                mean_i_row.append(mean_i)
                mean_q_row.append(mean_q)
                mean_amplitude_row.append(mean_amplitude)
                mean_phase_row.append(mean_phase)

                # --------------------------------------------------
                # Save individual current datasets
                # --------------------------------------------------

                if self.mode == "full_trace":

                    datasets["i"].append(I)
                    datasets["q"].append(Q)
                    datasets["amplitude"].append(amplitude)
                    datasets["phase"].append(phase_deg)

                else:

                    datasets["i"].append(
                        np.asarray([mean_i])
                    )

                    datasets["q"].append(
                        np.asarray([mean_q])
                    )

                    datasets["amplitude"].append(
                        np.asarray([mean_amplitude])
                    )

                    datasets["phase"].append(
                        np.asarray([mean_phase])
                    )

                # --------------------------------------------------
                # Update progress after successful acquisition
                # and storage of this frequency trace.
                # --------------------------------------------------

                if self.progress_bar:
                    self._progress.iterate(
                        param=(
                            f"{self.x_coordname}={current:g} "
                            f"{self.x_unit}, "
                            f"f={frequency / 1e9:.6f} GHz"
                        )
                    )

                qkit.flow.sleep()

            # --------------------------------------------------
            # Complete current sweep.
            #
            # Append one complete row to each heatmap.
            # --------------------------------------------------

            expected_length = len(self.frequency_vec)

            heatmap_rows = {
                "i": mean_i_row,
                "q": mean_q_row,
                "amplitude": mean_amplitude_row,
                "phase": mean_phase_row,
            }

            for name, values in heatmap_rows.items():

                row = np.asarray(values, dtype=np.float64)

                if row.shape != (expected_length,):
                    raise ValueError(
                        f"Invalid {name} heatmap row shape: "
                        f"{row.shape}; expected {(expected_length,)}"
                    )

                if not np.all(np.isfinite(row)):
                    raise ValueError(
                        f"Non-finite values in {name} heatmap "
                        f"at current {current}"
                    )

                self._heatmap_datasets[name].append(row)

            # Flush after the completed current sweep.
            self._data_file.flush()

            logging.info(
                "Completed current point %d/%d: %g %s",
                ix + 1,
                len(self.x_vec),
                current,
                self.x_unit,
            )