import numpy as np
import qkit
from qkit.core.instrument_base import Instrument
import copy
from qkit.drivers.RFSoC4x2_fpga_client import FPGAClient

#qkit driver for FPGA

class RFSoC4x2(Instrument):

    def __init__(self, name, fpga_ip, port=5555):
        super().__init__(name, tags=["physical"])

        self.client = FPGAClient(fpga_ip, port=port)
        self._frequency = None
        self._config = None

        self.add_parameter(
            "frequency",
            type=float,
            flags=Instrument.FLAG_GETSET,
            units="Hz",
        )

    def do_set_frequency(self, value):
        """Set frequency in Hz; FPGA expects MHz."""
        frequency_hz = float(value)
        self.client.set_frequency(frequency_hz / 1e6)
        self._frequency = frequency_hz

    def do_get_frequency(self):
        """Return frequency in Hz."""
        return self._frequency

    def measure(self):
        """Return complex IQ samples from the RFSoC."""
        return self.client.measure()

    def get_time_axis(self):
        return self.client.time_axis()

    def configure(self, config):
        result = self.client.configure(config)
        self._config = dict(config)
        return result

    def get_config(self):
        """Return a copy of the active configuration."""
        if self._config is None:
            return None
        return copy.deepcopy(self._config)

    def get_status(self):
        return self.client.status()

    def close(self):
        self.client.close()