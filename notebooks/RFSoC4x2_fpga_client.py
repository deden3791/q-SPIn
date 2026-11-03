import zmq
import numpy as np
import json 

#qkit driver to interface PC to FPGA server

class FPGAConfigurationError(RuntimeError):
    """The FPGA is not configured."""


class FPGAConnectionError(ConnectionError):
    """Communication with the FPGA server failed."""


class FPGAAcquisitionError(RuntimeError):
    """The FPGA server could not complete an acquisition."""

class FPGAClient:

    def __init__(self, fpga_ip, port=5555, timeout=10000):
        """
        Connect to the FPGA control computer.

        Parameters
        ----------
        fpga_ip : str
            IP address of the computer running the FPGA controller.
        port : int
            ZeroMQ port.
        timeout : int
            Communication timeout in milliseconds.
        """

        self.fpga_ip = fpga_ip
        self.port = port

        self.context = zmq.Context()

        self.socket = self.context.socket(zmq.REQ)

        self.socket.setsockopt( zmq.RCVTIMEO, timeout )

        self.socket.setsockopt( zmq.SNDTIMEO, timeout )

        self.socket.connect( f"tcp://{fpga_ip}:{port}" )

    def send_command(self, command, **parameters):
        """
        Send a command to the FPGA computer.
        """

        message = {
            "command": command,
            "parameters": parameters
        }

        self.socket.send_json(message)

        response = self.socket.recv_json()

        if response.get("status") != "ok": raise RuntimeError( response.get("error", "Unknown FPGA error") )

        return response


    def configure(self, config):
        """
        Send measurement configuration to FPGA computer.
        """

        return self.send_command( "configure", config=config )

    def set_frequency(self, frequency):
        """
        Set measurement/generator frequency.
        """

        return self.send_command( "set_frequency", frequency=frequency )

    
    def measure(self):
        """Request complex IQ samples from the FPGA server."""

        try:
            self.socket.send_json({
                "command": "measure",
                "parameters": {},
            })
            response = self.socket.recv()

        except zmq.Again as exc:
            raise TimeoutError(
                "Timed out waiting for the FPGA server. "
                "Check server availability and acquisition duration."
            ) from exc

        except zmq.ZMQError as exc:
            raise FPGAConnectionError(
                f"ZeroMQ communication failed: {exc}"
            ) from exc

        # Decode JSON server errors; successful IQ responses are binary.
        try:
            payload = json.loads(response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            payload = None

        if isinstance(payload, dict) and payload.get("status") == "error":
            error = str(payload.get("error", "Unknown FPGA server error"))
            error_lower = error.lower()

            if "not configured" in error_lower:
                raise FPGAConfigurationError(error)

            raise FPGAAcquisitionError(error)

        if not isinstance(response, bytes):
            raise ValueError("Expected a binary IQ response.")

        sample_size = np.dtype(np.complex64).itemsize

        if not response:
            raise ValueError("FPGA server returned an empty IQ response.")

        if len(response) % sample_size != 0:
            raise ValueError(
                f"Malformed IQ response: {len(response)} bytes; "
                f"expected a multiple of {sample_size}."
            )

        return np.frombuffer(response, dtype=np.complex64).copy()





    def time_axis(self):
        self.socket.send_json({
            "command": "time_axis"
        })

        return np.frombuffer(
            self.socket.recv(),
            dtype=np.float32
        )

    def status(self):
        """
        Ask FPGA computer for its current status.
        """

        return self.send_command( "status" )


    def close(self):
        """
        Close ZeroMQ connection.
        """

        self.socket.close()
        self.context.term()