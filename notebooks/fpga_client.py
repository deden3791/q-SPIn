import zmq
import numpy as np


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
        """
        Request an IQ measurement.

        Returns
        -------
        iq : numpy.ndarray
            Complex IQ samples.
        """

        self.socket.send_json({ "command": "measure", "parameters": {} })

        response = self.socket.recv()

        # The FPGA computer should return a binary
        # NumPy array containing complex IQ samples.

        iq = np.frombuffer( response, dtype=np.complex64 ).copy()

        return iq


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