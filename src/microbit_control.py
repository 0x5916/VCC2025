"""
microbit_control.py

Handles serial communication to, e.g., a micro:bit or other device.
"""

import serial

class SerialConnector:
    def __init__(self, serial_port: str, baud_rate: int = 115200, enable: bool = True):
        """
        :param serial_port: The port on which micro:bit/serial device is connected.
        :param baud_rate: Baud rate for serial communication.
        :param enable: If False, disables serial comm entirely (useful for testing).
        """
        self._enabled = enable
        if not self._enabled:
            return

        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port}")

        # Mapping of (color_code, side) -> command. 
        # Adjust to suit your hardware logic:
        self.command_map = {
            ('r', 'l'): 'C',
            ('r', 'r'): 'A',
            ('g', 'l'): 'C',  # Example: 'g' maps to same commands as 'r'
            ('g', 'r'): 'A',
            ('b', 'l'): 'B',
            ('b', 'r'): 'C',
            ('y', 'l'): 'A',
            ('y', 'r'): 'D'
        }

    def send_serial(self, color_code: str, side: str):
        """
        Sends a command based on (color_code, side), e.g. ('r','l') -> 'C'.
        """
        if not self._enabled:
            return

        cmd = self.command_map.get((color_code, side))
        if cmd:
            self.ser.write(bytearray(cmd + "$", "utf-8"))

    def close(self):
        """Closes the serial port if enabled."""
        if self._enabled:
            self.ser.close()