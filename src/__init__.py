from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path("settings.env")
load_dotenv(dotenv_path=dotenv_path)

CAM_INDEX = int(os.getenv("CAM_INDEX"))

# Marble must cross this fraction of the frame’s height to trigger specific logic.
TRIGGER_HEIGHT = float(os.getenv("TRIGGER_HEIGHT"))

# If using a micro:bit for serial communication, specify the port and baud rate:
USE_MICROBIT = bool(os.getenv("USE_MICROBIT"))
MICROBIT_SERIAL_PORT = os.getenv("MICROBIT_SERIAL_PORT")
MICROBIT_BAND_RATE = int(os.getenv("MICROBIT_BAND_RATE"))
