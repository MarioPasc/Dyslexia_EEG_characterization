import time
import psutil
import csv
from datetime import datetime

LOG_FILE = (
    "/home/mariopasc/Python/Datasets/EEG/timeseries/processed/rqe/cpu_temp_log.csv"
)
INTERVAL = 5  # Time in seconds between readings


def get_cpu_temperature():
    """Get CPU temperature using psutil and lm-sensors."""
    try:
        sensors = psutil.sensors_temperatures()
        if "coretemp" in sensors:
            return sensors["coretemp"][0].current  # First core temperature
        else:
            return None
    except AttributeError:
        return None


def log_temperature():
    """Continuously logs CPU temperature."""
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU Temperature (°C)"])  # Write header

        while True:
            temp = get_cpu_temperature()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if temp is not None:
                writer.writerow([timestamp, temp])
                print(f"{timestamp} - CPU Temp: {temp}°C")
            else:
                print(f"{timestamp} - Temperature reading failed.")

            time.sleep(INTERVAL)


if __name__ == "__main__":
    log_temperature()
