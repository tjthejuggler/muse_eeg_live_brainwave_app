import asyncio
import threading
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bleak import BleakScanner, BleakClient
import numpy as np
import time

# Muse BLE characteristics
MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
ACC_CHANNELS = ["X", "Y", "Z"]
GYRO_CHANNELS = ["X", "Y", "Z"]
PPG_CHANNELS = ["PPG1", "PPG2", "PPG3"]

# Characteristic UUIDs
MUSE_CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
MUSE_EEG_UUID = "273e0003-4c4d-454d-96be-f03bac821358"
MUSE_ACCELEROMETER_UUID = "273e000a-4c4d-454d-96be-f03bac821358"
MUSE_GYROSCOPE_UUID = "273e0009-4c4d-454d-96be-f03bac821358"
MUSE_PPG_UUID = "273e000f-4c4d-454d-96be-f03bac821358"

# Commands
MUSE_START_COMMAND = bytearray([0x02, 0x64, 0x0a])
MUSE_STOP_COMMAND = bytearray([0x02, 0x68, 0x0a])
MUSE_PRESET_12 = bytearray([0x02, 0x73, 0x0a])

class AsyncHelper:
    """Helper class to run async tasks"""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coroutine(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class MuseDataVisualizer:
    def __init__(self):
        self.client = None
        self.start_time = None
        self.window_size = 20  # Fixed 20-second window
        
        # Initialize data buffers
        self.eeg_data = {ch: [] for ch in MUSE_CHANNELS}
        self.eeg_times = []
        self.acc_data = {ch: [] for ch in ACC_CHANNELS}
        self.acc_times = []
        self.gyro_data = {ch: [] for ch in GYRO_CHANNELS}
        self.gyro_times = []
        self.ppg_data = {ch: [] for ch in PPG_CHANNELS}
        self.ppg_times = []
        
        # Initialize async helper
        self.async_helper = AsyncHelper()
        
        # Setup matplotlib figure
        plt.style.use('dark_background')
        total_plots = len(MUSE_CHANNELS) + len(ACC_CHANNELS) + len(GYRO_CHANNELS) + len(PPG_CHANNELS)
        self.fig, self.axes = plt.subplots(total_plots, 1, figsize=(12, 16), sharex=True)
        self.fig.suptitle('Muse Sensor Data - Last 20 Seconds', color='white')
        
        # Initialize lines for each sensor type
        self.lines = {'eeg': {}, 'acc': {}, 'gyro': {}, 'ppg': {}}
        current_ax = 0
        
        # EEG plots
        for channel in MUSE_CHANNELS:
            line, = self.axes[current_ax].plot([], [], label=f'EEG {channel}')
            self.lines['eeg'][channel] = line
            self.axes[current_ax].set_ylabel('µV')
            self.axes[current_ax].legend(loc='upper right')
            self.axes[current_ax].grid(True, alpha=0.3)
            self.axes[current_ax].set_ylim(-150, 150)
            current_ax += 1
        
        # Accelerometer plots
        for channel in ACC_CHANNELS:
            line, = self.axes[current_ax].plot([], [], label=f'ACC {channel}')
            self.lines['acc'][channel] = line
            self.axes[current_ax].set_ylabel('g')
            self.axes[current_ax].legend(loc='upper right')
            self.axes[current_ax].grid(True, alpha=0.3)
            self.axes[current_ax].set_ylim(-2, 2)
            current_ax += 1
            
        # Gyroscope plots
        for channel in GYRO_CHANNELS:
            line, = self.axes[current_ax].plot([], [], label=f'GYRO {channel}')
            self.lines['gyro'][channel] = line
            self.axes[current_ax].set_ylabel('deg/s')
            self.axes[current_ax].legend(loc='upper right')
            self.axes[current_ax].grid(True, alpha=0.3)
            self.axes[current_ax].set_ylim(-500, 500)
            current_ax += 1
            
        # PPG plots
        for channel in PPG_CHANNELS:
            line, = self.axes[current_ax].plot([], [], label=channel)
            self.lines['ppg'][channel] = line
            self.axes[current_ax].set_ylabel('au')
            self.axes[current_ax].legend(loc='upper right')
            self.axes[current_ax].grid(True, alpha=0.3)
            self.axes[current_ax].set_ylim(-2000, 2000)
            current_ax += 1
        
        self.axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()

    def handle_eeg(self, _, data):
        """Handle incoming EEG data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        try:
            if len(data) >= 20:  # Each channel uses 5 bytes
                timestamp_added = False
                for i, ch in enumerate(MUSE_CHANNELS):
                    start_idx = i * 5
                    value = int.from_bytes(data[start_idx:start_idx+2], byteorder='little', signed=True)
                    value = value * 0.1  # Scale to microvolts
                    
                    self.eeg_data[ch].append(value)
                    
                    # Keep only last 20 seconds of data
                    current_time = now - self.start_time
                    while self.eeg_times and (current_time - self.eeg_times[0]) > self.window_size:
                        self.eeg_times.pop(0)
                        for channel in MUSE_CHANNELS:
                            self.eeg_data[channel].pop(0)
                    
                    if not timestamp_added:
                        self.eeg_times.append(current_time)
                        timestamp_added = True
                        
        except Exception as e:
            print(f"Error parsing EEG data: {e}")

    async def scan_for_muse(self):
        print("Scanning for Muse devices...")
        scanner = BleakScanner()
        devices = await scanner.discover()
        
        for d in devices:
            if d.name and "Muse" in d.name:
                print(f"Found Muse device: {d.name}")
                return d
        
        return None

    async def connect_muse(self):
        try:
            device = await self.scan_for_muse()
            if not device:
                print("No Muse device found")
                return False

            print(f"Connecting to {device.name}...")
            self.client = BleakClient(device)
            await self.client.connect()
            
            # Initialize device
            await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_STOP_COMMAND)
            await asyncio.sleep(0.5)
            
            await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_PRESET_12)
            await asyncio.sleep(1)
            
            # Subscribe to all data streams
            await self.client.start_notify(MUSE_EEG_UUID, self.handle_eeg)
            await self.client.start_notify(MUSE_ACCELEROMETER_UUID, self.handle_accelerometer)
            await self.client.start_notify(MUSE_GYROSCOPE_UUID, self.handle_gyroscope)
            await self.client.start_notify(MUSE_PPG_UUID, self.handle_ppg)
            await asyncio.sleep(0.5)
            
            # Start the data stream
            await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_START_COMMAND)
            
            self.start_time = time.time()
            print("Connected and streaming!")
            return True
            
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False

    def handle_accelerometer(self, _, data):
        """Handle incoming accelerometer data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        try:
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(ACC_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    value = value / 16384.0  # Convert to g (±2g range)
                    
                    self.acc_data[ch].append(value)
                    
                    # Keep only last 20 seconds of data
                    current_time = now - self.start_time
                    while self.acc_times and (current_time - self.acc_times[0]) > self.window_size:
                        self.acc_times.pop(0)
                        for channel in ACC_CHANNELS:
                            self.acc_data[channel].pop(0)
                    
                    if not timestamp_added:
                        self.acc_times.append(current_time)
                        timestamp_added = True
                        
        except Exception as e:
            print(f"Error parsing accelerometer data: {e}")

    def handle_gyroscope(self, _, data):
        """Handle incoming gyroscope data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        try:
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(GYRO_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    value = value * 0.0074768  # Convert to deg/s (±500 deg/s range)
                    
                    self.gyro_data[ch].append(value)
                    
                    # Keep only last 20 seconds of data
                    current_time = now - self.start_time
                    while self.gyro_times and (current_time - self.gyro_times[0]) > self.window_size:
                        self.gyro_times.pop(0)
                        for channel in GYRO_CHANNELS:
                            self.gyro_data[channel].pop(0)
                    
                    if not timestamp_added:
                        self.gyro_times.append(current_time)
                        timestamp_added = True
                        
        except Exception as e:
            print(f"Error parsing gyroscope data: {e}")

    def handle_ppg(self, _, data):
        """Handle incoming PPG data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        try:
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(PPG_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    
                    self.ppg_data[ch].append(value)
                    
                    # Keep only last 20 seconds of data
                    current_time = now - self.start_time
                    while self.ppg_times and (current_time - self.ppg_times[0]) > self.window_size:
                        self.ppg_times.pop(0)
                        for channel in PPG_CHANNELS:
                            self.ppg_data[channel].pop(0)
                    
                    if not timestamp_added:
                        self.ppg_times.append(current_time)
                        timestamp_added = True
                        
        except Exception as e:
            print(f"Error parsing PPG data: {e}")

    def update_plot(self, frame):
        if not any([self.eeg_times, self.acc_times, self.gyro_times, self.ppg_times]):
            return [line for sensor_lines in self.lines.values() for line in sensor_lines.values()]
        
        current_time = time.time() - self.start_time
        start_time = max(0, current_time - self.window_size)
        
        # Update x-axis limits to show last 20 seconds
        for ax in self.axes:
            ax.set_xlim(start_time, current_time)
        
        # Update each sensor's data
        for channel, line in self.lines['eeg'].items():
            if self.eeg_times:
                line.set_data(self.eeg_times, self.eeg_data[channel])
                
        for channel, line in self.lines['acc'].items():
            if self.acc_times:
                line.set_data(self.acc_times, self.acc_data[channel])
                
        for channel, line in self.lines['gyro'].items():
            if self.gyro_times:
                line.set_data(self.gyro_times, self.gyro_data[channel])
                
        for channel, line in self.lines['ppg'].items():
            if self.ppg_times:
                line.set_data(self.ppg_times, self.ppg_data[channel])
        
        return [line for sensor_lines in self.lines.values() for line in sensor_lines.values()]

    def run(self):
        # Connect to Muse
        if not self.async_helper.run_coroutine(self.connect_muse()).result():
            print("Failed to connect to Muse")
            return
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=50, 
                          blit=True, cache_frame_data=False)
        
        # Show plot
        plt.show()

if __name__ == "__main__":
    visualizer = MuseDataVisualizer()
    visualizer.run()
