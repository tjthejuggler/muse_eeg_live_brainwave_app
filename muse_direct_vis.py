import sys
import asyncio
import threading
from datetime import datetime
from collections import defaultdict
from bleak import BleakScanner, BleakClient
from PyQt6.QtWidgets import (QApplication, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout,
                          QComboBox, QLabel, QPushButton, QDoubleSpinBox, QFileDialog)
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg
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
MUSE_PRESET_12 = bytearray([0x02, 0x73, 0x0a])  # Preset 12 enables all data streams

class AsyncHelper(QObject):
    """Helper class to run async tasks from Qt"""
    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coroutine(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class MuseDataPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.client = None
        self.last_timestamp = time.time()
        self.packet_count = defaultdict(int)
        self.start_time = None
        
        # Initialize data buffers with separate timestamp arrays
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
        
        self.setup_ui()

    def setup_ui(self):
        # Main layout with scroll area
        main_widget = QWidget()
        self.layout = QVBoxLayout(main_widget)
        
        scroll = QScrollArea()
        scroll.setWidget(main_widget)
        scroll.setWidgetResizable(True)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        
        # Control panel
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_widget)
        self.layout.addWidget(ctrl_widget)

        # Connect button
        self.connect_button = QPushButton("Search for Muse")
        ctrl_layout.addWidget(self.connect_button)
        self.connect_button.clicked.connect(self.toggle_connection)

        # Time range control
        time_range_label = QLabel("Time Window (s):")
        ctrl_layout.addWidget(time_range_label)
        self.time_range_spin = QDoubleSpinBox()
        self.time_range_spin.setRange(1, 60)
        self.time_range_spin.setValue(5)
        self.time_range_spin.setSingleStep(1)
        ctrl_layout.addWidget(self.time_range_spin)

        # Save button
        self.save_button = QPushButton("Save Data")
        ctrl_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_data)

        # Status label
        self.status_label = QLabel("Status: Disconnected")
        ctrl_layout.addWidget(self.status_label)

        # Debug info panel
        debug_group = QWidget()
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.addWidget(QLabel("<b>Current Values</b>"))
        
        # Create labels for each sensor type
        self.debug_labels = {
            'eeg': {ch: QLabel(f"{ch}: --") for ch in MUSE_CHANNELS},
            'acc': {ch: QLabel(f"Acc {ch}: --") for ch in ACC_CHANNELS},
            'gyro': {ch: QLabel(f"Gyro {ch}: --") for ch in GYRO_CHANNELS},
            'ppg': {ch: QLabel(f"PPG {ch}: --") for ch in PPG_CHANNELS}
        }
        
        # Add labels to debug panel
        for sensor_type in ['eeg', 'acc', 'gyro', 'ppg']:
            for label in self.debug_labels[sensor_type].values():
                debug_layout.addWidget(label)
        
        self.layout.addWidget(debug_group)

        # Initialize plot containers
        self.plots = {}
        self.curves = {}
        
        # Define plot configurations
        plot_configs = {
            'eeg': {
                'title': 'EEG Data',
                'channels': MUSE_CHANNELS,
                'y_label': ('Amplitude', 'µV'),
                'pen_color': 'y',
                'channel_prefix': 'EEG Channel'
            },
            'acc': {
                'title': 'Accelerometer Data',
                'channels': ACC_CHANNELS,
                'y_label': ('Acceleration', 'g'),
                'pen_color': 'g',
                'channel_prefix': 'Accelerometer'
            },
            'gyro': {
                'title': 'Gyroscope Data',
                'channels': GYRO_CHANNELS,
                'y_label': ('Angular Velocity', 'deg/s'),
                'pen_color': 'r',
                'channel_prefix': 'Gyroscope'
            },
            'ppg': {
                'title': 'PPG Data',
                'channels': PPG_CHANNELS,
                'y_label': ('Amplitude', 'au'),
                'pen_color': 'b',
                'channel_prefix': 'PPG Channel'
            }
        }
        
        # Create plot groups with consistent configuration
        for plot_type, config in plot_configs.items():
            group = QWidget()
            layout = QVBoxLayout(group)
            layout.addWidget(QLabel(f"<b>{config['title']}</b>"))
            
            self.plots[plot_type] = []
            self.curves[plot_type] = []
            
            for ch in config['channels']:
                plot = pg.PlotWidget(title=f"{config['channel_prefix']} {ch}")
                
                # Configure plot
                plot.setLabel('left', *config['y_label'])
                plot.setLabel('bottom', 'Time', 's')
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setMouseEnabled(x=True, y=False)  # Only allow x-axis zoom
                plot.enableAutoRange(axis='y', enable=False)  # Disable y-axis auto-range
                plot.setClipToView(True)  # Improve performance
                plot.setDownsampling(auto=True, mode='peak')  # Enable automatic downsampling
                
                # Create curve with anti-aliasing for smoother display
                curve = plot.plot(pen={'color': config['pen_color'], 'width': 1})
                curve.setDownsampling(auto=True, method='peak')
                
                self.plots[plot_type].append(plot)
                self.curves[plot_type].append(curve)
                layout.addWidget(plot)
            
            self.layout.addWidget(group)

        # Update timer - increased update rate from 100ms to 50ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # Update every 50ms for smoother display

        self.setWindowTitle('Muse Data Visualizer')
        self.resize(1000, 800)

    async def scan_for_muse(self):
        self.status_label.setText("Status: Scanning for Muse...")
        print("Starting Muse scan...")
        
        def detection_callback(device, advertising_data):
            # Log all discovered devices for debugging
            print(f"Discovered device: {device.name} ({device.address})")
            if advertising_data.local_name:
                print(f"  Local name: {advertising_data.local_name}")
            if advertising_data.manufacturer_data:
                print(f"  Manufacturer data: {advertising_data.manufacturer_data}")
            if advertising_data.service_data:
                print(f"  Service data: {advertising_data.service_data}")
            if advertising_data.service_uuids:
                print(f"  Service UUIDs: {advertising_data.service_uuids}")

        scanner = BleakScanner(detection_callback=detection_callback)
        
        # Start scanning
        await scanner.start()
        await asyncio.sleep(5.0)  # Scan for 5 seconds
        await scanner.stop()
        
        # Get discovered devices
        devices = await scanner.get_discovered_devices()
        
        # Look for Muse devices
        muse_devices = []
        for d in devices:
            device_info = f"Name: {d.name}, Address: {d.address}"
            if d.name:
                if "Muse" in d.name:
                    print(f"Found Muse device: {device_info}")
                    muse_devices.append(d)
                else:
                    print(f"Other device: {device_info}")
        
        if muse_devices:
            # If multiple Muse devices found, use the first one
            selected_device = muse_devices[0]
            self.status_label.setText(f"Status: Found {selected_device.name}")
            print(f"Selected Muse device: {selected_device.name} ({selected_device.address})")
            return selected_device
        
        self.status_label.setText("Status: No Muse device found")
        return None

    def handle_eeg(self, _, data):
        """Handle incoming EEG data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        self.packet_count['eeg'] += 1
        rate = self.packet_count['eeg'] / (now - self.last_timestamp) if now > self.last_timestamp else 0
        
        try:
            print("\n=== EEG Packet Processing ===")
            print(f"Packet size: {len(data)} bytes")
            print(f"Raw data: {[hex(x) for x in data]}")
            
            if len(data) >= 20:  # Each channel uses 5 bytes
                print("\nProcessing channels:")
                timestamp_added = False
                for i, ch in enumerate(MUSE_CHANNELS):
                    start_idx = i * 5
                    value = int.from_bytes(data[start_idx:start_idx+2], byteorder='little', signed=True)
                    value = value * 0.1  # Scale to microvolts
                    
                    print(f"\n{ch} Channel:")
                    print(f"  Bytes: {[hex(x) for x in data[start_idx:start_idx+2]]}")
                    print(f"  Raw value: {value}")
                    
                    if len(self.eeg_data[ch]) > 500:
                        old_value = self.eeg_data[ch][0]
                        self.eeg_data[ch].pop(0)
                        print(f"  Removed old value: {old_value}")
                        if not timestamp_added:
                            self.eeg_times.pop(0)
                    
                    self.eeg_data[ch].append(value)
                    print(f"  Added new value: {value}")
                    print(f"  Current buffer size: {len(self.eeg_data[ch])}")
                    
                    if not timestamp_added:
                        self.eeg_times.append(now - self.start_time)
                        timestamp_added = True
                    
                    # Update debug label
                    self.debug_labels['eeg'][ch].setText(f"{ch}: {value:.2f} µV")
                
                print("\nTimestamp info:")
                print(f"  Buffer size: {len(self.eeg_times)}")
                if self.eeg_times:
                    print(f"  Time range: {self.eeg_times[0]:.2f} to {self.eeg_times[-1]:.2f}")
        except Exception as e:
            print(f"Error parsing EEG data: {e}")
            import traceback
            traceback.print_exc()

    def handle_accelerometer(self, _, data):
        """Handle incoming accelerometer data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        self.packet_count['acc'] += 1
        rate = self.packet_count['acc'] / (now - self.last_timestamp) if now > self.last_timestamp else 0
        
        try:
            print(f"ACC packet ({rate:.1f} Hz): {len(data)} bytes - {[hex(x) for x in data]}")
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(ACC_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    value = value / 16384.0  # Convert to g (±2g range)
                    
                    # Update debug label
                    self.debug_labels['acc'][ch].setText(f"Acc {ch}: {value:.3f} g")
                    
                    if len(self.acc_data[ch]) > 500:
                        self.acc_data[ch].pop(0)
                        if not timestamp_added:  # Only remove timestamp once per packet
                            self.acc_times.pop(0)
                    
                    self.acc_data[ch].append(value)
                    if not timestamp_added:  # Only add timestamp once per packet
                        self.acc_times.append(now - self.start_time)
                        timestamp_added = True
        except Exception as e:
            print(f"Error parsing accelerometer data: {e}")

    def handle_gyroscope(self, _, data):
        """Handle incoming gyroscope data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        self.packet_count['gyro'] += 1
        rate = self.packet_count['gyro'] / (now - self.last_timestamp) if now > self.last_timestamp else 0
        
        try:
            print(f"GYRO packet ({rate:.1f} Hz): {len(data)} bytes - {[hex(x) for x in data]}")
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(GYRO_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    value = value * 0.0074768  # Convert to deg/s (±500 deg/s range)
                    
                    # Update debug label
                    self.debug_labels['gyro'][ch].setText(f"Gyro {ch}: {value:.1f} deg/s")
                    
                    if len(self.gyro_data[ch]) > 500:
                        self.gyro_data[ch].pop(0)
                        if not timestamp_added:  # Only remove timestamp once per packet
                            self.gyro_times.pop(0)
                    
                    self.gyro_data[ch].append(value)
                    if not timestamp_added:  # Only add timestamp once per packet
                        self.gyro_times.append(now - self.start_time)
                        timestamp_added = True
        except Exception as e:
            print(f"Error parsing gyroscope data: {e}")

    def handle_ppg(self, _, data):
        """Handle incoming PPG data"""
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        self.packet_count['ppg'] += 1
        rate = self.packet_count['ppg'] / (now - self.last_timestamp) if now > self.last_timestamp else 0
        
        try:
            print(f"PPG packet ({rate:.1f} Hz): {len(data)} bytes - {[hex(x) for x in data]}")
            if len(data) >= 6:
                timestamp_added = False
                for i, ch in enumerate(PPG_CHANNELS):
                    value = int.from_bytes(data[i*2:(i+1)*2], byteorder='little', signed=True)
                    
                    # Update debug label
                    self.debug_labels['ppg'][ch].setText(f"PPG {ch}: {value}")
                    
                    if len(self.ppg_data[ch]) > 500:
                        self.ppg_data[ch].pop(0)
                        if not timestamp_added:  # Only remove timestamp once per packet
                            self.ppg_times.pop(0)
                    
                    self.ppg_data[ch].append(value)
                    if not timestamp_added:  # Only add timestamp once per packet
                        self.ppg_times.append(now - self.start_time)
                        timestamp_added = True
        except Exception as e:
            print(f"Error parsing PPG data: {e}")

    def toggle_connection(self):
        if not self.client or not self.client.is_connected:
            self.connect_button.setEnabled(False)
            self.connect_button.setText("Connecting...")
            self.async_helper.run_coroutine(self.connect_muse())
        else:
            self.async_helper.run_coroutine(self.disconnect_muse())
            self.connect_button.setText("Search for Muse")
            self.status_label.setText("Status: Disconnected")

    async def connect_muse(self):
        try:
            device = await self.scan_for_muse()
            if not device:
                self.connect_button.setEnabled(True)
                self.connect_button.setText("Search for Muse")
                return

            print(f"Connecting to {device.name}...")
            self.client = BleakClient(device)
            await self.client.connect()
            print("Connected! Getting services...")
            
            # List all services and characteristics for debugging
            for service in self.client.services:
                print(f"Service: {service.uuid}")
                for char in service.characteristics:
                    print(f"  Characteristic: {char.uuid}")
                    print(f"  Properties: {char.properties}")
            
            # Initialize the device with multiple attempts
            print("Initializing device...")
            for attempt in range(3):
                try:
                    # Stop any existing streams
                    await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_STOP_COMMAND)
                    await asyncio.sleep(0.5)
                    
                    # Set preset for all data streams
                    await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_PRESET_12)
                    await asyncio.sleep(1)
                    
                    # Subscribe to all data streams
                    print("Subscribing to data streams...")
                    await self.client.start_notify(MUSE_EEG_UUID, self.handle_eeg)
                    await self.client.start_notify(MUSE_ACCELEROMETER_UUID, self.handle_accelerometer)
                    await self.client.start_notify(MUSE_GYROSCOPE_UUID, self.handle_gyroscope)
                    await self.client.start_notify(MUSE_PPG_UUID, self.handle_ppg)
                    await asyncio.sleep(0.5)
                    
                    # Start the data stream
                    print("Starting data streams...")
                    await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_START_COMMAND)
                    await asyncio.sleep(0.5)
                    
                    break
                except Exception as e:
                    print(f"Initialization attempt {attempt + 1} failed: {e}")
                    if attempt == 2:  # Last attempt
                        raise
                    await asyncio.sleep(1)
            
            self.last_timestamp = time.time()
            self.packet_count.clear()
            self.start_time = None  # Reset start time for new connection
            
            self.connect_button.setText("Disconnect")
            self.status_label.setText(f"Status: Connected to {device.name}")
            print("Setup complete!")
            
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.status_label.setText(f"Status: Connection failed - {str(e)}")
            self.connect_button.setText("Search for Muse")
        finally:
            self.connect_button.setEnabled(True)

    async def disconnect_muse(self):
        if self.client and self.client.is_connected:
            try:
                # Stop the data stream
                await self.client.write_gatt_char(MUSE_CONTROL_UUID, MUSE_STOP_COMMAND)
                # Disconnect
                await self.client.disconnect()
            except Exception as e:
                print(f"Error during disconnect: {str(e)}")
            finally:
                self.client = None
                self.start_time = None

    def update_plots(self):
        """Update all plots with new data using a simplified, more robust approach"""
        if not self.start_time:
            return

        # Fixed Y-axis ranges for each sensor type
        y_ranges = {
            'eeg': (-150, 150),    # µV range for EEG
            'acc': (-2, 2),        # g range for accelerometer
            'gyro': (-500, 500),   # deg/s range for gyroscope
            'ppg': (-2000, 2000)   # arbitrary units for PPG
        }
        
        time_window = self.time_range_spin.value()
        current_time = time.time() - self.start_time

        def update_single_plot(plot_type, channel_idx, channel_name, times, data):
            if not times or not data[channel_name]:
                return
                
            # Get visible time window
            visible_start = max(0, current_time - time_window)
            visible_end = current_time
            
            # Find indices for visible data
            visible_indices = [i for i, t in enumerate(times) if visible_start <= t <= visible_end]
            
            if not visible_indices:
                return
                
            # Extract visible data
            visible_times = [times[i] for i in visible_indices]
            visible_data = [data[channel_name][i] for i in visible_indices]
            
            # Update curve
            self.curves[plot_type][channel_idx].setData(visible_times, visible_data)
            
            # Update X axis
            self.plots[plot_type][channel_idx].setXRange(visible_start, visible_end)
            
            # Set fixed Y range
            self.plots[plot_type][channel_idx].setYRange(*y_ranges[plot_type])

        # Update all plots
        for i, ch in enumerate(MUSE_CHANNELS):
            update_single_plot('eeg', i, ch, self.eeg_times, self.eeg_data)
            
        for i, ch in enumerate(ACC_CHANNELS):
            update_single_plot('acc', i, ch, self.acc_times, self.acc_data)
            
        for i, ch in enumerate(GYRO_CHANNELS):
            update_single_plot('gyro', i, ch, self.gyro_times, self.gyro_data)
            
        for i, ch in enumerate(PPG_CHANNELS):
            update_single_plot('ppg', i, ch, self.ppg_times, self.ppg_data)

    def save_data(self):
        """Save all sensor data to CSV files"""
        base_name, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV files (*.csv)")
        if base_name:
            # Save EEG data
            with open(f"{base_name}_eeg.csv", 'w') as f:
                f.write("Time," + ",".join(MUSE_CHANNELS) + "\n")
                for i in range(len(self.eeg_times)):
                    row = [str(self.eeg_times[i])]
                    for ch in MUSE_CHANNELS:
                        row.append(str(self.eeg_data[ch][i]))
                    f.write(",".join(row) + "\n")
            
            # Save accelerometer data
            with open(f"{base_name}_acc.csv", 'w') as f:
                f.write("Time," + ",".join(ACC_CHANNELS) + "\n")
                for i in range(len(self.acc_times)):
                    row = [str(self.acc_times[i])]
                    for ch in ACC_CHANNELS:
                        row.append(str(self.acc_data[ch][i]))
                    f.write(",".join(row) + "\n")
            
            # Save gyroscope data
            with open(f"{base_name}_gyro.csv", 'w') as f:
                f.write("Time," + ",".join(GYRO_CHANNELS) + "\n")
                for i in range(len(self.gyro_times)):
                    row = [str(self.gyro_times[i])]
                    for ch in GYRO_CHANNELS:
                        row.append(str(self.gyro_data[ch][i]))
                    f.write(",".join(row) + "\n")
            
            # Save PPG data
            with open(f"{base_name}_ppg.csv", 'w') as f:
                f.write("Time," + ",".join(PPG_CHANNELS) + "\n")
                for i in range(len(self.ppg_times)):
                    row = [str(self.ppg_times[i])]
                    for ch in PPG_CHANNELS:
                        row.append(str(self.ppg_data[ch][i]))
                    f.write(",".join(row) + "\n")

def main():
    app = QApplication(sys.argv)
    window = MuseDataPlotter()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
