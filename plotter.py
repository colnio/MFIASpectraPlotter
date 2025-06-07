import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QComboBox, 
                            QListWidget, QLabel, QCheckBox, QGroupBox, QListWidgetItem,
                            QSlider, QSpinBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from file_reader import read_data, get_sample_names, get_field_names, extract_sample_field


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class SpectralPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Data Plotter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data storage
        self.data_df = None
        self.header_df = None
        self.sample_names = []
        self.field_names = []
        self.active_samples = set()
        self.color_palette = []
        self.line_thickness = 2  # Default line thickness
        self.legend_font_size = 10  # Default legend font size
        self.legend = None  # Store legend object
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # File selection buttons
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.header_btn = QPushButton("Select Header File")
        self.data_btn = QPushButton("Select Data File")
        self.header_btn.clicked.connect(self.select_header_file)
        self.data_btn.clicked.connect(self.select_data_file)
        file_layout.addWidget(self.header_btn)
        file_layout.addWidget(self.data_btn)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Axis selection
        axis_group = QGroupBox("Axis Selection")
        axis_layout = QVBoxLayout()
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        axis_layout.addWidget(QLabel("X Axis:"))
        axis_layout.addWidget(self.x_axis_combo)
        axis_layout.addWidget(QLabel("Y Axis:"))
        axis_layout.addWidget(self.y_axis_combo)
        self.x_axis_combo.currentTextChanged.connect(self.update_plot)
        self.y_axis_combo.currentTextChanged.connect(self.update_plot)
        axis_group.setLayout(axis_layout)
        control_layout.addWidget(axis_group)
        
        # Sample selection
        sample_group = QGroupBox("Sample Selection")
        sample_layout = QVBoxLayout()
        
        # Add select/deselect all buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_samples)
        self.deselect_all_btn.clicked.connect(self.deselect_all_samples)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        sample_layout.addLayout(button_layout)
        
        self.sample_list = QListWidget()
        self.sample_list.itemChanged.connect(self.update_plot)
        sample_layout.addWidget(self.sample_list)
        sample_group.setLayout(sample_layout)
        control_layout.addWidget(sample_group)
        
        # Save button
        self.save_btn = QPushButton("Save Visible Data")
        self.save_btn.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_btn)
        
        # Add plot appearance controls
        appearance_group = QGroupBox("Plot Appearance")
        appearance_layout = QVBoxLayout()
        
        # Line thickness control
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Line Thickness:"))
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(10)
        self.thickness_slider.setValue(self.line_thickness)
        self.thickness_slider.valueChanged.connect(self.update_line_thickness)
        thickness_layout.addWidget(self.thickness_slider)
        appearance_layout.addLayout(thickness_layout)
        
        # Legend font size control
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Legend Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setMinimum(8)
        self.font_size_spin.setMaximum(20)
        self.font_size_spin.setValue(self.legend_font_size)
        self.font_size_spin.valueChanged.connect(self.update_legend_font_size)
        font_layout.addWidget(self.font_size_spin)
        appearance_layout.addLayout(font_layout)
        
        appearance_group.setLayout(appearance_layout)
        control_layout.addWidget(appearance_group)
        
        # Add control panel to main layout
        layout.addWidget(control_panel, stretch=1)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLogMode(x=True, y=False)
        layout.addWidget(self.plot_widget, stretch=4)
        
        # Initialize plot
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        
    def select_header_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Header File", "", "CSV Files (*.csv)")
        if file_name:
            self.header_file = file_name
            
    def select_data_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "CSV Files (*.csv)")
        if file_name and hasattr(self, 'header_file'):
            self.load_data(file_name, self.header_file)
            
    def generate_color_palette(self, n_colors):
        """Generate a color palette with n_colors distinct colors using a perceptually uniform colormap."""
        # Use a perceptually uniform colormap
        if n_colors <= 20:
            cmap = plt.colormaps['tab20']
        else:
            # If we need more colors, use a different colormap
            cmap = plt.colormaps['viridis']
        
        # Generate evenly spaced colors
        colors = [to_hex(cmap(i/n_colors)) for i in range(n_colors)]
        return colors

    def load_data(self, data_file, header_file):
        try:
            self.data_df, self.header_df = read_data(data_file, header_file)
            self.sample_names = get_sample_names(self.header_df)
            self.field_names = get_field_names(self.data_df)
            
            # Generate color palette based on number of samples
            self.color_palette = self.generate_color_palette(len(self.sample_names))
            
            # Update UI elements
            self.x_axis_combo.clear()
            self.y_axis_combo.clear()
            self.x_axis_combo.addItems(self.field_names)
            self.y_axis_combo.addItems(self.field_names)
            
            # Set default axes
            if 'frequency' in self.field_names:
                self.x_axis_combo.setCurrentText('frequency')
            if 'param1' in self.field_names:
                self.y_axis_combo.setCurrentText('param1')
            
            # Update sample list
            self.sample_list.clear()
            for sample in self.sample_names:
                item = QListWidgetItem(sample)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.sample_list.addItem(item)
                self.active_samples.add(sample)
            
            self.update_plot()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            
    def update_plot(self):
        if self.data_df is None or self.header_df is None:
            return
            
        self.plot_widget.clear()
        x_field = self.x_axis_combo.currentText()
        y_field = self.y_axis_combo.currentText()
        
        # Add legend with custom font size
        if self.legend is not None:
            self.plot_widget.removeItem(self.legend)
        self.legend = self.plot_widget.addLegend(offset=(10, 10))
        self.legend.setLabelTextSize(f'{self.legend_font_size}pt')
       
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            if item.checkState() == Qt.Checked:
                sample_name = item.text()
                x_data = extract_sample_field(self.data_df, self.header_df, sample_name, x_field)
                y_data = extract_sample_field(self.data_df, self.header_df, sample_name, y_field)
                
                # Use color from the palette
                color = self.color_palette[i]
                pen = pg.mkPen(color=color, width=self.line_thickness)
                
                for x, y in zip(x_data, y_data):
                    self.plot_widget.plot(x, y, name=sample_name, pen=pen)
        
        self.plot_widget.setLabel('left', y_field)
        self.plot_widget.setLabel('bottom', x_field)
        
    def save_data(self):
        if self.data_df is None or self.header_df is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if file_name:
            x_field = self.x_axis_combo.currentText()
            y_field = self.y_axis_combo.currentText()
            
            # Create a new DataFrame for visible data
            data_dict = {}
            for i in range(self.sample_list.count()):
                item = self.sample_list.item(i)
                if item.checkState() == Qt.Checked:
                    sample_name = item.text()
                    x_data = extract_sample_field(self.data_df, self.header_df, sample_name, x_field)
                    y_data = extract_sample_field(self.data_df, self.header_df, sample_name, y_field)
                    
                    # Flatten the arrays if they are multi-dimensional
                    x_data = x_data.flatten()
                    y_data = y_data.flatten()
                    
                    data_dict[f'{sample_name}_{x_field}'] = x_data
                    data_dict[f'{sample_name}_{y_field}'] = y_data
            
            df = pd.DataFrame(data_dict)
            df.to_csv(file_name, index=False)

    def select_all_samples(self):
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            item.setCheckState(Qt.Checked)
            self.active_samples.add(item.text())
        self.update_plot()
        
    def deselect_all_samples(self):
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            item.setCheckState(Qt.Unchecked)
            self.active_samples.discard(item.text())
        self.update_plot()

    def update_line_thickness(self):
        self.line_thickness = self.thickness_slider.value()
        self.update_plot()

    def update_legend_font_size(self):
        self.legend_font_size = self.font_size_spin.value()
        self.update_plot()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpectralPlotter()
    window.show()
    sys.exit(app.exec_())
