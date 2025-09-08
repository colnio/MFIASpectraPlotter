#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV Breakdown Plotter — PyQt5 + pyqtgraph

What it does
------------
• Choose a *parent* folder that contains many measurement subfolders (e.g. ./IV/<sample>_*_big_* ...).
• For each subfolder it loads CSVs from subfolder/data/*.csv with columns 'Voltage' and 'Current'.
• Computes electric field E = V / thickness / 1e8  (MV/cm) and current density j = I / A * 1e-4 (A/cm^2),
  where A is the device area in m^2 inferred from folder name tokens ('big'/'mid'/'small')
  with default areas editable in the UI (µm² -> converted to m²).
• Colors by size: big=red, mid=blue, small=green; unknown tokens fall back to a palette.
• Film thickness is *mandatory*.
• X-axis toggle: Voltage (V) or Field (MV/cm).
• Breakdown is detected per curve from dI/dV threshold: mean(dI/dV)+K*std(dI/dV) with V > Vmin.
• Shows mean±std for breakdown *voltage* and *field* across visible curves.
• Simple filters to include/exclude sizes and substring filter; manual hide via list checkboxes.
• Export the visible curves to CSV in tidy (long) format, using the currently selected X-axis.

Install
-------
pip install PyQt5 pyqtgraph numpy pandas matplotlib

Run
---
python iv_breakdown_gui.py
"""

import os
import sys
import glob
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QLabel, QGroupBox, QFormLayout,
    QDoubleSpinBox, QCheckBox, QLineEdit, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

# ---------------------------
# Helpers
# ---------------------------

def um2_to_m2(x_um2: float) -> float:
    return float(x_um2) * 1e-12


class Curve:
    def __init__(self, folder: str, size_tag: str, area_m2: float, voltage: np.ndarray, current: np.ndarray):
        self.folder = folder
        self.size_tag = size_tag
        self.area_m2 = area_m2
        self.V = voltage.astype(float)
        self.I = current.astype(float)
        self.j = self.I / (self.area_m2 if self.area_m2 > 0 else np.nan) * 1e-4  # A/cm^2
        # default thickness placeholder (needed for E calculation outside)
        self.E = None  # set later when thickness known
        self.bd_V = None

    def compute_field(self, thickness_m: float):
        if thickness_m <= 0:
            self.E = None
        else:
            self.E = self.V / thickness_m / 1e8  # MV/cm

    def detect_breakdown(self, k_sigma: float = 3.0, vmin: float = 2.0) -> float:
        if self.V.size < 10:
            self.bd_V = None
            return None
        dIdV = np.gradient(self.I, self.V)
        thr = np.nanmean(dIdV) + k_sigma * np.nanstd(dIdV)
        idxs = np.where((dIdV > thr) & (self.V > vmin))[0]
        if idxs.size:
            self.bd_V = float(self.V[idxs[-1]])
        else:
            self.bd_V = None
        return self.bd_V

# Formatter 

class Log10PowerAxis(pg.AxisItem):
    """Show labels only at decades as 10^n; hide others; no ×1e… suffix."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.enableAutoSIPrefix(False)   # new API
        except AttributeError:
            self.autoSIPrefix = False        # old API fallback

    def tickStrings(self, values, scale, spacing):
        out = []
        # In log mode, values are already log10(x); in linear mode they are x
        is_log = getattr(self, "logMode", False)

        for v in values:
            if not np.isfinite(v):
                out.append('')
                continue

            if is_log:
                # decade ticks are integers in log10 space: …, -1, 0, 1, 2, …
                if np.isclose(v, round(v), atol=1e-9):
                    n = int(round(v))
                    out.append(ten_to_sup(n))   # or use superscripts (see below)
                else:
                    out.append('')
            else:
                # linear axis: label only exact powers of 10 in linear space
                if v <= 0:
                    out.append('')
                    continue
                n = int(round(np.log10(v)))
                if np.isclose(v, 10**n, rtol=0, atol=10**n*1e-12):
                    out.append(ten_to_sup(n))
                else:
                    out.append('')
        return out

# optional pretty superscripts:
def ten_to_sup(n: int) -> str:
    tr = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return "10" + str(n).translate(tr)


# ---------------------------
# GUI
# ---------------------------

class IVBreakdownGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IV Breakdown Plotter")
        self.resize(1280, 820)

        # State
        self.parent_folder: str = ''
        self.curves: List[Curve] = []

        # Defaults
        self.area_um2_defaults: Dict[str, float] = {
            'big': 246447.0,
            'mid': 61428.0,
            'small': 8755.0,
        }
        self.fixed_colors = {'big': 'red', 'mid': 'blue', 'small': 'green'}

        self._init_ui()

    def _init_ui(self):
        central = QWidget(self); self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # Left: controls
        controls = QVBoxLayout(); main.addLayout(controls, 0)

        # Folder chooser
        file_grp = QGroupBox("Data folder")
        file_v = QVBoxLayout(file_grp)
        self.btn_folder = QPushButton("Choose parent folder…")
        self.btn_folder.clicked.connect(self.choose_parent_folder)
        self.lbl_folder = QLabel("(none)")
        file_v.addWidget(self.btn_folder); file_v.addWidget(self.lbl_folder)
        controls.addWidget(file_grp)

        # Physics params
        phys_grp = QGroupBox("Parameters")
        phys_form = QFormLayout(phys_grp)
        self.spin_thick_nm = QDoubleSpinBox(); self.spin_thick_nm.setDecimals(3); self.spin_thick_nm.setRange(0.0, 1e6); self.spin_thick_nm.setSingleStep(0.1); self.spin_thick_nm.setValue(0.0); self.spin_thick_nm.valueChanged.connect(self.update_plot)
        phys_form.addRow("Film thickness (nm) *", self.spin_thick_nm)

        self.cmb_xaxis = QComboBox(); self.cmb_xaxis.addItems(["Field (MV/cm)", "Voltage (V)"]); self.cmb_xaxis.currentIndexChanged.connect(self.update_plot)
        phys_form.addRow("X-axis", self.cmb_xaxis)

        # Areas (µm²)
        self.spin_big_um2 = QDoubleSpinBox(); self._setup_area(self.spin_big_um2, self.area_um2_defaults['big'])
        self.spin_mid_um2 = QDoubleSpinBox(); self._setup_area(self.spin_mid_um2, self.area_um2_defaults['mid'])
        self.spin_small_um2 = QDoubleSpinBox(); self._setup_area(self.spin_small_um2, self.area_um2_defaults['small'])
        phys_form.addRow("Area BIG (µm²)", self.spin_big_um2)
        phys_form.addRow("Area MID (µm²)", self.spin_mid_um2)
        phys_form.addRow("Area SMALL (µm²)", self.spin_small_um2)
        controls.addWidget(phys_grp)

        # Filters
        filt_grp = QGroupBox("Filters")
        filt_v = QVBoxLayout(filt_grp)
        self.edit_sub = QLineEdit(); self.edit_sub.setPlaceholderText("substring filter (case-insensitive)"); self.edit_sub.textChanged.connect(self._apply_filter)
        filt_v.addWidget(self.edit_sub)
        row = QHBoxLayout();
        self.chk_big = QCheckBox("big"); self.chk_big.setChecked(True); self.chk_big.stateChanged.connect(self.update_plot)
        self.chk_mid = QCheckBox("mid"); self.chk_mid.setChecked(True); self.chk_mid.stateChanged.connect(self.update_plot)
        self.chk_small = QCheckBox("small"); self.chk_small.setChecked(True); self.chk_small.stateChanged.connect(self.update_plot)
        row.addWidget(self.chk_big); row.addWidget(self.chk_mid); row.addWidget(self.chk_small); row.addStretch(1)
        filt_v.addLayout(row)
        controls.addWidget(filt_grp)

        # Breakdown detection params
        bd_grp = QGroupBox("Breakdown detection")
        bd_form = QFormLayout(bd_grp)
        self.spin_k = QDoubleSpinBox(); self.spin_k.setDecimals(2); self.spin_k.setRange(0.0, 100.0); self.spin_k.setValue(3.0); self.spin_k.valueChanged.connect(self.update_plot)
        bd_form.addRow("k·σ threshold", self.spin_k)
        self.spin_vmin = QDoubleSpinBox(); self.spin_vmin.setDecimals(2); self.spin_vmin.setRange(-1e3, 1e3); self.spin_vmin.setValue(2.0); self.spin_vmin.valueChanged.connect(self.update_plot)
        bd_form.addRow("Min V for BD (V)", self.spin_vmin)
        controls.addWidget(bd_grp)
        # Line width control
        self.spin_linewidth = QDoubleSpinBox()
        self.spin_linewidth.setDecimals(1)
        self.spin_linewidth.setRange(0.5, 10.0)
        self.spin_linewidth.setSingleStep(0.5)
        self.spin_linewidth.setValue(1.5)  # default
        self.spin_linewidth.valueChanged.connect(self.update_plot)
        phys_form.addRow("Line width", self.spin_linewidth)

        # Font size control
        self.spin_fontsize = QDoubleSpinBox()
        self.spin_fontsize.setDecimals(0)
        self.spin_fontsize.setRange(6, 32)
        self.spin_fontsize.setSingleStep(1)
        self.spin_fontsize.setValue(12)  # default
        self.spin_fontsize.valueChanged.connect(self.update_plot)
        phys_form.addRow("Font size", self.spin_fontsize)
        # Sample list
        lst_grp = QGroupBox("Subfolders (check to show)")
        lst_v = QVBoxLayout(lst_grp)
        self.btn_sel_all = QPushButton("Select All"); self.btn_sel_all.clicked.connect(self._select_all)
        self.btn_desel_all = QPushButton("Deselect All"); self.btn_desel_all.clicked.connect(self._deselect_all)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_sel_all); row2.addWidget(self.btn_desel_all); row2.addStretch(1)
        lst_v.addLayout(row2)
        self.list_folders = QListWidget(); self.list_folders.itemChanged.connect(self.update_plot)
        lst_v.addWidget(self.list_folders)
        controls.addWidget(lst_grp, 1)

        # Stats + export
        exp_grp = QGroupBox("Stats & Export")
        exp_v = QVBoxLayout(exp_grp)
        self.lbl_bd = QLabel("—")
        exp_v.addWidget(self.lbl_bd)
        self.btn_export = QPushButton("Export visible curves…"); self.btn_export.clicked.connect(self.export_visible)
        exp_v.addWidget(self.btn_export)
        controls.addWidget(exp_grp)

        # Right: plot
        left = Log10PowerAxis(orientation='left')
        right = QVBoxLayout(); main.addLayout(right, 1)
        self.plot = pg.PlotWidget(axisItems={'left' : left}); self.plot.setBackground('w'); self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'j (A/cm²)')
        self.plot.setLabel('bottom', 'Field (MV/cm)')
        self.plot.setLogMode(x=False, y=True)  # j on log scale
        right.addWidget(self.plot, 1)

        # Legend: add fixed-size dummies so colors are always visible
        self.legend = self.plot.addLegend()
        for tag, col in self.fixed_colors.items():
            # Add faint dummy lines for legend
            self.plot.plot([np.nan], [np.nan], pen=pg.mkPen(color=col, width=2), name=tag)

        self.statusBar().showMessage("Set thickness and choose a parent folder…")

    def _setup_area(self, spin: QDoubleSpinBox, default_um2: float):
        spin.setDecimals(1); spin.setRange(0.0, 1e12); spin.setSingleStep(10.0); spin.setValue(default_um2); spin.valueChanged.connect(self.update_plot)

    # ---------- Folder loading ----------
    def choose_parent_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select parent folder", "")
        if folder:
            self.parent_folder = folder
            self.lbl_folder.setText(os.path.basename(folder))
            self.load_curves_from_parent()

    def _infer_size_tag(self, name: str) -> str:
        s = name.lower()
        if 'big' in s: return 'big'
        if 'mid' in s: return 'mid'
        if 'small' in s: return 'small'
        return 'unknown'

    def _area_for_tag(self, tag: str) -> float:
        if tag == 'big': return um2_to_m2(self.spin_big_um2.value())
        if tag == 'mid': return um2_to_m2(self.spin_mid_um2.value())
        if tag == 'small': return um2_to_m2(self.spin_small_um2.value())
        # if unknown, default to MID
        return um2_to_m2(self.spin_mid_um2.value())

    def load_curves_from_parent(self):
        if not self.parent_folder:
            return
        # find immediate subfolders only
        subfolders = [os.path.join(self.parent_folder, d) for d in os.listdir(self.parent_folder)
                      if os.path.isdir(os.path.join(self.parent_folder, d))]
        curves: List[Curve] = []
        for folder in subfolders:
            size_tag = self._infer_size_tag(os.path.basename(folder))
            area_m2 = self._area_for_tag(size_tag)
            # data files pattern
            data_files = glob.glob(os.path.join(folder, 'data', '*.data'))
            for csvf in data_files:
                try:
                    df = pd.read_csv(csvf)
                    if not {'Voltage', 'Current'}.issubset(df.columns):
                        continue
                    V = df['Voltage'].to_numpy()
                    I = df['Current'].to_numpy()
                    curves.append(Curve(folder=os.path.basename(folder), size_tag=size_tag, area_m2=area_m2, voltage=V, current=I))
                except Exception:
                    continue
        self.curves = curves
        # Populate list with folders (unique)
        self._populate_folder_list(sorted(set(c.folder for c in self.curves)))
        self.update_plot()

    def _populate_folder_list(self, names: List[str]):
        self.list_folders.blockSignals(True)
        self.list_folders.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.list_folders.addItem(it)
        self.list_folders.blockSignals(False)

    # ---------- Filters ----------
    def _apply_filter(self):
        ss = self.edit_sub.text().strip().lower()
        for i in range(self.list_folders.count()):
            it = self.list_folders.item(i)
            it.setHidden(False if not ss else (ss not in it.text().lower()))
        self.update_plot()

    def _select_all(self):
        for i in range(self.list_folders.count()):
            it = self.list_folders.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Checked)
        self.update_plot()

    def _deselect_all(self):
        for i in range(self.list_folders.count()):
            it = self.list_folders.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Unchecked)
        self.update_plot()

    # ---------- Plotting ----------
    def update_plot(self):
        thick_nm = self.spin_thick_nm.value()
        if thick_nm <= 0:
            self.plot.clear()
            self.legend = self.plot.addLegend()
            for tag, col in self.fixed_colors.items():
                self.plot.plot([np.nan], [np.nan], pg.mkPen(color=col, width=self.spin_linewidth.value()), name=tag)
            self.plot.addItem(pg.TextItem("Set film thickness (nm)", color=(150, 0, 0)))
            self.lbl_bd.setText("—")
            return

        self.plot.clear()
        self.legend = self.plot.addLegend()
        for tag, col in self.fixed_colors.items():
            # self.plot.plot([np.nan], [np.nan], pen=pg.mkPen(color=col, width=2))
            self.plot.plot([np.nan], [np.nan], pen=pg.mkPen(color=col, width=self.spin_linewidth.value()), name=tag)

        # Axis
        use_field = (self.cmb_xaxis.currentText().startswith('Field'))
        self.plot.setLabel('bottom', 'Field (MV/cm)' if use_field else 'Voltage (V)')
        self.plot.setLogMode(x=False, y=True)

        # Visible folder names
        visible_names = set()
        for i in range(self.list_folders.count()):
            it = self.list_folders.item(i)
            if not it.isHidden() and it.checkState() == Qt.Checked:
                visible_names.add(it.text())

        # Palette for unknowns by folder name
        unk_names = sorted({c.folder for c in self.curves if c.size_tag == 'unknown'})
        pal = self._palette(len(unk_names) or 1)
        unk_color_by_name = {unk_names[i]: pal[i] for i in range(len(unk_names))}

        thick_m = thick_nm * 1e-9
        bd_volts: List[float] = []

        for c in self.curves:
            if c.folder not in visible_names:
                continue
            c.compute_field(thick_m)
            c.detect_breakdown(k_sigma=self.spin_k.value(), vmin=self.spin_vmin.value())

            col = self.fixed_colors.get(c.size_tag) or unk_color_by_name.get(c.folder, '#000000')
            pen = pg.mkPen(color=col, width=self.spin_linewidth.value())

            if use_field and c.E is not None:
                x = c.E; self.plot.plot(x, np.abs(c.j), pen=pen)
                # x = c.E; self.plot.plot(x, np.abs(c.j), pen=pen, name=c.folder)
            else:
                # x = c.V; self.plot.plot(x, np.abs(c.j), pen=pen, name=c.folder)
                x = c.V; self.plot.plot(x, np.abs(c.j), pen=pen)

            if c.bd_V is not None:
                bd_volts.append(c.bd_V)

        # Stats
        if bd_volts:
            bd_volts = np.array(bd_volts)
            mean_v = bd_volts.mean(); std_v = bd_volts.std(ddof=1) if bd_volts.size > 1 else 0.0
            mean_e = mean_v / thick_m / 1e8
            std_e = std_v / thick_m / 1e8
            self.lbl_bd.setText(f"Breakdown voltage: {mean_v:.2f} ± {std_v:.2f} V\nBreakdown field: {mean_e:.2f} ± {std_e:.2f} MV/cm")
        else:
            self.lbl_bd.setText("—")
        fontsize = int(self.spin_fontsize.value())
        font = QFont("Arial", fontsize)
        self.plot.setTitle(c.folder.split('_')[0], size = str(fontsize) + 'pt')

        axL = self.plot.getAxis('left')
        axB = self.plot.getAxis('bottom')

        # tick label font (older pyqtgraph: color handled by setTextPen / global foreground)
        axL.setStyle(tickFont=font)
        axB.setStyle(tickFont=font)

        # make sure axis lines and tick texts are black on older versions
        if hasattr(axL, "setTextPen"): axL.setTextPen(pg.mkPen('k'))
        if hasattr(axB, "setTextPen"): axB.setTextPen(pg.mkPen('k'))
        if hasattr(axL, "setPen"):     axL.setPen(pg.mkPen('k'))
        if hasattr(axB, "setPen"):     axB.setPen(pg.mkPen('k'))

        # axis titles
        axL.label.setFont(font)
        axB.label.setFont(font)

        # for sample in self.legend.items:  # legend.items = [(sample, label), ...]
        #     label = sample[1]        # second element is the text label
        #     label.setFont(font)

    def _palette(self, n: int) -> List[str]:
        cmap = plt.colormaps['tab20'] if n <= 20 else plt.colormaps['viridis']
        return [to_hex(cmap(i / max(n-1, 1))) for i in range(n)]

    # ---------- Export ----------
    def export_visible(self):
        if not self.curves:
            return
        from PyQt5.QtWidgets import QFileDialog
        fn, _ = QFileDialog.getSaveFileName(self, "Export visible curves to CSV", "iv_curves.csv", "CSV Files (*.csv)")
        if not fn:
            return
        thick_nm = self.spin_thick_nm.value(); thick_m = thick_nm * 1e-9
        use_field = (self.cmb_xaxis.currentText().startswith('Field'))

        visible_names = set()
        for i in range(self.list_folders.count()):
            it = self.list_folders.item(i)
            if not it.isHidden() and it.checkState() == Qt.Checked:
                visible_names.add(it.text())

        rows: List[Dict[str, Any]] = []
        for c in self.curves:
            if c.folder not in visible_names:
                continue
            c.compute_field(thick_m)
            X = c.E if use_field else c.V
            Xname = 'Field_MV_per_cm' if use_field else 'Voltage_V'
            for xi, ji in zip(X, np.abs(c.j)):
                rows.append({
                    'folder': c.folder,
                    'size_tag': c.size_tag,
                    'area_m2': c.area_m2,
                    Xname: float(xi),
                    'j_A_per_cm2': float(ji),
                    'thickness_nm': float(thick_nm),
                })
        if not rows:
            QMessageBox.information(self, "Nothing to export", "No visible curves.")
            return
        df = pd.DataFrame(rows)
        df.to_csv(fn, index=False)
        self.statusBar().showMessage(f"Exported {len(df)} rows to {os.path.basename(fn)}")


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, foreground='k')  # <- make text/ticks black globally
    # pg.setConfigOptions(antialias=True)
    win = IVBreakdownGUI(); win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
