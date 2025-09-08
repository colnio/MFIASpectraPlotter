#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabbed IV Breakdown + K‑Spectral GUI — PyQt5 + pyqtgraph

Layout (per your spec)
----------------------
LEFT (top):  Shared params — thickness, areas, font size, line width, base title
LEFT (bottom):  Tab switch (Breakdown / k spectra) — each tab shows its OWN file pickers,
                filters, and mode‑specific parameters
RIGHT:  One shared PlotWidget that updates according to the selected tab and shows only the plot
        (title included).

Install
-------
  pip install PyQt5 pyqtgraph numpy pandas matplotlib

Run
---
  python iv_k_tabbed_gui.py
"""

import os
import sys
import glob
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QListWidget, QLabel, QCheckBox, QGroupBox, QListWidgetItem,
    QLineEdit, QDoubleSpinBox, QFormLayout, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

EPS0 = 8.8541878128e-12  # F/m

# ---------------------------
# Shared helpers
# ---------------------------

def um2_to_m2(x_um2: float) -> float:
    return float(x_um2) * 1e-12


def ten_to_sup(n: int) -> str:
    tr = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return "10" + str(n).translate(tr)


class Log10PowerAxis(pg.AxisItem):
    """Show labels only at decades as 10^n; hide others; no ×1e… suffix."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.enableAutoSIPrefix(False)
        except AttributeError:
            self.autoSIPrefix = False

    def tickStrings(self, values, scale, spacing):
        out = []
        is_log = getattr(self, "logMode", False)
        for v in values:
            if not np.isfinite(v):
                out.append('')
                continue
            if is_log:
                if np.isclose(v, round(v), atol=1e-9):
                    n = int(round(v))
                    out.append(ten_to_sup(n))
                else:
                    out.append('')
            else:
                if v <= 0:
                    out.append('')
                    continue
                n = int(round(np.log10(v)))
                if np.isclose(v, 10**n, rtol=0, atol=10**n*1e-12):
                    out.append(ten_to_sup(n))
                else:
                    out.append('')
        return out


# ---------------------------
# Shared settings/signals
# ---------------------------
class SharedSettings(QObject):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.area_um2_defaults: Dict[str, float] = {
            'big': 246447.0,
            'mid': 61428.0,
            'small': 8755.0,
        }
        self.fixed_colors = {'big': 'red', 'mid': 'blue', 'small': 'green'}
        self.thickness_nm: float = 0.0
        self.area_big_um2: float = self.area_um2_defaults['big']
        self.area_mid_um2: float = self.area_um2_defaults['mid']
        self.area_small_um2: float = self.area_um2_defaults['small']
        self.line_width: float = 1.5
        self.font_size_pt: int = 12
        self.base_title: str = ''

    def area_for_tag_m2(self, tag: str) -> float:
        if tag == 'big':
            return um2_to_m2(self.area_big_um2)
        if tag == 'mid':
            return um2_to_m2(self.area_mid_um2)
        if tag == 'small':
            return um2_to_m2(self.area_small_um2)
        return um2_to_m2(self.area_mid_um2)


# ---------------------------
# BREAKDOWN logic (no internal PlotWidget; draws on shared_plot)
# ---------------------------
class Curve:
    def __init__(self, folder: str, size_tag: str, area_m2: float, voltage: np.ndarray, current: np.ndarray):
        self.folder = folder
        self.size_tag = size_tag
        self.area_m2 = area_m2
        self.V = voltage.astype(float)
        self.I = current.astype(float)
        self.j = self.I / (self.area_m2 if self.area_m2 > 0 else np.nan) * 1e-4  # A/cm^2
        self.E = None
        self.bd_V = None

    def compute_field(self, thickness_m: float):
        self.E = self.V / thickness_m / 1e8 if thickness_m > 0 else None

    def detect_breakdown(self, k_sigma: float = 3.0, vmin: float = 2.0) -> float:
        if self.V.size < 10:
            self.bd_V = None
            return None
        dIdV = np.gradient(self.I, self.V)
        thr = np.nanmean(dIdV) + k_sigma * np.nanstd(dIdV)
        idxs = np.where((dIdV > thr) & (self.V > vmin))[0]
        self.bd_V = float(self.V[idxs[-1]]) if idxs.size else None
        return self.bd_V


class BreakdownControls(QWidget):
    def __init__(self, shared: SharedSettings, shared_plot: pg.PlotWidget):
        super().__init__()
        self.shared = shared
        self.shared_plot = shared_plot
        self.parent_folder: str = ''
        self.curves: List[Curve] = []

        # UI
        root = QVBoxLayout(self)

        # Data + params
        file_grp = QGroupBox("Breakdown — Files & Params")
        file_v = QVBoxLayout(file_grp)
        row = QHBoxLayout()
        self.btn_folder = QPushButton("Choose parent folder…"); self.btn_folder.clicked.connect(self.choose_parent_folder)
        self.lbl_folder = QLabel("(none)")
        row.addWidget(self.btn_folder); row.addWidget(self.lbl_folder); row.addStretch(1)
        file_v.addLayout(row)

        form = QFormLayout()
        self.cmb_xaxis = QComboBox(); self.cmb_xaxis.addItems(["Field (MV/cm)", "Voltage (V)"]); self.cmb_xaxis.currentIndexChanged.connect(self.update_plot)
        form.addRow("X-axis", self.cmb_xaxis)
        self.spin_k = QDoubleSpinBox(); self.spin_k.setDecimals(2); self.spin_k.setRange(0.0, 100.0); self.spin_k.setValue(3.0); self.spin_k.valueChanged.connect(self.update_plot)
        form.addRow("k·σ threshold", self.spin_k)
        self.spin_vmin = QDoubleSpinBox(); self.spin_vmin.setDecimals(2); self.spin_vmin.setRange(-1e3, 1e3); self.spin_vmin.setValue(2.0); self.spin_vmin.valueChanged.connect(self.update_plot)
        form.addRow("Min V for BD (V)", self.spin_vmin)
        file_v.addLayout(form)

        # Filters + list
        filt_grp = QGroupBox("Filters")
        filt_v = QVBoxLayout(filt_grp)
        self.edit_sub = QLineEdit(); self.edit_sub.setPlaceholderText("substring filter (case-insensitive)"); self.edit_sub.textChanged.connect(self._apply_filter)
        filt_v.addWidget(self.edit_sub)
        row2 = QHBoxLayout();
        self.chk_big = QCheckBox("big"); self.chk_big.setChecked(True); self.chk_big.stateChanged.connect(self.update_plot)
        self.chk_mid = QCheckBox("mid"); self.chk_mid.setChecked(True); self.chk_mid.stateChanged.connect(self.update_plot)
        self.chk_small = QCheckBox("small"); self.chk_small.setChecked(True); self.chk_small.stateChanged.connect(self.update_plot)
        row2.addWidget(self.chk_big); row2.addWidget(self.chk_mid); row2.addWidget(self.chk_small); row2.addStretch(1)
        filt_v.addLayout(row2)

        lst_grp = QGroupBox("Subfolders (check to show)")
        lst_v = QVBoxLayout(lst_grp)
        row3 = QHBoxLayout();
        self.btn_sel_all = QPushButton("Select All"); self.btn_sel_all.clicked.connect(self._select_all)
        self.btn_desel_all = QPushButton("Deselect All"); self.btn_desel_all.clicked.connect(self._deselect_all)
        row3.addWidget(self.btn_sel_all); row3.addWidget(self.btn_desel_all); row3.addStretch(1)
        lst_v.addLayout(row3)
        self.list_folders = QListWidget(); self.list_folders.itemChanged.connect(self.update_plot)
        lst_v.addWidget(self.list_folders)

        # Stats & export
        exp_grp = QGroupBox("Stats & Export")
        exp_v = QVBoxLayout(exp_grp)
        self.lbl_bd = QLabel("—")
        self.btn_export = QPushButton("Export visible curves…"); self.btn_export.clicked.connect(self.export_visible)
        exp_v.addWidget(self.lbl_bd); exp_v.addWidget(self.btn_export)

        root.addWidget(file_grp)
        root.addWidget(filt_grp)
        root.addWidget(lst_grp, 1)
        root.addWidget(exp_grp)

        # react to shared changes
        self.shared.changed.connect(self.update_plot)

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

    def load_curves_from_parent(self):
        if not self.parent_folder:
            return
        subfolders = [os.path.join(self.parent_folder, d) for d in os.listdir(self.parent_folder)
                      if os.path.isdir(os.path.join(self.parent_folder, d))]
        curves: List[Curve] = []
        for folder in subfolders:
            size_tag = self._infer_size_tag(os.path.basename(folder))
            area_m2 = self.shared.area_for_tag_m2(size_tag)
            data_files = glob.glob(os.path.join(folder, 'data', '*.csv'))
            if not data_files:
                data_files = glob.glob(os.path.join(folder, 'data', '*.data'))
            for csvf in data_files:
                try:
                    df = pd.read_csv(csvf)
                    if not {'Voltage', 'Current'}.issubset(df.columns):
                        continue
                    V = df['Voltage'].to_numpy(); I = df['Current'].to_numpy()
                    curves.append(Curve(folder=os.path.basename(folder), size_tag=size_tag, area_m2=area_m2, voltage=V, current=I))
                except Exception:
                    continue
        self.curves = curves
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

    # ---------- Draw on shared plot ----------
    def update_plot(self):
        plot = self.shared_plot
        plot.clear()

        thick_nm = self.shared.thickness_nm
        if thick_nm <= 0:
            plot.addItem(pg.TextItem("Set shared film thickness (nm)", color=(150, 0, 0)))
            self.lbl_bd.setText("—")
            self._apply_title_and_fonts(mode_title="Breakdown")
            return

        # Legend dummies using shared colors
        legend = plot.addLegend()
        for tag, col in self.shared.fixed_colors.items():
            plot.plot([np.nan], [np.nan], pen=pg.mkPen(color=col, width=self.shared.line_width), name=tag)

        use_field = (self.cmb_xaxis.currentText().startswith('Field'))
        # ensure axis types: Y=Log10Power, X=normal
        pi = plot.getPlotItem()
        if hasattr(pi, 'setAxisItems'):
            pi.setAxisItems({'left': Log10PowerAxis(orientation='left'), 'bottom': pg.AxisItem(orientation='bottom')})
        plot.setLabel('left', 'j (A/cm²)')
        plot.setLabel('bottom', 'Field (MV/cm)' if use_field else 'Voltage (V)')
        plot.setLogMode(x=False, y=True)

        # Visible selection
        visible_names = {self.list_folders.item(i).text() for i in range(self.list_folders.count())
                         if (not self.list_folders.item(i).isHidden()) and self.list_folders.item(i).checkState() == Qt.Checked}

        # Palette for unknowns by folder name
        unk_names = sorted({c.folder for c in self.curves if c.size_tag == 'unknown'})
        pal = self._palette(len(unk_names) or 1)
        unk_color_by_name = {unk_names[i]: pal[i] for i in range(len(unk_names))}

        thick_m = thick_nm * 1e-9
        bd_volts: List[float] = []

        for c in self.curves:
            if c.folder not in visible_names:
                continue
            # update area from shared
            c.area_m2 = self.shared.area_for_tag_m2(c.size_tag)
            c.j = c.I / (c.area_m2 if c.area_m2 > 0 else np.nan) * 1e-4

            c.compute_field(thick_m)
            c.detect_breakdown(k_sigma=self.spin_k.value(), vmin=self.spin_vmin.value())

            col = self.shared.fixed_colors.get(c.size_tag) or unk_color_by_name.get(c.folder, '#000000')
            pen = pg.mkPen(color=col, width=self.shared.line_width)

            if use_field and c.E is not None:
                plot.plot(c.E, np.abs(c.j), pen=pen)
            else:
                plot.plot(c.V, np.abs(c.j), pen=pen)

            if c.bd_V is not None:
                bd_volts.append(c.bd_V)

        # Stats
        if bd_volts:
            arr = np.asarray(bd_volts, dtype=float)
            mean_v = float(np.nanmean(arr))
            std_v = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            mean_e = mean_v / thick_m / 1e8
            std_e = std_v / thick_m / 1e8
            self.lbl_bd.setText(f"""Breakdown voltage: {mean_v:.2f} ± {std_v:.2f} V
Breakdown field: {mean_e:.2f} ± {std_e:.2f} MV/cm""")
        else:
            self.lbl_bd.setText("—")

        self._apply_title_and_fonts(mode_title="Breakdown")

    def _apply_title_and_fonts(self, mode_title: str):
        fontsize = int(self.shared.font_size_pt)
        font = QFont("Arial", fontsize)
        base = self.shared.base_title.strip()
        title = (base + " — " if base else "") + mode_title
        self.shared_plot.setTitle(title, size=f"{fontsize}pt")
        axL = self.shared_plot.getAxis('left'); axB = self.shared_plot.getAxis('bottom')
        axL.setStyle(tickFont=font); axB.setStyle(tickFont=font)
        if hasattr(axL, "setTextPen"): axL.setTextPen(pg.mkPen('k'))
        if hasattr(axB, "setTextPen"): axB.setTextPen(pg.mkPen('k'))
        if hasattr(axL, "setPen"):     axL.setPen(pg.mkPen('k'))
        if hasattr(axB, "setPen"):     axB.setPen(pg.mkPen('k'))
        axL.label.setFont(font); axB.label.setFont(font)

    def _palette(self, n: int) -> List[str]:
        cmap = plt.colormaps['tab20'] if n <= 20 else plt.colormaps['viridis']
        return [to_hex(cmap(i / max(n-1, 1))) for i in range(n)]

    # ---------- Export ----------
    def export_visible(self):
        if not self.curves:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export visible curves to CSV", "iv_curves.csv", "CSV Files (*.csv)")
        if not fn:
            return
        thick_nm = self.shared.thickness_nm; thick_m = thick_nm * 1e-9
        use_field = (self.cmb_xaxis.currentText().startswith('Field'))

        visible_names = {self.list_folders.item(i).text() for i in range(self.list_folders.count())
                         if (not self.list_folders.item(i).isHidden()) and self.list_folders.item(i).checkState() == Qt.Checked}

        rows: List[Dict[str, Any]] = []
        for c in self.curves:
            if c.folder not in visible_names:
                continue
            X = (c.V if not use_field else (c.E if c.E is not None else []))
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
        pd.DataFrame(rows).to_csv(fn, index=False)


# ---------------------------
# K‑SPECTRAL logic (no internal PlotWidget; draws on shared_plot)
# ---------------------------

def load_header(header_file: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(header_file, 'r', newline='') as fh:
        reader = csv.reader(fh, delimiter=';')
        rows = list(reader)
    if not rows:
        return pairs
    for row in rows[1:]:
        if len(row) < 17:
            continue
        idx = row[0].strip()
        name = row[16].strip()
        pairs.append((idx, name))
    return pairs


def load_data_file(data_file: str) -> List[List[str]]:
    with open(data_file, 'r', newline='') as fd:
        reader = csv.reader(fd, delimiter=';')
        rows = list(reader)
    return rows[1:] if len(rows) > 1 else []


def parse_dataset(header_pairs: List[Tuple[str, str]], data_rows: List[List[str]]) -> Dict[str, Dict[str, List[np.ndarray]]]:
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    rows_by_idx: Dict[str, List[List[str]]] = {}
    for r in data_rows:
        if not r:
            continue
        idx = r[0].strip()
        rows_by_idx.setdefault(idx, []).append(r)

    for idx, sample_name in header_pairs:
        sample_rows = rows_by_idx.get(idx, [])
        if not sample_rows:
            continue
        sample_dict: Dict[str, List[np.ndarray]] = {}
        for r in sample_rows:
            if len(r) < 5:
                continue
            field = r[3].strip()
            try:
                values = np.array([float(x) for x in r[4:] if x != ''])
            except ValueError:
                continue
            sample_dict.setdefault(field, []).append(values)
        if sample_dict:
            out[sample_name] = sample_dict
    return out


def epsilon_r_from_c(c: np.ndarray, thickness_m: float, area_m2: float) -> np.ndarray:
    if thickness_m <= 0 or area_m2 <= 0:
        return np.full_like(c, np.nan)
    return c * thickness_m / (EPS0 * area_m2)


class KSpectralControls(QWidget):
    def __init__(self, shared: SharedSettings, shared_plot: pg.PlotWidget):
        super().__init__()
        self.shared = shared
        self.shared_plot = shared_plot
        self.header_file: str = ""
        self.data_file: str = ""
        self.dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.sample_names: List[str] = []

        main = QVBoxLayout(self)

        # Files
        file_group = QGroupBox("k spectra — Files")
        file_v = QVBoxLayout(file_group)
        row = QHBoxLayout()
        self.btn_header = QPushButton("Select Header CSV…"); self.btn_header.clicked.connect(self.on_select_header)
        self.lbl_header = QLabel("(none)")
        row.addWidget(self.btn_header); row.addWidget(self.lbl_header); row.addStretch(1)
        file_v.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_data = QPushButton("Select Data CSV…"); self.btn_data.clicked.connect(self.on_select_data)
        self.lbl_data = QLabel("(none)")
        row2.addWidget(self.btn_data); row2.addWidget(self.lbl_data); row2.addStretch(1)
        file_v.addLayout(row2)
        main.addWidget(file_group)

        # Filters
        filt_group = QGroupBox("Filters")
        filt_v = QVBoxLayout(filt_group)
        self.edit_substr = QLineEdit(); self.edit_substr.setPlaceholderText("substring filter (case‑insensitive)"); self.edit_substr.textChanged.connect(self._apply_sample_filter)
        filt_v.addWidget(self.edit_substr)
        bias_row = QHBoxLayout(); self.chk_0v = QCheckBox("0V"); self.chk_0v.setChecked(True); self.chk_0v.stateChanged.connect(self.update_plot); self.chk_1v = QCheckBox("1V"); self.chk_1v.setChecked(True); self.chk_1v.stateChanged.connect(self.update_plot)
        bias_row.addWidget(QLabel("Bias:")); bias_row.addWidget(self.chk_0v); bias_row.addWidget(self.chk_1v); bias_row.addStretch(1)
        filt_v.addLayout(bias_row)
        area_row = QHBoxLayout(); self.chk_big = QCheckBox("big"); self.chk_big.setChecked(True); self.chk_big.stateChanged.connect(self.update_plot); self.chk_mid = QCheckBox("mid"); self.chk_mid.setChecked(True); self.chk_mid.stateChanged.connect(self.update_plot); self.chk_small = QCheckBox("small"); self.chk_small.setChecked(True); self.chk_small.stateChanged.connect(self.update_plot)
        area_row.addWidget(QLabel("Areas:")); area_row.addWidget(self.chk_big); area_row.addWidget(self.chk_mid); area_row.addWidget(self.chk_small); area_row.addStretch(1)
        filt_v.addLayout(area_row)
        main.addWidget(filt_group)

        # Sample list
        sample_group = QGroupBox("Samples (check to show / uncheck to hide)")
        sample_v = QVBoxLayout(sample_group)
        btn_row = QHBoxLayout(); self.btn_sel_all = QPushButton("Select All"); self.btn_sel_all.clicked.connect(self._select_all); self.btn_desel_all = QPushButton("Deselect All"); self.btn_desel_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(self.btn_sel_all); btn_row.addWidget(self.btn_desel_all); btn_row.addStretch(1)
        sample_v.addLayout(btn_row)
        self.list_samples = QListWidget(); self.list_samples.itemChanged.connect(self.update_plot)
        sample_v.addWidget(self.list_samples)
        main.addWidget(sample_group, 1)

        # Export & stats
        exp_group = QGroupBox("Export & Stats")
        exp_form = QFormLayout(exp_group)
        self.spin_stat_freq = QDoubleSpinBox(); self.spin_stat_freq.setDecimals(0); self.spin_stat_freq.setRange(1.0, 1e12); self.spin_stat_freq.setValue(1e5); self.spin_stat_freq.valueChanged.connect(self.update_plot)
        exp_form.addRow("Stats @ freq (Hz)", self.spin_stat_freq)
        self.lbl_stats = QLabel("—"); exp_form.addRow("Mean±STD of visible k", self.lbl_stats)
        self.btn_export = QPushButton("Export visible k(F)…"); self.btn_export.clicked.connect(self.on_export)
        exp_form.addRow(self.btn_export)
        main.addWidget(exp_group)

        # react to shared changes
        self.shared.changed.connect(self.update_plot)

    # Files
    def on_select_header(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Header CSV", "", "CSV Files (*.csv)")
        if fn:
            self.header_file = fn; self.lbl_header.setText(os.path.basename(fn)); self._try_load_dataset()

    def on_select_data(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Data CSV", "", "CSV Files (*.csv)")
        if fn:
            self.data_file = fn; self.lbl_data.setText(os.path.basename(fn)); self._try_load_dataset()

    def _try_load_dataset(self):
        if not self.header_file or not self.data_file:
            return
        try:
            pairs = load_header(self.header_file)
            rows = load_data_file(self.data_file)
            self.dataset = parse_dataset(pairs, rows)
            self.sample_names = sorted(self.dataset.keys())
            # Default base title from first sample prefix
            if self.sample_names:
                default_base = self.sample_names[0].split('_')[0]
                if not self.shared.base_title:
                    self.shared.base_title = default_base
                    self.shared.changed.emit()
            self._populate_samples()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _populate_samples(self):
        self.list_samples.blockSignals(True)
        self.list_samples.clear()
        for name in self.sample_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_samples.addItem(item)
        self.list_samples.blockSignals(False)
        self.update_plot()

    # Filters
    def _apply_sample_filter(self):
        substr = self.edit_substr.text().strip().lower()
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            visible = (substr in item.text().lower()) if substr else True
            item.setHidden(not visible)
        self.update_plot()

    def _select_all(self):
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)
        self.update_plot()

    def _deselect_all(self):
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)
        self.update_plot()

    # Area helpers
    def _area_m2_for_sample(self, sample_name: str) -> Tuple[str, float]:
        s = sample_name.lower()
        if 'big' in s:   return 'big',   self.shared.area_for_tag_m2('big')
        if 'small' in s: return 'small', self.shared.area_for_tag_m2('small')
        return 'mid', self.shared.area_for_tag_m2('mid')

    # Draw on shared plot
    def update_plot(self):
        plot = self.shared_plot
        plot.clear()

        thickness_nm = self.shared.thickness_nm
        if thickness_nm <= 0:
            plot.addItem(pg.TextItem("Set shared film thickness (nm) to compute k", color=(150, 0, 0)))
            self.lbl_stats.setText("—")
            self._apply_title_and_fonts(mode_title="k spectra")
            return

        legend = plot.addLegend()
        for tag, col in self.shared.fixed_colors.items():
            plot.plot([np.nan], [np.nan], pen=pg.mkPen(color=col, width=self.shared.line_width), name=tag)

        # ensure axis types: X=Log10Power, Y=normal
        pi = plot.getPlotItem()
        if hasattr(pi, 'setAxisItems'):
            pi.setAxisItems({'left': pg.AxisItem(orientation='left'), 'bottom': Log10PowerAxis(orientation='bottom')})
        plot.setLabel('bottom', 'Frequency (Hz)')
        plot.setLabel('left', 'k')
        plot.setLogMode(x=True, y=False)

        want_0v = self.chk_0v.isChecked(); want_1v = self.chk_1v.isChecked()
        want_big = self.chk_big.isChecked(); want_mid = self.chk_mid.isChecked(); want_small = self.chk_small.isChecked()

        thickness_m = thickness_nm * 1e-9
        target_f = self.spin_stat_freq.value()
        ks_at_target: List[float] = []

        visible_items = [self.list_samples.item(i) for i in range(self.list_samples.count()) if not self.list_samples.item(i).isHidden()]
        palette_colors = self._generate_palette(len(visible_items) or 1)
        visible_names = [it.text() for it in visible_items]
        color_by_name = {visible_names[i]: palette_colors[i] for i in range(len(visible_names))}

        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            if item.isHidden() or item.checkState() != Qt.Checked:
                continue
            name = item.text(); name_l = name.lower()
            if (('0v' in name_l) and not want_0v) or (('1v' in name_l) and not want_1v):
                continue

            area_tag, area_m2 = self._area_m2_for_sample(name)
            if (area_tag == 'big' and not want_big) or (area_tag == 'mid' and not want_mid) or (area_tag == 'small' and not want_small):
                continue

            data = self.dataset.get(name, {})
            freqs_list = data.get('frequency', [])
            caps_list  = data.get('param1', [])
            if not freqs_list or not caps_list:
                continue

            col = self.shared.fixed_colors.get(area_tag, color_by_name.get(name, '#000000'))
            pen = pg.mkPen(color=col, width=self.shared.line_width)

            nseg = min(len(freqs_list), len(caps_list))
            for seg in range(nseg):
                f = freqs_list[seg]; c = caps_list[seg]
                if f.size == 0 or c.size == 0:
                    continue
                n = min(len(f), len(c)); f = f[:n]; c = c[:n]
                k = epsilon_r_from_c(c, thickness_m, area_m2)
                plot.plot(f, k, pen=pen)
                idx = int(np.argmin(np.abs(np.log10(f) - np.log10(target_f))))
                if 0 <= idx < len(k):
                    val = float(k[idx])
                    if np.isfinite(val):
                        ks_at_target.append(val)

        ks = np.array(ks_at_target, dtype=float)
        ks = ks[np.isfinite(ks)]
        if ks.size:
            self.lbl_stats.setText(f"{ks.mean():.2f} ± {ks.std(ddof=1) if ks.size>1 else 0.0:.2f}")
        else:
            self.lbl_stats.setText("—")

        self._apply_title_and_fonts(mode_title="k spectra")

    def _apply_title_and_fonts(self, mode_title: str):
        fontsize = int(self.shared.font_size_pt)
        font = QFont("Arial", fontsize)
        base = self.shared.base_title.strip()
        title = (base + " — " if base else "") + mode_title
        self.shared_plot.setTitle(title, size=f"{fontsize}pt")
        axL = self.shared_plot.getAxis('left'); axB = self.shared_plot.getAxis('bottom')
        axL.setStyle(tickFont=font); axB.setStyle(tickFont=font)
        if hasattr(axL, "setTextPen"): axL.setTextPen(pg.mkPen('k'))
        if hasattr(axB, "setTextPen"): axB.setTextPen(pg.mkPen('k'))
        if hasattr(axL, "setPen"):     axL.setPen(pg.mkPen('k'))
        if hasattr(axB, "setPen"):     axB.setPen(pg.mkPen('k'))
        axL.label.setFont(font); axB.label.setFont(font)

    def _generate_palette(self, n):
        cmap = plt.colormaps['tab20'] if n <= 20 else plt.colormaps['viridis']
        return [to_hex(cmap(i / max(n-1, 1))) for i in range(n)]

    # Export
    def on_export(self):
        if not self.dataset:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export visible k(F) to CSV", "k_spectra.csv", "CSV Files (*.csv)")
        if not fn:
            return

        thickness_nm = self.shared.thickness_nm
        if thickness_nm <= 0:
            QMessageBox.warning(self, "Missing thickness", "Set shared film thickness (nm) before exporting.")
            return
        thickness_m = thickness_nm * 1e-9

        want_0v = self.chk_0v.isChecked(); want_1v = self.chk_1v.isChecked()
        want_big = self.chk_big.isChecked(); want_mid = self.chk_mid.isChecked(); want_small = self.chk_small.isChecked()

        rows: List[Dict[str, Any]] = []
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            if item.isHidden() or item.checkState() != Qt.Checked:
                continue
            name = item.text(); name_l = name.lower()
            if (('0v' in name_l) and not want_0v) or (('1v' in name_l) and not want_1v):
                continue
            area_tag, area_m2 = self._area_m2_for_sample(name)
            if (area_tag == 'big' and not want_big) or (area_tag == 'mid' and not want_mid) or (area_tag == 'small' and not want_small):
                continue
            data = self.dataset.get(name, {})
            freqs_list = data.get('frequency', [])
            caps_list  = data.get('param1', [])
            if not freqs_list or not caps_list:
                continue
            nseg = min(len(freqs_list), len(caps_list))
            for seg in range(nseg):
                f = freqs_list[seg]; c = caps_list[seg]
                n = min(len(f), len(c)); f = f[:n]; c = c[:n]
                k = epsilon_r_from_c(c, thickness_m, area_m2)
                for fi, ki in zip(f, k):
                    rows.append({
                        'sample': name,
                        'area_tag': area_tag,
                        'area_m2': area_m2,
                        'frequency_Hz': float(fi),
                        'k': float(ki),
                        'film_thickness_nm': thickness_nm,
                    })
        if not rows:
            QMessageBox.information(self, "Nothing to export", "No visible curves matched filters.")
            return
        pd.DataFrame(rows).to_csv(fn, index=False)


# ---------------------------
# Main Window with LEFT controls + bottom-left tab switch, RIGHT shared plot
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IV + k Spectra")
        self.resize(1400, 860)

        self.shared = SharedSettings()

        central = QWidget(self); self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # RIGHT: shared plot only
        left = QVBoxLayout(); main.addLayout(left, 0)
        # create the shared plot first so controls can draw on it
        self.shared_plot = pg.PlotWidget()
        self.shared_plot.setBackground('w')
        self.shared_plot.showGrid(x=True, y=True, alpha=0.3)
        main.addWidget(self.shared_plot, 1)

        # LEFT (top): Shared params
        shared_grp = QGroupBox("Shared Parameters")
        form = QFormLayout(shared_grp)

        self.spin_thick_nm = QDoubleSpinBox(); self.spin_thick_nm.setDecimals(3); self.spin_thick_nm.setRange(0.0, 1e6); self.spin_thick_nm.setSingleStep(0.1); self.spin_thick_nm.setValue(self.shared.thickness_nm); self.spin_thick_nm.valueChanged.connect(self._on_shared_changed)
        form.addRow("Film thickness (nm) *", self.spin_thick_nm)

        self.spin_big_um2 = QDoubleSpinBox(); self._setup_area(self.spin_big_um2, self.shared.area_big_um2)
        self.spin_mid_um2 = QDoubleSpinBox(); self._setup_area(self.spin_mid_um2, self.shared.area_mid_um2)
        self.spin_small_um2 = QDoubleSpinBox(); self._setup_area(self.spin_small_um2, self.shared.area_small_um2)
        form.addRow("Area BIG (µm²)", self.spin_big_um2)
        form.addRow("Area MID (µm²)", self.spin_mid_um2)
        form.addRow("Area SMALL (µm²)", self.spin_small_um2)

        self.spin_linewidth = QDoubleSpinBox(); self.spin_linewidth.setDecimals(1); self.spin_linewidth.setRange(0.5, 10.0); self.spin_linewidth.setSingleStep(0.5); self.spin_linewidth.setValue(self.shared.line_width); self.spin_linewidth.valueChanged.connect(self._on_shared_changed)
        form.addRow("Line width", self.spin_linewidth)

        self.spin_fontsize = QDoubleSpinBox(); self.spin_fontsize.setDecimals(0); self.spin_fontsize.setRange(6, 32); self.spin_fontsize.setSingleStep(1); self.spin_fontsize.setValue(self.shared.font_size_pt); self.spin_fontsize.valueChanged.connect(self._on_shared_changed)
        form.addRow("Font size", self.spin_fontsize)

        self.edit_title = QLineEdit(); self.edit_title.setPlaceholderText("Base title (applies to both tabs)"); self.edit_title.textChanged.connect(self._on_shared_changed)
        form.addRow("Base Title", self.edit_title)

        left.addWidget(shared_grp)

        # LEFT (bottom): bottom-left tab bar with mode-specific controls
        tabs = QTabWidget();
        tabs.setTabPosition(QTabWidget.North)  # bottom-left placement
        tabs.setDocumentMode(True)
        tabs.setMovable(False)

        self.break_controls = BreakdownControls(self.shared, self.shared_plot)
        self.k_controls = KSpectralControls(self.shared, self.shared_plot)

        tabs.addTab(self.break_controls, "Breakdown")
        tabs.addTab(self.k_controls, "k spectra")

        # when user switches tabs, redraw appropriate content on the shared plot
        def _on_tab_changed(idx: int):
            if idx == 0:
                self.break_controls.update_plot()
            else:
                self.k_controls.update_plot()
        tabs.currentChanged.connect(_on_tab_changed)

        left.addWidget(tabs, 1)

        self.statusBar().showMessage("Set shared params, then use the bottom-left tab to load data and plot.")

    def _setup_area(self, spin: QDoubleSpinBox, default_um2: float):
        spin.setDecimals(1); spin.setRange(0.0, 1e12); spin.setSingleStep(10.0); spin.setValue(default_um2); spin.valueChanged.connect(self._on_shared_changed)

    def _on_shared_changed(self, *args):
        self.shared.thickness_nm = self.spin_thick_nm.value()
        self.shared.area_big_um2 = self.spin_big_um2.value()
        self.shared.area_mid_um2 = self.spin_mid_um2.value()
        self.shared.area_small_um2 = self.spin_small_um2.value()
        self.shared.line_width = self.spin_linewidth.value()
        self.shared.font_size_pt = int(self.spin_fontsize.value())
        self.shared.base_title = self.edit_title.text()
        self.shared.changed.emit()


# ---------------------------
# Entry point
# ---------------------------
def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, foreground='k')
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
