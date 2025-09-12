#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Spectral GUI — PyQt5 + pyqtgraph (v3.2)

This patch:
• Removes axis range controls (X/Y min/max). Autoscale is used; log toggles remain.
• Stats input adapts to plot mode:
    - k(F): "Stats @ freq (Hz)" (positive). Nearest picked in log10(f) space.
    - k(V): "Stats @ voltage (V)" (negative or positive). Nearest picked in linear V space.
• k(V) keeps X linear and uses plain numeric X-axis labels (no 10^n).
"""

import sys
import os
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QListWidget, QLabel, QGroupBox, QListWidgetItem,
    QLineEdit, QDoubleSpinBox, QFormLayout, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg

EPS0 = 8.8541878128e-12  # F/m


# ---------------------------
# Data loading/parsing helpers
# ---------------------------

def load_header(header_file: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(header_file, 'r', newline='') as fh:
        reader = csv.reader(fh, delimiter=';')
        rows = list(reader)
    if not rows:
        return pairs
    for row in rows[1:]:  # skip header
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
            field = r[3].strip().lower()
            try:
                values = np.array([float(x) for x in r[4:] if x != ''])
            except ValueError:
                continue
            sample_dict.setdefault(field, []).append(values)
        if sample_dict:
            out[sample_name] = sample_dict
    return out


# ---------------------------
# Physics helpers
# ---------------------------

def account_for_oxide_series(c: np.ndarray, area_m2: float, d_ox_m: float, eps_ox: float) -> np.ndarray:
    if d_ox_m <= 0 or eps_ox <= 0:
        return c
    c_ox = EPS0 * eps_ox * area_m2 / d_ox_m
    denom = (c_ox - c)
    safe = np.abs(denom) > np.finfo(float).eps
    c_corr = np.copy(c)
    c_corr[safe] = (c_ox * c[safe]) / denom[safe]
    return c_corr


def epsilon_r_from_c(c: np.ndarray, thickness_m: float, area_m2: float) -> np.ndarray:
    if thickness_m <= 0 or area_m2 <= 0:
        return np.full_like(c, np.nan)
    return c * thickness_m / (EPS0 * area_m2)


# ---------------------------
# Formatter (10^n on decades, plain numbers on demand)
# ---------------------------

class Log10PowerAxis(pg.AxisItem):
    """
    Show labels only at decades as 10^n when log; allow plain numeric labels in linear mode.
    Set self.force_plain_linear_labels = True to use default numeric formatting when not log.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_plain_linear_labels = False
        try:
            self.enableAutoSIPrefix(False)   # new API
        except AttributeError:
            self.autoSIPrefix = False        # old API fallback

    def tickStrings(self, values, scale, spacing):
        is_log = getattr(self, "logMode", False)
        if (not is_log) and self.force_plain_linear_labels:
            return pg.AxisItem.tickStrings(self, values, scale, spacing)

        out = []
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
                # linear axis: show only exact powers of 10 if we are *not* forced to plain labels
                if v <= 0:
                    out.append('')
                    continue
                n = int(round(np.log10(v)))
                out.append(ten_to_sup(n) if np.isclose(v, 10**n, rtol=0, atol=10**n*1e-12) else '')
        return out


def ten_to_sup(n: int) -> str:
    tr = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return "10" + str(n).translate(tr)


# ---------------------------
# GUI application
# ---------------------------

class KSpectralGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K-Spectral Plotter")
        self.resize(1280, 840)

        self.header_file: str = ""
        self.data_file: str = ""
        self.dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.sample_names: List[str] = []

        # Areas (µm²)
        self.area_um2_defaults_square = {'big': 246447.0, 'mid': 61428.0, 'small': 8755.0}
        self.area_um2_defaults_circle = {'big': 771786.0, 'mid': 192180.0, 'small': 67400.0, 'little': 28508.0, 'tiny': 6333.0}

        # Fixed colors by size
        self.fixed_colors = {'big': 'red', 'mid': 'blue', 'small': 'green', 'little': 'orange', 'tiny': 'purple'}

        self._init_ui()

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # ---------------- Left Controls ----------------
        control = QVBoxLayout()
        main.addLayout(control, 0)

        # Warning banner
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        pal = self.warn_label.palette()
        pal.setColor(QPalette.WindowText, QColor(180, 0, 0))
        self.warn_label.setPalette(pal)
        font_warn = QFont(); font_warn.setPointSize(10); font_warn.setBold(True)
        self.warn_label.setFont(font_warn)
        control.addWidget(self.warn_label)

        # Files
        file_group = QGroupBox("Files")
        file_v = QVBoxLayout(file_group)
        self.btn_header = QPushButton("Select Header CSV…"); self.btn_header.clicked.connect(self.on_select_header)
        self.lbl_header = QLabel("(none)")
        self.btn_data = QPushButton("Select Data CSV…"); self.btn_data.clicked.connect(self.on_select_data)
        self.lbl_data = QLabel("(none)")
        file_v.addWidget(self.btn_header); file_v.addWidget(self.lbl_header)
        file_v.addWidget(self.btn_data); file_v.addWidget(self.lbl_data)
        control.addWidget(file_group)

        # Film & oxide
        phys_group = QGroupBox("Film & Oxide")
        phys_form = QFormLayout(phys_group)

        self.spin_thickness_nm = QDoubleSpinBox()
        self.spin_thickness_nm.setDecimals(3); self.spin_thickness_nm.setRange(0.0, 1e6)
        self.spin_thickness_nm.setSingleStep(0.1); self.spin_thickness_nm.setValue(0.0)
        self.spin_thickness_nm.valueChanged.connect(self.update_plot)
        phys_form.addRow("Film thickness (nm) *", self.spin_thickness_nm)

        self.chk_oxide = QCheckBox("Account for oxide (series C)")
        self.chk_oxide.setChecked(True); self.chk_oxide.stateChanged.connect(self.update_plot)
        phys_form.addRow(self.chk_oxide)

        self.spin_dox_nm = QDoubleSpinBox()
        self.spin_dox_nm.setDecimals(3); self.spin_dox_nm.setRange(0.0, 1e6)
        self.spin_dox_nm.setSingleStep(0.1); self.spin_dox_nm.setValue(1.7)
        self.spin_dox_nm.valueChanged.connect(self.update_plot)
        phys_form.addRow("Oxide thickness (nm)", self.spin_dox_nm)

        self.spin_eps_ox = QDoubleSpinBox()
        self.spin_eps_ox.setDecimals(3); self.spin_eps_ox.setRange(0.0, 1e6)
        self.spin_eps_ox.setSingleStep(0.1); self.spin_eps_ox.setValue(3.9)
        self.spin_eps_ox.valueChanged.connect(self.update_plot)
        phys_form.addRow("Oxide εr", self.spin_eps_ox)

        # Line width
        self.spin_linewidth = QDoubleSpinBox()
        self.spin_linewidth.setDecimals(1); self.spin_linewidth.setRange(0.5, 10.0)
        self.spin_linewidth.setSingleStep(0.5); self.spin_linewidth.setValue(1.5)
        self.spin_linewidth.valueChanged.connect(self.update_plot)
        phys_form.addRow("Line width", self.spin_linewidth)

        # Font size
        self.spin_fontsize = QDoubleSpinBox()
        self.spin_fontsize.setDecimals(0); self.spin_fontsize.setRange(6, 32)
        self.spin_fontsize.setSingleStep(1); self.spin_fontsize.setValue(12)
        self.spin_fontsize.valueChanged.connect(self.update_plot)
        phys_form.addRow("Font size", self.spin_fontsize)

        control.addWidget(phys_group)

        # Plot Mode & Mask & Grid
        mm_group = QGroupBox("Plot mode & Mask")
        mm_form = QFormLayout(mm_group)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["k(F)", "k(V)"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        mm_form.addRow("Plot:", self.combo_mode)

        self.combo_mask = QComboBox()
        self.combo_mask.addItems(["Square mask", "Circle mask"])
        self.combo_mask.currentIndexChanged.connect(self.on_mask_changed)
        mm_form.addRow("Mask:", self.combo_mask)

        self.chk_grid = QCheckBox("Show grid"); self.chk_grid.setChecked(False)
        self.chk_grid.stateChanged.connect(self.on_grid_toggled)
        mm_form.addRow(self.chk_grid)

        control.addWidget(mm_group)

        # Areas (labels kept so we can hide/show cleanly)
        area_group = QGroupBox("Areas (µm²) — token-based detection in sample name")
        area_form = QFormLayout(area_group)

        self.lbl_area_big = QLabel("BIG");    self.lbl_area_mid = QLabel("MID")
        self.lbl_area_small = QLabel("SMALL"); self.lbl_area_little = QLabel("LITTLE"); self.lbl_area_tiny = QLabel("TINY")

        self.spin_area_big_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_big_um2, 0.0)
        self.spin_area_mid_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_mid_um2, 0.0)
        self.spin_area_small_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_small_um2, 0.0)
        self.spin_area_little_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_little_um2, 0.0)
        self.spin_area_tiny_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_tiny_um2, 0.0)

        area_form.addRow(self.lbl_area_big, self.spin_area_big_um2)
        area_form.addRow(self.lbl_area_mid, self.spin_area_mid_um2)
        area_form.addRow(self.lbl_area_small, self.spin_area_small_um2)
        area_form.addRow(self.lbl_area_little, self.spin_area_little_um2)
        area_form.addRow(self.lbl_area_tiny, self.spin_area_tiny_um2)
        control.addWidget(area_group)

        # Filter (substring only)
        filt_group = QGroupBox("Filter")
        filt_v = QVBoxLayout(filt_group)
        self.edit_substr = QLineEdit(); self.edit_substr.setPlaceholderText("substring filter (case-insensitive)")
        self.edit_substr.textChanged.connect(self._apply_sample_filter)
        filt_v.addWidget(self.edit_substr)
        control.addWidget(filt_group)

        # Samples list
        sample_group = QGroupBox("Samples (check to show / uncheck to hide)")
        sample_v = QVBoxLayout(sample_group)
        btn_row = QHBoxLayout()
        self.btn_sel_all = QPushButton("Select All"); self.btn_sel_all.clicked.connect(self._select_all)
        self.btn_desel_all = QPushButton("Deselect All"); self.btn_desel_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(self.btn_sel_all); btn_row.addWidget(self.btn_desel_all); btn_row.addStretch(1)
        sample_v.addLayout(btn_row)
        self.list_samples = QListWidget(); self.list_samples.itemChanged.connect(self.update_plot)
        sample_v.addWidget(self.list_samples)
        control.addWidget(sample_group, stretch=1)

        # Stats (adaptive)
        stat_group = QGroupBox("Stats")
        stat_form = QFormLayout(stat_group)

        self.lbl_stat_prompt = QLabel("Stats @ freq (Hz)")
        self.spin_stat_value = QDoubleSpinBox()
        self._setup_stat_for_freq()  # default
        self.spin_stat_value.valueChanged.connect(self.update_plot)

        stat_form.addRow(self.lbl_stat_prompt, self.spin_stat_value)
        self.lbl_stats = QLabel("—")
        stat_form.addRow("Mean±STD of visible k", self.lbl_stats)
        control.addWidget(stat_group)

        # ---------------- Right Plot ----------------
        right = QVBoxLayout(); main.addLayout(right, 1)

        # Plot with custom bottom axis
        self.bottom_axis = Log10PowerAxis(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': self.bottom_axis})
        self.plot.setBackground('w')
        self.plot.showGrid(x=False, y=False, alpha=0.3)
        self.plot.setLabel('bottom', 'Frequency (Hz)')
        self.plot.setLabel('left', 'k')
        self.plot.setLogMode(x=True, y=False)
        right.addWidget(self.plot, 1)

        # Axis toggles (kept)
        axis_group = QGroupBox("Axes")
        axis_form = QFormLayout(axis_group)
        self.chk_logx = QCheckBox("log X"); self.chk_logx.setChecked(True); self.chk_logx.stateChanged.connect(self._toggle_logx)
        self.chk_logy = QCheckBox("log Y"); self.chk_logy.setChecked(False); self.chk_logy.stateChanged.connect(self._toggle_logy)
        toggles = QHBoxLayout(); toggles.addWidget(self.chk_logx); toggles.addWidget(self.chk_logy); toggles.addStretch(1)
        axis_form.addRow(toggles)
        right.addWidget(axis_group)

        self.statusBar().showMessage("Set film thickness and load files to begin…")
        self._apply_mask_defaults()

    # ---------- helpers (UI setup) ----------

    def _setup_area_spin(self, spin: QDoubleSpinBox, default_um2: float):
        spin.setDecimals(3); spin.setRange(0.0, 1e12); spin.setSingleStep(10.0); spin.setValue(default_um2)
        spin.valueChanged.connect(self.update_plot)

    def _setup_stat_for_freq(self):
        self.lbl_stat_prompt.setText("Stats @ freq (Hz)")
        self.spin_stat_value.blockSignals(True)
        self.spin_stat_value.setDecimals(0)
        self.spin_stat_value.setRange(1.0, 1e12)
        self.spin_stat_value.setSingleStep(10.0)
        if self.spin_stat_value.value() <= 0:
            self.spin_stat_value.setValue(1e5)
        self.spin_stat_value.blockSignals(False)

    def _setup_stat_for_voltage(self):
        self.lbl_stat_prompt.setText("Stats @ voltage (V)")
        self.spin_stat_value.blockSignals(True)
        self.spin_stat_value.setDecimals(6)
        self.spin_stat_value.setRange(-1e6, 1e6)  # allow negative & positive
        self.spin_stat_value.setSingleStep(0.01)
        if self.spin_stat_value.value() == 0:
            self.spin_stat_value.setValue(0.0)
        self.spin_stat_value.blockSignals(False)

    # ---------- Files ----------

    def on_select_header(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Header CSV", "", "CSV Files (*.csv)")
        if fn:
            self.header_file = fn; self.lbl_header.setText(os.path.basename(fn))
            self._check_dirs_warning()
            self._try_load_dataset()

    def on_select_data(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Data CSV", "", "CSV Files (*.csv)")
        if fn:
            self.data_file = fn; self.lbl_data.setText(os.path.basename(fn))
            self._check_dirs_warning()
            self._try_load_dataset()

    def _check_dirs_warning(self):
        if self.header_file and self.data_file:
            d1 = os.path.abspath(os.path.dirname(self.header_file))
            d2 = os.path.abspath(os.path.dirname(self.data_file))
            self.warn_label.setText("⚠ Header and Data files are in DIFFERENT folders." if d1 != d2 else "")

    def _try_load_dataset(self):
        if not self.header_file or not self.data_file:
            return
        try:
            pairs = load_header(self.header_file)
            rows = load_data_file(self.data_file)
            self.dataset = parse_dataset(pairs, rows)
            self.sample_names = sorted(self.dataset.keys())
            self._populate_samples()
            self.statusBar().showMessage(f"Loaded {len(self.sample_names)} samples.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self.statusBar().showMessage("Load failed.")

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

    # ---------- Filters ----------

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

    # ---------- Mask / Mode / Grid ----------

    def on_mask_changed(self, _=None):
        self._apply_mask_defaults()
        self.update_plot()

    def _apply_mask_defaults(self):
        circle = (self.combo_mask.currentText() == "Circle mask")
        if circle:
            d = self.area_um2_defaults_circle
            self.spin_area_big_um2.setValue(d['big'])
            self.spin_area_mid_um2.setValue(d['mid'])
            self.spin_area_small_um2.setValue(d['small'])
            self.spin_area_little_um2.setValue(d['little'])
            self.spin_area_tiny_um2.setValue(d['tiny'])
        else:
            d = self.area_um2_defaults_square
            self.spin_area_big_um2.setValue(d['big'])
            self.spin_area_mid_um2.setValue(d['mid'])
            self.spin_area_small_um2.setValue(d['small'])

        # Hide/show ONLY little/tiny rows (both label & field)
        self.lbl_area_little.setVisible(circle); self.spin_area_little_um2.setVisible(circle)
        self.lbl_area_tiny.setVisible(circle);   self.spin_area_tiny_um2.setVisible(circle)

    def on_mode_changed(self, _=None):
        mode = self.combo_mode.currentText()
        if mode == "k(V)":
            # force linear X and plain numeric ticks
            self.chk_logx.blockSignals(True); self.chk_logx.setChecked(False); self.chk_logx.blockSignals(False)
            self.chk_logx.setEnabled(False)
            self.bottom_axis.force_plain_linear_labels = True
            self.plot.setLabel('bottom', 'Bias (V)')
            self._setup_stat_for_voltage()
        else:
            self.chk_logx.setEnabled(True)
            self.bottom_axis.force_plain_linear_labels = False
            self.plot.setLabel('bottom', 'Frequency (Hz)')
            self._setup_stat_for_freq()
        self.update_plot()

    def on_grid_toggled(self):
        show = self.chk_grid.isChecked()
        self.plot.showGrid(x=show, y=show, alpha=0.3)

    # ---------- Axes ----------

    def _toggle_logx(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())

    def _toggle_logy(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())

    # ---------- Area helpers ----------

    def _area_m2_for_sample(self, sample_name: str) -> Tuple[str, float]:
        s = sample_name.lower()
        circle = (self.combo_mask.currentText() == "Circle mask")
        if 'tiny' in s and circle:
            return 'tiny', self.spin_area_tiny_um2.value() * 1e-12
        if 'little' in s and circle:
            return 'little', self.spin_area_little_um2.value() * 1e-12
        if 'big' in s:
            return 'big', self.spin_area_big_um2.value() * 1e-12
        if 'small' in s:
            return 'small', self.spin_area_small_um2.value() * 1e-12
        return 'mid', self.spin_area_mid_um2.value() * 1e-12

    # ---------- Core compute & plot ----------

    def update_plot(self):
        self.plot.clear()

        # legend reflecting the selected mask
        legend = self.plot.addLegend()
        mask = self.combo_mask.currentText()
        sizes_for_mask = ['big', 'mid', 'small'] if mask == "Square mask" else ['big', 'mid', 'small', 'little', 'tiny']
        for tag in sizes_for_mask:
            self.plot.plot([np.nan], [np.nan],
                           pen=pg.mkPen(color=self.fixed_colors[tag], width=self.spin_linewidth.value()),
                           name=tag)

        self.on_grid_toggled()

        thickness_nm = self.spin_thickness_nm.value()
        if thickness_nm <= 0:
            self.plot.addItem(pg.TextItem("Set film thickness (nm) to compute k", color=(150, 0, 0)))
            self.statusBar().showMessage("Film thickness required.")
            self.lbl_stats.setText("—")
            self._apply_fonts()
            return

        if not self.dataset:
            self.lbl_stats.setText("—")
            self._apply_fonts()
            return

        # Mode
        mode = self.combo_mode.currentText()
        if mode == "k(V)":
            self.plot.setLogMode(x=False, y=self.chk_logy.isChecked())
            self.plot.setLabel('bottom', 'Bias (V)')
            self.bottom_axis.force_plain_linear_labels = True
        else:
            self.chk_logx.setChecked(True)
            self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())
            self.plot.setLabel('bottom', 'Frequency (Hz)')
            self.bottom_axis.force_plain_linear_labels = False

        self.plot.setLabel('left', 'k')

        thickness_m = thickness_nm * 1e-9
        use_oxide = self.chk_oxide.isChecked()
        dox_m = self.spin_dox_nm.value() * 1e-9
        eps_ox = self.spin_eps_ox.value()

        stat_value = self.spin_stat_value.value()  # Hz or V depending on mode
        ks_at_target = []

        visible_items = [self.list_samples.item(i) for i in range(self.list_samples.count())
                         if not self.list_samples.item(i).isHidden() and self.list_samples.item(i).checkState() == Qt.Checked]
        visible_names = [it.text() for it in visible_items]

        for name in visible_names:
            area_tag, area_m2 = self._area_m2_for_sample(name)
            if (mask == "Square mask") and (area_tag in ('little', 'tiny')):
                continue

            data = self.dataset.get(name, {})
            freqs_list = data.get('frequency', [])
            caps_list  = data.get('param1',   [])
            bias_list  = data.get('bias',     [])  # for k(V)
            if not caps_list:
                continue

            pen = pg.mkPen(color=self.fixed_colors.get(area_tag, 'k'), width=self.spin_linewidth.value())

            if mode == "k(F)":
                if not freqs_list:
                    continue
                nseg = min(len(freqs_list), len(caps_list))
                for seg in range(nseg):
                    f = np.asarray(freqs_list[seg]); c = np.asarray(caps_list[seg])
                    if f.size == 0 or c.size == 0:
                        continue
                    n = min(len(f), len(c))
                    f = f[:n]; c = c[:n]
                    c_eff = account_for_oxide_series(c, area_m2, dox_m, eps_ox) if use_oxide else c
                    k = epsilon_r_from_c(c_eff, thickness_m, area_m2)
                    self.plot.plot(f, k, pen=pen)

                    # Stats at frequency: nearest in log10 space
                    pos = (f > 0)
                    if pos.any():
                        fpos = f[pos]; kpos = k[pos]
                        idx = int(np.argmin(np.abs(np.log10(fpos) - np.log10(max(stat_value, 1e-18)))))
                        if 0 <= idx < len(kpos):
                            ks_at_target.append(float(kpos[idx]))
            else:
                if not bias_list:
                    continue
                nseg = min(len(bias_list), len(caps_list))
                for seg in range(nseg):
                    v = np.asarray(bias_list[seg]); c = np.asarray(caps_list[seg])
                    if v.size == 0 or c.size == 0:
                        continue
                    n = min(len(v), len(c))
                    v = v[:n]; c = c[:n]
                    c_eff = account_for_oxide_series(c, area_m2, dox_m, eps_ox) if use_oxide else c
                    k = epsilon_r_from_c(c_eff, thickness_m, area_m2)
                    self.plot.plot(v, k, pen=pen)

                    # Stats at voltage: nearest in linear space (allow ±)
                    idx = int(np.argpartition(np.abs(v - stat_value), 1)[1])
                    if 0 <= idx < len(k):
                        ks_at_target.append(float(k[idx]))

        ks_at_target = np.array([x for x in ks_at_target if np.isfinite(x)])
        if ks_at_target.size > 0:
            self.lbl_stats.setText(f"{ks_at_target.mean():.2f} ± {ks_at_target.std(ddof=1) if ks_at_target.size>1 else 0.0:.2f}")
        else:
            self.lbl_stats.setText("—")

        # Title: base of FIRST sample name if available
        if self.sample_names:
            base = self.sample_names[0].split('_')[0]
            self.plot.setTitle(base, size=f"{int(self.spin_fontsize.value())}pt")

        self._apply_fonts()

    def _apply_fonts(self):
        fontsize = int(self.spin_fontsize.value())
        font = QFont("Arial", fontsize)
        axL = self.plot.getAxis('left'); axB = self.plot.getAxis('bottom')
        axL.setStyle(tickFont=font); axB.setStyle(tickFont=font)
        if hasattr(axL, "setTextPen"): axL.setTextPen(pg.mkPen('k'))
        if hasattr(axB, "setTextPen"): axB.setTextPen(pg.mkPen('k'))
        if hasattr(axL, "setPen"):     axL.setPen(pg.mkPen('k'))
        if hasattr(axB, "setPen"):     axB.setPen(pg.mkPen('k'))
        axL.label.setFont(font); axB.label.setFont(font)


# ---------- Main ----------

def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, foreground='k')  # black ticks/labels
    win = KSpectralGUI(); win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
