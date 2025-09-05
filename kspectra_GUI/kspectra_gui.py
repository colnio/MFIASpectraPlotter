#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K‑Spectral GUI — PyQt5 + pyqtgraph (v2)

Fixes
-----
• Avoids "TypeError: unhashable type: 'QListWidgetItem'" by mapping colors by sample *name* (string) instead of QListWidgetItem.
• Everything else same as v1: oxide toggle (1.7 nm default), mandatory film thickness, default areas (big/mid/small) editable, filters, manual hide via checklist, export visible k(F).

Run
---
pip install PyQt5 pyqtgraph numpy matplotlib pandas
python k_spectral_gui_v2.py
"""

import sys
import os
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QListWidget, QLabel, QCheckBox, QGroupBox, QListWidgetItem,
    QLineEdit, QDoubleSpinBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt

import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

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
            field = r[3].strip()
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
# GUI application
# ---------------------------

class KSpectralGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K‑Spectral Plotter")
        self.resize(1280, 820)

        self.header_file: str = ""
        self.data_file: str = ""
        self.dataset: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.sample_names: List[str] = []

        self.area_um2_defaults = {
            'big': 246447.0,
            'mid': 61428.0,
            'small': 8755.0,
        }

        self.fixed_colors = {
            'big': 'red',
            'mid': 'blue',
            'small': 'green'
        }
        self._init_ui()

    def _init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        control = QVBoxLayout()
        main.addLayout(control, 0)

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

        self.spin_thickness_nm = QDoubleSpinBox(); self.spin_thickness_nm.setDecimals(3); self.spin_thickness_nm.setRange(0.0, 1e6); self.spin_thickness_nm.setSingleStep(0.1); self.spin_thickness_nm.setValue(0.0); self.spin_thickness_nm.valueChanged.connect(self.update_plot)
        phys_form.addRow("Film thickness (nm) *", self.spin_thickness_nm)

        self.chk_oxide = QCheckBox("Account for oxide (series C)"); self.chk_oxide.setChecked(True); self.chk_oxide.stateChanged.connect(self.update_plot)
        phys_form.addRow(self.chk_oxide)

        self.spin_dox_nm = QDoubleSpinBox(); self.spin_dox_nm.setDecimals(3); self.spin_dox_nm.setRange(0.0, 1e6); self.spin_dox_nm.setSingleStep(0.1); self.spin_dox_nm.setValue(1.7); self.spin_dox_nm.valueChanged.connect(self.update_plot)
        phys_form.addRow("Oxide thickness (nm)", self.spin_dox_nm)

        self.spin_eps_ox = QDoubleSpinBox(); self.spin_eps_ox.setDecimals(3); self.spin_eps_ox.setRange(0.0, 1e6); self.spin_eps_ox.setSingleStep(0.1); self.spin_eps_ox.setValue(3.9); self.spin_eps_ox.valueChanged.connect(self.update_plot)
        phys_form.addRow("Oxide εr", self.spin_eps_ox)

        control.addWidget(phys_group)

        # Areas
        area_group = QGroupBox("Areas (token‑based: 'big'/'mid'/'small')")
        area_form = QFormLayout(area_group)
        self.spin_area_big_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_big_um2, self.area_um2_defaults['big'])
        self.spin_area_mid_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_mid_um2, self.area_um2_defaults['mid'])
        self.spin_area_small_um2 = QDoubleSpinBox(); self._setup_area_spin(self.spin_area_small_um2, self.area_um2_defaults['small'])
        area_form.addRow("BIG (µm²)", self.spin_area_big_um2)
        area_form.addRow("MID (µm²)", self.spin_area_mid_um2)
        area_form.addRow("SMALL (µm²)", self.spin_area_small_um2)
        control.addWidget(area_group)

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

        control.addWidget(filt_group)

        # Sample list
        sample_group = QGroupBox("Samples (check to show / uncheck to hide)")
        sample_v = QVBoxLayout(sample_group)
        btn_row = QHBoxLayout();
        self.btn_sel_all = QPushButton("Select All"); self.btn_sel_all.clicked.connect(self._select_all)
        self.btn_desel_all = QPushButton("Deselect All"); self.btn_desel_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(self.btn_sel_all); btn_row.addWidget(self.btn_desel_all); btn_row.addStretch(1)
        sample_v.addLayout(btn_row)
        self.list_samples = QListWidget(); self.list_samples.itemChanged.connect(self.update_plot)
        sample_v.addWidget(self.list_samples)
        control.addWidget(sample_group, stretch=1)

        # Export & stats
        exp_group = QGroupBox("Export & Stats")
        exp_form = QFormLayout(exp_group)
        self.spin_stat_freq = QDoubleSpinBox(); self.spin_stat_freq.setDecimals(0); self.spin_stat_freq.setRange(1.0, 1e12); self.spin_stat_freq.setValue(1e5); self.spin_stat_freq.valueChanged.connect(self.update_plot)
        exp_form.addRow("Stats @ freq (Hz)", self.spin_stat_freq)
        self.lbl_stats = QLabel("—"); exp_form.addRow("Mean±STD of visible k", self.lbl_stats)
        self.btn_export = QPushButton("Export visible k(F)…"); self.btn_export.clicked.connect(self.on_export)
        exp_form.addRow(self.btn_export)
        control.addWidget(exp_group)

        # Plot
        plot_v = QVBoxLayout(); main.addLayout(plot_v, 1)
        self.plot = pg.PlotWidget(); self.plot.setBackground('w'); self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', 'Frequency (Hz)'); self.plot.setLabel('left', 'k'); self.plot.setLogMode(x=True, y=False)
        plot_v.addWidget(self.plot, 1)

        axis_row = QHBoxLayout(); self.chk_logx = QCheckBox("log X"); self.chk_logx.setChecked(True); self.chk_logx.stateChanged.connect(self._toggle_logx); self.chk_logy = QCheckBox("log Y"); self.chk_logy.setChecked(False); self.chk_logy.stateChanged.connect(self._toggle_logy)
        axis_row.addWidget(self.chk_logx); axis_row.addWidget(self.chk_logy); axis_row.addStretch(1)
        plot_v.addLayout(axis_row)

        self.statusBar().showMessage("Set film thickness and load files to begin…")

    def _setup_area_spin(self, spin: QDoubleSpinBox, default_um2: float):
        spin.setDecimals(3); spin.setRange(0.0, 1e12); spin.setSingleStep(10.0); spin.setValue(default_um2); spin.valueChanged.connect(self.update_plot)

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

    # Filters
    def _apply_sample_filter(self):
        substr = self.edit_substr.text().strip().lower()
        for i in range(self.list_samples.count()):
            item = self.list_samples.item(i)
            visible = (substr in item.text().lower()) if substr else True
            item.setHidden(not visible) if False else item.setHidden(not visible)  # keep simple, avoid syntax confusion
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

    # Axes
    def _toggle_logx(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())

    def _toggle_logy(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())

    # Area helpers
    def _area_m2_for_sample(self, sample_name: str) -> Tuple[str, float]:
        s = sample_name.lower()
        if 'big' in s:
            return 'big', self.spin_area_big_um2.value() * 1e-12
        if 'small' in s:
            return 'small', self.spin_area_small_um2.value() * 1e-12
        return 'mid', self.spin_area_mid_um2.value() * 1e-12

    # Core compute & plot
    def update_plot(self):
        thickness_nm = self.spin_thickness_nm.value()
        if thickness_nm <= 0:
            self.plot.clear(); self.plot.addItem(pg.TextItem("Set film thickness (nm) to compute k", color=(150, 0, 0)))
            self.statusBar().showMessage("Film thickness required.")
            self.lbl_stats.setText("—")
            return

        self.plot.clear(); self.plot.addLegend()
        if not self.dataset:
            self.lbl_stats.setText("—"); return

        want_0v = self.chk_0v.isChecked(); want_1v = self.chk_1v.isChecked()
        want_big = self.chk_big.isChecked(); want_mid = self.chk_mid.isChecked(); want_small = self.chk_small.isChecked()

        thickness_m = thickness_nm * 1e-9
        use_oxide = self.chk_oxide.isChecked(); dox_m = self.spin_dox_nm.value() * 1e-9; eps_ox = self.spin_eps_ox.value()
        target_f = self.spin_stat_freq.value()
        ks_at_target = []

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
            caps_list = data.get('param1', [])
            if not freqs_list or not caps_list:
                continue

            # choose fixed color if area_tag is in fixed_colors
            if area_tag in self.fixed_colors:
                col = self.fixed_colors[area_tag]
            else:
                col = color_by_name.get(name, '#000000')
            pen = pg.mkPen(color=col, width=2)

            nseg = min(len(freqs_list), len(caps_list))
            for seg in range(nseg):
                f = freqs_list[seg]; c = caps_list[seg]
                if f.size == 0 or c.size == 0:
                    continue
                n = min(len(f), len(c))
                f = f[:n]; c = c[:n]
                c_eff = account_for_oxide_series(c, area_m2, dox_m, eps_ox) if use_oxide else c
                k = epsilon_r_from_c(c_eff, thickness_m, area_m2)
                self.plot.plot(f, k, pen=pen, name=name)
                idx = int(np.argmin(np.abs(np.log10(f) - np.log10(target_f))))
                if 0 <= idx < len(k): ks_at_target.append(float(k[idx]))
        ks_at_target = np.array(ks_at_target)
        ks_at_target = ks_at_target[~np.isnan(ks_at_target)]
        if len(ks_at_target) > 0:
            arr = np.array(ks_at_target)
            self.lbl_stats.setText(f"{arr.mean():.2f} ± {arr.std(ddof=1) if len(arr)>1 else 0.0:.2f}")
        else:
            self.lbl_stats.setText("—")

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

        thickness_nm = self.spin_thickness_nm.value()
        if thickness_nm <= 0:
            QMessageBox.warning(self, "Missing thickness", "Set film thickness (nm) before exporting.")
            return

        thickness_m = thickness_nm * 1e-9
        use_oxide = self.chk_oxide.isChecked(); dox_m = self.spin_dox_nm.value() * 1e-9; eps_ox = self.spin_eps_ox.value()
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
            caps_list = data.get('param1', [])
            if not freqs_list or not caps_list:
                continue

            nseg = min(len(freqs_list), len(caps_list))
            for seg in range(nseg):
                f = freqs_list[seg]
                c = caps_list[seg]
                n = min(len(f), len(c))
                f = f[:n]; c = c[:n]
                c_eff = account_for_oxide_series(c, area_m2, dox_m, eps_ox) if use_oxide else c
                k = epsilon_r_from_c(c_eff, thickness_m, area_m2)
                for fi, ki in zip(f, k):
                    rows.append({
                        'sample': name,
                        'area_tag': area_tag,
                        'area_m2': area_m2,
                        'frequency_Hz': float(fi),
                        'k': float(ki),
                        'film_thickness_nm': thickness_nm,
                        'account_for_oxide': bool(use_oxide),
                        'oxide_thickness_nm': self.spin_dox_nm.value(),
                        'oxide_eps_r': eps_ox,
                    })

        if not rows:
            QMessageBox.information(self, "Nothing to export", "No visible curves matched filters.")
            return

        df = pd.DataFrame(rows)
        df.to_csv(fn, index=False)
        self.statusBar().showMessage(f"Exported {len(df)} rows to {os.path.basename(fn)}")

    # Axis toggles
    def _toggle_logx(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())

    def _toggle_logy(self):
        self.plot.setLogMode(x=self.chk_logx.isChecked(), y=self.chk_logy.isChecked())


def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = KSpectralGUI(); win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
