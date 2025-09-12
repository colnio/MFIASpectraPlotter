# -*- coding: utf-8 -*-
"""
mfia_cf_gui.py — C(f) GUI for Zurich Instruments MFIA
Requirements:
    pip install PyQt5 pyqtgraph zhinst
Files:
    - mfia_helper.py (from previous step) must be importable
"""

import sys
import math
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox
)
import pyqtgraph as pg

# ---- Import the MFIA helper you already have ----
from mfia_helper import ZIMFIA, MFIAConfig

# ---------------- Data model ----------------
@dataclass
class Spectrum:
    name: str
    freq: List[float]
    C: List[float]
    bias: List[float]  # from sweeper waves; length matches freq
    drive: List[float]
    model: str
    x_log: bool

# ---------------- Worker thread for sweeps ----------------
class SweepWorker(QThread):
    finished_ok = pyqtSignal(object)   # emits Spectrum
    failed = pyqtSignal(str)

    def __init__(self,
                 mfia: ZIMFIA,
                 name: str,
                 f_start: float,
                 f_stop: float,
                 points: int,
                 x_log: bool,
                 model: str,
                 avg_samples: int,
                 avg_time_ms: int):
        super().__init__()
        self.mfia = mfia
        self.name = name
        self.f_start = f_start
        self.f_stop = f_stop
        self.points = points
        self.x_log = x_log
        self.model = model
        self.avg_samples = avg_samples
        self.avg_time_ms = avg_time_ms
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Configure model
            self.mfia.set_model("d_cs" if self.model == "d_cs" else "rc_parallel")

            sweep = self.mfia.daq.sweeper()
            sweep.setByte("sweep/device", self.mfia.dev)
            sweep.setString("sweep/gridnode", f"/{self.mfia.dev}/oscs/{self.mfia.cfg.osc_index}/freq")
            sweep.setDouble("sweep/start", self.f_start)
            sweep.setDouble("sweep/stop", self.f_stop)
            sweep.setDouble("sweep/samplecount", self.points)
            sweep.setDouble("sweep/xmapping", 1 if self.x_log else 0)
            # Bandwidth / settling / averaging (sane defaults; tweak if needed)
            sweep.setDouble("sweep/bandwidthcontrol", 2)  # auto
            sweep.setDouble("sweep/maxbandwidth", 10.0)
            sweep.setDouble("sweep/settling/inaccuracy", 1e-2)
            sweep.setDouble("sweep/averaging/sample", self.avg_samples)
            # Averaging time (optional; supported on most recent LabOne versions)
            try:
                sweep.setDouble("sweep/averaging/time", max(0.0, self.avg_time_ms / 1000.0))
            except Exception:
                # If node not available, ignore
                pass

            path = f"/{self.mfia.dev}/imps/{self.mfia.cfg.imp_index}/sample"
            sweep.subscribe(path)
            sweep.execute()
            while not sweep.finished():
                if self._stop:
                    sweep.finish()
                    raise RuntimeError("Sweep aborted by user.")
                self.mfia.daq.sleep(0.05)

            lookup = sweep.read()
            waves = lookup[path][0].sweeperImpedanceWaves[0]
            freq = list(map(float, waves.grid))
            p1 = list(map(float, waves.param1))   # C for both rc_parallel and d_cs
            bias = list(map(float, waves.bias))
            drive = list(map(float, waves.drive))

            spectrum = Spectrum(
                name=self.name,
                freq=freq,
                C=p1,
                bias=bias,
                drive=drive,
                model=self.model,
                x_log=self.x_log
            )
            self.finished_ok.emit(spectrum)

        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

# ---------------- Main Window ----------------
class CFGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MFIA — C(f) GUI")
        self.resize(1100, 650)

        # Zurich instrument
        self.mfia: Optional[ZIMFIA] = None

        # Storage
        self.spectra: Dict[str, Spectrum] = {}

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel: Controls and Live
        left = QVBoxLayout()
        layout.addLayout(left, 0)

        controls = self._make_controls()
        left.addWidget(controls)

        live = self._make_live_panel()
        left.addWidget(live)

        # Buttons row
        btn_row = QHBoxLayout()
        left.addLayout(btn_row)
        self.btn_connect = QPushButton("Connect")
        self.btn_run = QPushButton("Run Sweep")
        self.btn_stop = QPushButton("Stop")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)

        # Middle: spectra list + actions
        mid = QVBoxLayout()
        layout.addLayout(mid, 0)
        self.list = QListWidget()
        mid.addWidget(self.list)

        list_btns = QHBoxLayout()
        mid.addLayout(list_btns)
        self.btn_sel_all = QPushButton("Select All")
        self.btn_unsel_all = QPushButton("Deselect All")
        self.btn_plot = QPushButton("Plot Selected")
        list_btns.addWidget(self.btn_sel_all)
        list_btns.addWidget(self.btn_unsel_all)
        list_btns.addWidget(self.btn_plot)

        save_btns = QHBoxLayout()
        mid.addLayout(save_btns)
        self.btn_save_each = QPushButton("Save Selected (each CSV)")
        self.btn_save_combined = QPushButton("Save Selected (combined CSV)")
        save_btns.addWidget(self.btn_save_each)
        save_btns.addWidget(self.btn_save_combined)

        # Right: plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'Frequency', units='Hz')
        self.plot.setLabel('left', 'Capacitance', units='F')
        layout.addWidget(self.plot, 1)

        # Signals
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_sel_all.clicked.connect(self.on_select_all)
        self.btn_unsel_all.clicked.connect(self.on_deselect_all)
        self.btn_plot.clicked.connect(self.on_plot_selected)
        self.btn_save_each.clicked.connect(self.on_save_each)
        self.btn_save_combined.clicked.connect(self.on_save_combined)

        # Live timer
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(100)  # 10 Hz
        self.live_timer.timeout.connect(self.update_live_panel)

        self.sweep_thread: Optional[SweepWorker] = None

    # ---------- UI builders ----------
    def _make_controls(self) -> QWidget:
        box = QGroupBox("Measurement Settings")
        grid = QGridLayout(box)

        r = 0
        # Spectrum name
        grid.addWidget(QLabel("Spectrum name:"), r, 0)
        self.ed_name = QLineEdit("Spec1")
        grid.addWidget(self.ed_name, r, 1, 1, 2)
        r += 1

        # Amplitude (RMS)
        grid.addWidget(QLabel("Amplitude (Vrms):"), r, 0)
        self.sp_amp = QDoubleSpinBox()
        self.sp_amp.setDecimals(3)
        self.sp_amp.setRange(0.001, 10.0)
        self.sp_amp.setSingleStep(0.01)
        self.sp_amp.setValue(0.300)  # 300 mV default
        grid.addWidget(self.sp_amp, r, 1)
        r += 1

        # Bias
        grid.addWidget(QLabel("DC Bias (V):"), r, 0)
        self.sp_bias = QDoubleSpinBox()
        self.sp_bias.setDecimals(3)
        self.sp_bias.setRange(-50.0, 50.0)
        self.sp_bias.setSingleStep(0.01)
        self.sp_bias.setValue(0.0)
        grid.addWidget(self.sp_bias, r, 1)
        r += 1

        # Frequency range
        grid.addWidget(QLabel("f start (Hz):"), r, 0)
        self.sp_fstart = QDoubleSpinBox()
        self.sp_fstart.setDecimals(1)
        self.sp_fstart.setRange(0.1, 5e7)
        self.sp_fstart.setValue(100.0)
        grid.addWidget(self.sp_fstart, r, 1)
        r += 1

        grid.addWidget(QLabel("f stop (Hz):"), r, 0)
        self.sp_fstop = QDoubleSpinBox()
        self.sp_fstop.setDecimals(1)
        self.sp_fstop.setRange(1.0, 5e7)
        self.sp_fstop.setValue(5e6)
        grid.addWidget(self.sp_fstop, r, 1)
        r += 1

        # X distribution (log)
        self.cb_logx = QCheckBox("Logarithmic X")
        self.cb_logx.setChecked(True)
        grid.addWidget(self.cb_logx, r, 0, 1, 2)
        r += 1

        # Points
        grid.addWidget(QLabel("Points:"), r, 0)
        self.sp_points = QSpinBox()
        self.sp_points.setRange(5, 20001)
        self.sp_points.setValue(200)
        grid.addWidget(self.sp_points, r, 1)
        r += 1

        # Samples per point
        grid.addWidget(QLabel("Samples / point:"), r, 0)
        self.sp_avg_samples = QSpinBox()
        self.sp_avg_samples.setRange(1, 10000)
        self.sp_avg_samples.setValue(5)
        grid.addWidget(self.sp_avg_samples, r, 1)
        r += 1

        # Averaging time
        grid.addWidget(QLabel("Averaging time (ms):"), r, 0)
        self.sp_avg_time = QSpinBox()
        self.sp_avg_time.setRange(0, 10000)
        self.sp_avg_time.setValue(100)
        grid.addWidget(self.sp_avg_time, r, 1)
        r += 1

        # Model (fixed to D+Cs for generality; change to a dropdown if you like)
        grid.addWidget(QLabel("Model:"), r, 0)
        self.lbl_model = QLabel("D + Cs")
        self.lbl_model.setToolTip("Using MFIA built-in D, Cs model; C is Cs.")
        grid.addWidget(self.lbl_model, r, 1)
        r += 1

        return box

    def _make_live_panel(self) -> QWidget:
        box = QGroupBox("Live values")
        grid = QGridLayout(box)

        labels = ["C (F):", "R (Ω):", "RealZ (Ω):", "ImagZ (Ω):", "Phase (deg):", "f (Hz):", "Bias (V):", "Drive (Vrms):"]
        self.live_vals: Dict[str, QLabel] = {}

        for r, name in enumerate(labels):
            grid.addWidget(QLabel(name), r, 0)
            lab = QLabel("—")
            lab.setStyleSheet("color: #ddd;")
            grid.addWidget(lab, r, 1)
            key = name.split()[0].lower()  # crude key
            # map better keys:
            if "C" in name and "(F)" in name: key = "C"
            if "R" in name and "(Ω)" in name: key = "R"
            if "RealZ" in name: key = "RealZ"
            if "ImagZ" in name: key = "ImagZ"
            if "Phase" in name: key = "Phase"
            if "(Hz)" in name: key = "f"
            if "Bias" in name: key = "Bias"
            if "Drive" in name: key = "Drive"
            self.live_vals[key] = lab

        return box

    # ---------- Handlers ----------
    def on_connect(self):
        try:
            if self.mfia:
                self.mfia.close()
                self.mfia = None
            self.mfia = ZIMFIA(MFIAConfig())
            # Apply initial drive/bias/freq for live readout stability
            self.mfia.set_amplitude(self.sp_amp.value())
            self.mfia.set_bias_voltage(self.sp_bias.value())
            self.mfia.set_frequency(1e3)  # 1 kHz idle
            self.mfia.set_model("d_cs")   # default for GUI
            self.live_timer.start()
            self.btn_run.setEnabled(True)
            QMessageBox.information(self, "MFIA", f"Connected to {self.mfia.dev}")
        except Exception as e:
            QMessageBox.critical(self, "Connection failed", str(e))

    def on_run(self):
        if not self.mfia:
            QMessageBox.warning(self, "Not connected", "Please connect to MFIA first.")
            return

        name = self.ed_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a spectrum name.")
            return
        if name in self.spectra:
            ret = QMessageBox.question(self, "Overwrite?",
                                       f"Spectrum '{name}' already exists. Overwrite?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                return

        # Freeze settings into device
        try:
            self.mfia.set_amplitude(self.sp_amp.value())
            self.mfia.set_bias_voltage(self.sp_bias.value())
        except Exception as e:
            QMessageBox.critical(self, "Device error", f"Failed to set amplitude/bias:\n{e}")
            return

        # Start worker
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.sweep_thread = SweepWorker(
            mfia=self.mfia,
            name=name,
            f_start=float(self.sp_fstart.value()),
            f_stop=float(self.sp_fstop.value()),
            points=int(self.sp_points.value()),
            x_log=self.cb_logx.isChecked(),
            model="d_cs",  # fixed for now; can expose in UI later
            avg_samples=int(self.sp_avg_samples.value()),
            avg_time_ms=int(self.sp_avg_time.value()),
        )
        self.sweep_thread.finished_ok.connect(self.on_sweep_ok)
        self.sweep_thread.failed.connect(self.on_sweep_failed)
        self.sweep_thread.start()

    def on_stop(self):
        if self.sweep_thread and self.sweep_thread.isRunning():
            self.sweep_thread.stop()
        self.btn_stop.setEnabled(False)

    def on_sweep_ok(self, spec: Spectrum):
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        # Save or replace
        self.spectra[spec.name] = spec
        # Update list
        self._upsert_item(spec.name)
        # Auto-plot just the newly acquired spectrum
        self.plot_selected(clear=True, only=[spec.name])

    def on_sweep_failed(self, msg: str):
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Sweep failed", msg)

    def on_select_all(self):
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked)

    def on_deselect_all(self):
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Unchecked)

    def on_plot_selected(self):
        self.plot_selected(clear=True)

    def _upsert_item(self, name: str):
        # find
        for i in range(self.list.count()):
            if self.list.item(i).text() == name:
                it = self.list.item(i)
                it.setCheckState(Qt.Checked)
                return
        it = QListWidgetItem(name)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.Checked)
        self.list.addItem(it)

    def plot_selected(self, clear: bool = True, only: Optional[List[str]] = None):
        if clear:
            self.plot.clear()
            self.plot.setLogMode(x=self.cb_logx.isChecked(), y=False)

        # Gather selected names
        names: List[str] = []
        if only is not None:
            names = only
        else:
            for i in range(self.list.count()):
                it = self.list.item(i)
                if it.checkState() == Qt.Checked:
                    names.append(it.text())

        if not names:
            return

        # Plot
        for nm in names:
            spec = self.spectra.get(nm)
            if not spec:
                continue
            x = spec.freq
            y = spec.C
            pen = pg.mkPen(width=2)
            self.plot.plot(x, y, pen=pen, name=nm)
        # Legend (recreate to refresh)
        try:
            self.plot.addLegend()
        except Exception:
            pass  # avoid duplicate legend exceptions

    def on_save_each(self):
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "Save", "No spectra selected.")
            return
        dirpath = QFileDialog.getExistingDirectory(self, "Select folder to save CSVs")
        if not dirpath:
            return
        saved = 0
        for nm in names:
            spec = self.spectra.get(nm)
            if not spec:
                continue
            path = f"{dirpath}/{nm}.csv"
            try:
                self._write_csv(path, spec)
                saved += 1
            except Exception as e:
                QMessageBox.warning(self, "Save error", f"{nm}: {e}")
        QMessageBox.information(self, "Save", f"Saved {saved} file(s) to:\n{dirpath}")

    def on_save_combined(self):
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "Save", "No spectra selected.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save combined CSV", "combined_cf.csv", "CSV (*.csv)")
        if not path:
            return

        try:
            # Combine by writing blocks one after another with a header line between spectra
            with open(path, "w", newline="") as f:
                f.write("spectrum,frequency_Hz,C_F,bias_V,drive_Vrms,model,x_log\n")
                for nm in names:
                    spec = self.spectra.get(nm)
                    if not spec:
                        continue
                    for i in range(len(spec.freq)):
                        f.write(f"{nm},{spec.freq[i]},{spec.C[i]},{spec.bias[i]},{spec.drive[i]},{spec.model},{int(spec.x_log)}\n")
            QMessageBox.information(self, "Save", f"Combined CSV saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _write_csv(self, path: str, spec: Spectrum):
        with open(path, "w", newline="") as f:
            f.write("frequency_Hz,C_F,bias_V,drive_Vrms,model,x_log\n")
            for i in range(len(spec.freq)):
                f.write(f"{spec.freq[i]},{spec.C[i]},{spec.bias[i]},{spec.drive[i]},{spec.model},{int(spec.x_log)}\n")

    def _selected_names(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    # ---------- Live panel ----------
    def update_live_panel(self):
        if not self.mfia:
            return
        try:
            s = self.mfia.poll_impedance(duration_s=0.01, timeout_s=0.25, return_raw=False)
            # Update labels
            def fmt(val, fmtstr):
                try:
                    return fmtstr.format(val)
                except Exception:
                    return "—"

            self.live_vals["C"].setText(fmt(s["C"], "{:.3e}"))
            self.live_vals["R"].setText(fmt(s["R"], "{:.3e}"))
            self.live_vals["RealZ"].setText(fmt(s["RealZ"], "{:.3e}"))
            self.live_vals["ImagZ"].setText(fmt(s["ImagZ"], "{:.3e}"))
            self.live_vals["Phase"].setText(fmt(s["phase_deg"], "{:.2f}"))
            self.live_vals["f"].setText(fmt(s["frequency"], "{:.3f}"))
            self.live_vals["Bias"].setText(fmt(s["bias"], "{:.3f}"))
            self.live_vals["Drive"].setText(fmt(s["drive"], "{:.3f}"))
        except Exception:
            # Don’t spam errors; just gray out
            pass


def main():
    app = QApplication(sys.argv)
    # nicer default look
    pg.setConfigOptions(antialias=True)
    w = CFGui()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
