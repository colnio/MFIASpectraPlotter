#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mfia_helper.py — Lightweight helper for Zurich Instruments MFIA

Features
--------
• Auto-discovery and connect to MFIA through LabOne Data Server
• Set amplitude (RMS), DC bias, and frequency
• Select built-in impedance model: parallel R||C (model 0) or D+Cs (model 4)
• Sweeps:
    - C(f): frequency sweep with Sweeper module
    - C(V): DC-bias sweep at fixed frequency with Sweeper module
• Single-shot poll of impedance sample → C, R, RealZ, ImagZ, |Z|, phase(rad/deg)
• Continuous streaming:
    - async generator: stream_impedance(period_s)
    - sync wrapper:   stream_impedance_sync(period_s)

Notes
-----
• For model 0 (R||C) we map param0→R (Rp), param1→C (Cp).
• For model 4 (D+Cs) we map param0→D, param1→Cs, and compute Rs = D / (2π f Cs).
• Phase is atan2(ImagZ, RealZ). Magnitude is hypot(RealZ, ImagZ).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Any, AsyncGenerator, Generator
import math
import time
import asyncio

# --- LabOne Python API imports (newer core first, then legacy fallback) ---
try:
    from zhinst.core import ziDAQServer, ziDiscovery
except ImportError:  # pragma: no cover
    from zhinst.ziPython import ziDAQServer, ziDiscovery  # type: ignore

# -------- Types & Model map --------
ModelName = Literal["rc_parallel", "d_cs"]

MODEL_MAP: Dict[ModelName, int] = {
    "rc_parallel": 0,  # R||C (param0=R, param1=Cp)
    "d_cs": 4,         # D+Cs (param0=D, param1=Cs)
}

@dataclass
class MFIAConfig:
    host: str = "localhost"
    port: int = 8004
    apilevel: int = 6
    imp_index: int = 0  # /imps/0
    osc_index: int = 0  # /oscs/0
    demod_order: int = 8  # a reasonable default


class ZIMFIA:
    """
    Convenience wrapper around MFIA impedance nodes and the Sweeper module.
    """

    # ---------------- lifecycle ----------------
    def __init__(self, cfg: MFIAConfig = MFIAConfig(), device_id: Optional[str] = None):
        self.cfg = cfg
        self.daq = ziDAQServer(cfg.host, cfg.port, cfg.apilevel)
        self.daq.connect()
        disc = ziDiscovery()

        # discover (or use provided) device ID, then attach it to this Data Server
        devs = disc.findAll()
        if not devs:
            raise RuntimeError("No Zurich Instruments devices discovered via LabOne.")
        self.dev = device_id or devs[0]
        disc.connectDevice(self.dev, self.cfg.host)

        # enable impedance app and set sensible demod defaults
        self._set_int(f"/{self.dev}/imps/{cfg.imp_index}/enable", 1)
        self._set_int(f"/{self.dev}/imps/{cfg.imp_index}/demod/order", cfg.demod_order)
        self._set_int(f"/{self.dev}/imps/{cfg.imp_index}/demod/oscselect", cfg.osc_index)
        self.daq.sync()

    def close(self):
        try:
            self._set_int(f"/{self.dev}/imps/{self.cfg.imp_index}/enable", 0)
        finally:
            self.daq.disconnect()

    # context manager sugar
    def __enter__(self) -> "ZIMFIA":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------- low-level helpers ----------------
    def _set_int(self, path: str, val: int) -> None:
        self.daq.setInt(path, int(val))

    def _get_int(self, path: str) -> int:
        return int(self.daq.getInt(path))

    def _set_double(self, path: str, val: float) -> None:
        self.daq.setDouble(path, float(val))

    def _get_double(self, path: str) -> float:
        return float(self.daq.getDouble(path))

    def _set_string(self, path: str, val: str) -> None:
        # In LabOne Python API, setByte is used to set strings
        self.daq.setByte(path, val)

    # ---------------- top-level configuration ----------------
    def set_model(self, model: ModelName) -> None:
        """Select the built-in model (0: R||C, 4: D+Cs)."""
        if model not in MODEL_MAP:
            raise ValueError(f"Unknown model '{model}'. Use one of {list(MODEL_MAP)}.")
        self._set_int(f"/{self.dev}/imps/{self.cfg.imp_index}/model", MODEL_MAP[model])
        self.daq.sync()

    # (1) amplitude
    def set_amplitude(self, ac_volts_rms: float) -> None:
        """Set AC drive amplitude (RMS volts)."""
        self._set_double(f"/{self.dev}/imps/{self.cfg.imp_index}/drive", ac_volts_rms)

    # (2) bias
    def set_bias_voltage(self, v_dc: float) -> None:
        """Set DC bias (volts)."""
        self._set_double(f"/{self.dev}/imps/{self.cfg.imp_index}/bias/value", v_dc)

    # (3) frequency
    def set_frequency(self, freq_hz: float) -> None:
        """Set fixed measurement frequency (Hz)."""
        self._set_double(f"/{self.dev}/imps/{self.cfg.imp_index}/freq", freq_hz)

    # Backward-compat convenience names
    set_drive = set_amplitude
    set_bias = set_bias_voltage
    set_fixed_frequency = set_frequency

    # ---------------- single-shot poll ----------------
    def poll_impedance(
        self,
        duration_s: float = 0.02,
        timeout_s: float = 0.5,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Grab the latest impedance sample and compute:
        C, R, RealZ, ImagZ, |Z|, phase_rad, phase_deg, plus frequency, drive, bias, model_id.

        Mapping by model:
            model 0 (R||C): C = param1 (Cp), R = param0 (Rp)
            model 4 (D+Cs): C = param1 (Cs), R = D / (2π f Cs) with D=param0

        Parameters
        ----------
        duration_s : float
            How long the DAQ should collect before returning a sample.
        timeout_s : float
            Poll timeout.
        return_raw : bool
            If True, include the raw dict returned by daq.poll() for debugging.

        Returns
        -------
        dict
        """
        path = f"/{self.dev}/imps/{self.cfg.imp_index}/sample"
        self.daq.unsubscribe("*")
        self.daq.subscribe(path)
        data = self.daq.poll(duration_s, timeout_s, 0, True)

        if path not in data or len(data[path]) == 0 or "impedance" not in data[path][0]:
            self.daq.unsubscribe(path)
            raise RuntimeError("No impedance sample received; verify /imps/0 is enabled and configured.")

        sample = data[path][0]["impedance"][-1]

        realz = float(sample["realz"])
        imagz = float(sample["imagz"])
        freq  = float(sample["frequency"])
        p0    = float(sample["param0"])
        p1    = float(sample["param1"])
        drive = float(sample["drive"])
        bias  = float(sample["bias"])

        model_id = self._get_int(f"/{self.dev}/imps/{self.cfg.imp_index}/model")

        # Convert param0/param1 to C, R
        if model_id == 0:
            C = p1
            R = p0
        elif model_id == 4:
            Cs = p1
            D  = p0
            C  = Cs
            R  = (D / (2.0 * math.pi * freq * Cs)) if (Cs > 0.0 and freq > 0.0) else float("nan")
        else:
            C = float("nan")
            R = float("nan")

        mag = math.hypot(realz, imagz)
        phase_rad = math.atan2(imagz, realz)
        phase_deg = math.degrees(phase_rad)

        out = {
            "C": C,
            "R": R,
            "RealZ": realz,
            "ImagZ": imagz,
            "Zmag": mag,
            "phase_rad": phase_rad,
            "phase_deg": phase_deg,
            "frequency": freq,
            "drive": drive,
            "bias": bias,
            "model_id": model_id,
            "param0": p0,
            "param1": p1,
        }
        if return_raw:
            out["raw"] = data[path][0]
        self.daq.unsubscribe(path)
        return out

    # ---------------- continuous streaming ----------------
    async def stream_impedance(
        self,
        period_s: float = 0.05,
        duration_per_poll_s: float = 0.01,
        timeout_s: float = 0.25,
        *,
        include_raw: bool = False,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronous generator that yields fresh impedance dicts at ~period_s cadence.

        Parameters
        ----------
        period_s : float
            Target cadence between yielded samples (sleep between polls).
        duration_per_poll_s : float
            DAQ poll acquisition time per sample (lower → lower SNR, higher update rate).
        timeout_s : float
            Poll timeout.
        include_raw : bool
            Include raw DAQ data in each yielded dict.
        stop_event : asyncio.Event
            If provided, streaming stops when this event is set.

        Usage
        -----
        async for s in mfia.stream_impedance(0.05):
            # update GUI with s["C"], s["R"], s["Zmag"], s["phase_deg"], ...
        """
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    sample = self.poll_impedance(duration_s=duration_per_poll_s,
                                                 timeout_s=timeout_s,
                                                 return_raw=include_raw)
                    yield sample
                except Exception as e:  # noisy environments may occasionally miss a poll
                    yield {"error": str(e)}

                await asyncio.sleep(max(0.0, period_s))
        finally:
            # ensure we leave with no lingering subscriptions
            self.daq.unsubscribe("*")

    def stream_impedance_sync(
        self,
        period_s: float = 0.05,
        duration_per_poll_s: float = 0.01,
        timeout_s: float = 0.25,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous generator wrapper around single-shot polling.
        Useful for non-async GUI loops (e.g., PyQt timers).
        """
        try:
            while True:
                try:
                    yield self.poll_impedance(duration_s=duration_per_poll_s,
                                              timeout_s=timeout_s,
                                              return_raw=False)
                except Exception as e:
                    yield {"error": str(e)}
                time.sleep(max(0.0, period_s))
        finally:
            self.daq.unsubscribe("*")

    # ---------------- sweeps ----------------
    def sweep_c_vs_f(
        self,
        f_start: float,
        f_stop: float,
        points: int = 50,
        *,
        x_log: bool = True,
        model: ModelName = "rc_parallel",
        averaging_samples: int = 200,
        settling_inaccuracy: float = 1e-2,
        max_bw_hz: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Run a C(f) frequency sweep using the Sweeper.

        Returns dict with arrays: freq, C, param0, param1, bias, drive, model
        """
        self.set_model(model)
        sweep = self.daq.sweeper()
        sweep.setByte("sweep/device", self.dev)
        sweep.setString("sweep/gridnode", f"/{self.dev}/oscs/{self.cfg.osc_index}/freq")
        sweep.setDouble("sweep/start", f_start)
        sweep.setDouble("sweep/stop", f_stop)
        sweep.setDouble("sweep/samplecount", points)
        sweep.setDouble("sweep/xmapping", 1 if x_log else 0)

        # bandwidth / settling / averaging
        sweep.setDouble("sweep/bandwidthcontrol", 2)  # auto
        sweep.setDouble("sweep/maxbandwidth", max_bw_hz)
        sweep.setDouble("sweep/settling/inaccuracy", settling_inaccuracy)
        sweep.setDouble("sweep/averaging/sample", averaging_samples)

        path = f"/{self.dev}/imps/{self.cfg.imp_index}/sample"
        sweep.subscribe(path)
        sweep.execute()
        while not sweep.finished():
            time.sleep(0.05)

        lookup = sweep.read()
        try:
            waves = lookup[path][0].sweeperImpedanceWaves[0]
        except Exception as e:
            raise RuntimeError(f"Unexpected sweeper data format: {e}")

        freq = waves.grid
        p0 = waves.param0
        p1 = waves.param1
        bias = waves.bias
        drive = waves.drive

        # param1 already equals C for both models used here
        C = p1

        return {
            "freq": freq,
            "C": C,
            "param0": p0,
            "param1": p1,
            "bias": bias,
            "drive": drive,
            "model": model,
        }

    def sweep_c_vs_v(
        self,
        v_start: float,
        v_stop: float,
        points: int,
        *,
        f_fixed_hz: float,
        model: ModelName = "d_cs",
        averaging_samples: int = 200,
        settling_inaccuracy: float = 1e-2,
        max_bw_hz: float = 10.0,
        derive_rs_for_series: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a C(V) DC-bias sweep at fixed frequency using the Sweeper.

        For model 'rc_parallel': param1 = Cp → C
        For model 'd_cs':        param1 = Cs → C, param0 = D; optionally compute Rs.
        """
        self.set_model(model)
        self.set_frequency(f_fixed_hz)

        sweep = self.daq.sweeper()
        sweep.setByte("sweep/device", self.dev)
        sweep.setString("sweep/gridnode", f"/{self.dev}/imps/{self.cfg.imp_index}/bias/value")
        sweep.setDouble("sweep/start", v_start)
        sweep.setDouble("sweep/stop", v_stop)
        sweep.setDouble("sweep/samplecount", points)
        sweep.setDouble("sweep/xmapping", 0)  # linear bias

        sweep.setDouble("sweep/bandwidthcontrol", 2)
        sweep.setDouble("sweep/maxbandwidth", max_bw_hz)
        sweep.setDouble("sweep/settling/inaccuracy", settling_inaccuracy)
        sweep.setDouble("sweep/averaging/sample", averaging_samples)

        path = f"/{self.dev}/imps/{self.cfg.imp_index}/sample"
        sweep.subscribe(path)
        sweep.execute()
        while not sweep.finished():
            time.sleep(0.05)

        lookup = sweep.read()
        try:
            waves = lookup[path][0].sweeperImpedanceWaves[0]
        except Exception as e:
            raise RuntimeError(f"Unexpected sweeper data format: {e}")

        vbias = waves.grid
        p0 = waves.param0
        p1 = waves.param1
        freq = waves.frequency
        drive = waves.drive

        if model == "rc_parallel":
            C = p1
            out: Dict[str, Any] = {
                "bias": vbias, "C": C, "param0": p0, "param1": p1,
                "frequency": freq, "drive": drive, "model": model
            }
        else:
            # D+Cs
            D = p0
            Cs = p1
            C = Cs
            out = {"bias": vbias, "C": C, "D": D, "Cs": Cs,
                   "frequency": freq, "drive": drive, "model": model}
            if derive_rs_for_series:
                Rs = []
                for d, f_, c in zip(D, freq, Cs):
                    fval = float(getattr(f_, "avg", f_))
                    cval = float(c)
                    if fval > 0.0 and cval > 0.0:
                        Rs.append(float(d) / (2.0 * math.pi * fval * cval))
                    else:
                        Rs.append(float("nan"))
                out["Rs"] = Rs

        return out


# ---------------- Example CLI usage ----------------
if __name__ == "__main__":
    # Minimal smoke test usage; adjust values to your setup.
    # (Assumes LabOne Data Server is running and MFIA is connected.)
    import json

    with ZIMFIA() as mfia:
        mfia.set_model("d_cs")
        mfia.set_amplitude(0.1)      # 100 mVrms
        mfia.set_bias_voltage(0.0)   # 0 V DC
        mfia.set_frequency(1e3)      # 1 kHz

        print("Single poll:")
        s = mfia.poll_impedance()
        print(json.dumps({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in s.items()}, indent=2))

        print("\nStreaming 5 samples (sync wrapper):")
        g = mfia.stream_impedance_sync(period_s=0.05, duration_per_poll_s=0.01)
        for i, sample in enumerate(g):
            print(f"{i+1}: phase_deg={sample.get('phase_deg'):.3f}, C={sample.get('C'):.3e}, R={sample.get('R'):.3e}")
            if i >= 4:
                break
