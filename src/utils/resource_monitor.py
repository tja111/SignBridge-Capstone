"""Low-overhead CPU/GPU monitoring for reproducible training reports."""

import csv
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResourceMonitor:
    """Sample resources and save per-run plus summary CSV files."""

    def __init__(self, mode: str, interval_seconds: float = 1.0):
        self.mode = mode
        self.interval_seconds = interval_seconds
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._stopped = False
        self._process = psutil.Process()
        self._start_time = None

    @staticmethod
    def _gpu_stats():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
            return float(values[0]), float(values[1]), float(values[2])
        except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
            return None, None, None

    def _sample(self):
        gpu_util, gpu_used, gpu_total = self._gpu_stats()
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "process_cpu_percent": self._process.cpu_percent(interval=None),
            "system_memory_percent": psutil.virtual_memory().percent,
            "gpu_utilization_percent": gpu_util,
            "gpu_memory_used_mib": gpu_used,
            "gpu_memory_total_mib": gpu_total,
        }

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            self._samples.append(self._sample())

    def start(self):
        self._start_time = time.monotonic()
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._samples.append(self._sample())
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    @staticmethod
    def _mean(values):
        values = [value for value in values if value is not None]
        return round(sum(values) / len(values), 2) if values else ""

    @staticmethod
    def _peak(values):
        values = [value for value in values if value is not None]
        return round(max(values), 2) if values else ""

    def _write_panel_report(self, output_dir, summary):
        """Create a figure and a concise explanation suitable for a capstone."""
        elapsed_minutes = [index * self.interval_seconds / 60 for index in range(len(self._samples))]
        chart_path = output_dir / f"{self.mode}_{self.run_id}_utilization.png"
        fig, (util_ax, memory_ax) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        util_ax.plot(elapsed_minutes, [s["system_cpu_percent"] for s in self._samples],
                     label="System CPU", color="#2563eb", linewidth=1.8)
        gpu_values = [s["gpu_utilization_percent"] for s in self._samples]
        if any(value is not None for value in gpu_values):
            util_ax.plot(elapsed_minutes, [value if value is not None else float("nan") for value in gpu_values],
                         label="GPU", color="#dc2626", linewidth=1.8)
        util_ax.set_ylabel("Utilization (%)")
        util_ax.set_ylim(0, 100)
        util_ax.set_title(f"{self.mode.title()} Training: CPU and GPU Utilization")
        util_ax.grid(alpha=0.25)
        util_ax.legend(loc="upper right")

        memory_ax.plot(elapsed_minutes, [s["gpu_memory_used_mib"] if s["gpu_memory_used_mib"] is not None else float("nan")
                                          for s in self._samples], color="#7c3aed", linewidth=1.8)
        memory_ax.set_ylabel("GPU memory (MiB)")
        memory_ax.set_xlabel("Elapsed time (minutes)")
        memory_ax.set_title("GPU Memory in Use")
        memory_ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        report_path = output_dir / f"{self.mode}_{self.run_id}_report.md"
        duration_seconds = time.monotonic() - self._start_time if self._start_time else 0
        with report_path.open("w", encoding="utf-8") as report:
            report.write(f"# {self.mode.title()} Training Summary\n\n")
            report.write(f"- **Total training time:** {duration_seconds / 60:.2f} minutes\n")
            report.write(f"- **Average CPU utilization:** {summary['average_system_cpu_percent']}%\n")
            report.write(f"- **Average GPU utilization:** {summary['average_gpu_utilization_percent']}%\n")

    def stop(self):
        if self._stopped:
            return None
        self._stopped = True
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 1)
        self._samples.append(self._sample())

        output_dir = Path("results") / "resource_usage"
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_path = output_dir / f"{self.mode}_{self.run_id}.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self._samples[0]))
            writer.writeheader()
            writer.writerows(self._samples)

        summary = {
            "run_id": self.run_id, "mode": self.mode, "sample_count": len(self._samples),
            "average_system_cpu_percent": self._mean([s["system_cpu_percent"] for s in self._samples]),
            "peak_system_cpu_percent": self._peak([s["system_cpu_percent"] for s in self._samples]),
            "average_process_cpu_percent": self._mean([s["process_cpu_percent"] for s in self._samples]),
            "peak_process_cpu_percent": self._peak([s["process_cpu_percent"] for s in self._samples]),
            "average_system_memory_percent": self._mean([s["system_memory_percent"] for s in self._samples]),
            "peak_system_memory_percent": self._peak([s["system_memory_percent"] for s in self._samples]),
            "average_gpu_utilization_percent": self._mean([s["gpu_utilization_percent"] for s in self._samples]),
            "peak_gpu_utilization_percent": self._peak([s["gpu_utilization_percent"] for s in self._samples]),
            "peak_gpu_memory_used_mib": self._peak([s["gpu_memory_used_mib"] for s in self._samples]),
            "gpu_memory_total_mib": self._peak([s["gpu_memory_total_mib"] for s in self._samples]),
            "samples_csv": str(samples_path),
        }
        summary_path = Path("results") / "resource_usage_summary.csv"
        write_header = not summary_path.exists() or summary_path.stat().st_size == 0
        with summary_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
        self._write_panel_report(output_dir, summary)
        return summary_path
