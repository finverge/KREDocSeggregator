"""
Desktop GUI for KYC Document Sorter using standard library Tkinter.
Allows non-technical users to select source and destination folders,
configure options, run processing with a progress bar, and view results.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import datetime
from kyc_sorter.sorter.batch_processor import BatchProcessor, BatchSummary, ProcessedItem
from kyc_sorter.sorter.watcher import SourceFolderWatcher
from kyc_sorter.utils.config_loader import Config, load_config
from kyc_sorter.utils.logger import setup_logger


class SorterApp(tk.Tk):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.title("KYC Document Sorter & Segregator")
        self.geometry("780;640".replace(";", "x"))
        self.minsize(700, 520)

        # Load initial config
        self.config = load_config(config_path)
        self.logger = setup_logger("kyc_sorter_gui")
        self.watcher_instance: Optional[SourceFolderWatcher] = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    def _on_close(self):
        if self.watcher_instance and self.watcher_instance.is_running:
            self.watcher_instance.stop()
        self.destroy()

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self, bg="#1a365d", padx=15, pady=12)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="KYC Document Sorter",
            font=("Segoe UI", 16, "bold"),
            fg="#ffffff",
            bg="#1a365d",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(
            header_frame,
            text="Automatic KYC identification, verification, and segregation into KYC & NON KYC folders",
            font=("Segoe UI", 9),
            fg="#cbd5e0",
            bg="#1a365d",
        )
        subtitle_lbl.pack(anchor=tk.W)

        # Main content container
        content = tk.Frame(self, padx=15, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        # ── Folder Selection Group ──
        folders_group = tk.LabelFrame(content, text=" Folder Settings ", font=("Segoe UI", 10, "bold"), padx=10, pady=8)
        folders_group.pack(fill=tk.X, pady=(0, 10))

        # Source input
        tk.Label(folders_group, text="Input / Source Folder:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=4)
        self.input_var = tk.StringVar(value=str(self.config.input.resolved_path))
        self.input_entry = tk.Entry(folders_group, textvariable=self.input_var, width=52, font=("Segoe UI", 9))
        self.input_entry.grid(row=0, column=1, padx=6, pady=4, sticky=tk.EW)
        tk.Button(folders_group, text="Browse...", command=self._browse_input, width=10).grid(row=0, column=2, pady=4)

        # Destination output
        tk.Label(folders_group, text="Output / Destination Folder:", font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W, pady=4)
        self.output_var = tk.StringVar(value=str(self.config.output.resolved_path))
        self.output_entry = tk.Entry(folders_group, textvariable=self.output_var, width=52, font=("Segoe UI", 9))
        self.output_entry.grid(row=1, column=1, padx=6, pady=4, sticky=tk.EW)
        tk.Button(folders_group, text="Browse...", command=self._browse_output, width=10).grid(row=1, column=2, pady=4)

        folders_group.columnconfigure(1, weight=1)

        # ── Options Group ──
        options_group = tk.LabelFrame(content, text=" Processing Options ", font=("Segoe UI", 10, "bold"), padx=10, pady=8)
        options_group.pack(fill=tk.X, pady=(0, 10))

        # Operation mode (Copy vs Move)
        tk.Label(options_group, text="Operation Mode:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=4)
        self.op_var = tk.StringVar(value=self.config.output.operation.title())
        op_menu = ttk.Combobox(options_group, textvariable=self.op_var, values=["Copy", "Move"], state="readonly", width=10)
        op_menu.grid(row=0, column=1, sticky=tk.W, padx=6, pady=4)

        # Dry run checkbox
        self.dry_run_var = tk.BooleanVar(value=self.config.processing.dry_run)
        self.dry_run_check = tk.Checkbutton(
            options_group,
            text="Dry Run (Test Mode - do not move or copy files)",
            variable=self.dry_run_var,
            font=("Segoe UI", 9),
        )
        self.dry_run_check.grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=15)

        # Confidence threshold
        tk.Label(options_group, text="Confidence Threshold:", font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W, pady=4)
        self.conf_var = tk.DoubleVar(value=self.config.processing.confidence_threshold)
        conf_scale = tk.Scale(
            options_group,
            from_=0.50,
            to=0.99,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            length=180,
        )
        conf_scale.grid(row=1, column=1, sticky=tk.W, padx=6, pady=4)

        # Date subfolder toggle
        self.date_folder_var = tk.BooleanVar(value=self.config.output.create_date_folder)
        date_check = tk.Checkbutton(
            options_group,
            text="Create Date-Based Subfolders (e.g., Aadhaar/2026-09-30/)",
            variable=self.date_folder_var,
            font=("Segoe UI", 9),
        )
        date_check.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=15)

        # ── Action Buttons & Progress Bar ──
        act_frame = tk.Frame(content)
        act_frame.pack(fill=tk.X, pady=(0, 8))

        self.start_btn = tk.Button(
            act_frame,
            text="Run Batch Sorter",
            command=self._start_processing,
            bg="#2b6cb0",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            relief=tk.RAISED,
            cursor="hand2",
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.watch_btn = tk.Button(
            act_frame,
            text="Start Folder Watcher",
            command=self._toggle_watcher,
            bg="#276749",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            relief=tk.RAISED,
            cursor="hand2",
        )
        self.watch_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.progress_lbl = tk.Label(act_frame, text="Ready", font=("Segoe UI", 9, "italic"), fg="#4a5568")
        self.progress_lbl.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(content, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        # ── Live Activity Log ──
        log_group = tk.LabelFrame(content, text=" Activity Log ", font=("Segoe UI", 10, "bold"), padx=5, pady=5)
        log_group.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_group, wrap=tk.WORD, font=("Consolas", 8), bg="#f7fafc", fg="#2d3748")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(log_group, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scroll.set)

    def _browse_input(self):
        path = filedialog.askdirectory(title="Select Input Folder Containing KYC Documents")
        if path:
            self.input_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Destination Folder for Segregated Output")
        if path:
            self.output_var.set(path)

    def _log_message(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _start_processing(self):
        in_dir = Path(self.input_var.get().strip())
        out_dir = Path(self.output_var.get().strip())

        if not in_dir.is_dir():
            messagebox.showerror("Error", f"Input directory does not exist:\n{in_dir}")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.log_text.delete("1.0", tk.END)
        self.progress_bar["value"] = 0
        self.progress_lbl.config(text="Scanning input folder...", fg="#2b6cb0")

        # Run in separate thread to keep UI fully responsive
        thread = threading.Thread(target=self._run_batch, args=(in_dir, out_dir), daemon=True)
        thread.start()

    def _run_batch(self, in_dir: Path, out_dir: Path):
        try:
            # Prepare overrides
            overrides = {
                "input": {"directory": str(in_dir)},
                "output": {
                    "directory": str(out_dir),
                    "operation": self.op_var.get().lower(),
                    "create_date_folder": self.date_folder_var.get(),
                },
                "processing": {
                    "confidence_threshold": self.conf_var.get(),
                    "dry_run": self.dry_run_var.get(),
                },
            }
            active_config = load_config(overrides=overrides)

            processor = BatchProcessor(config=active_config)

            def on_progress(current: int, total: int, item: ProcessedItem):
                pct = int((current / total) * 100) if total > 0 else 100
                self.after(0, lambda: self._update_ui_progress(current, total, pct, item))

            summary = processor.process_batch(source_dir=in_dir, progress_callback=on_progress)

            self.after(0, lambda: self._on_batch_complete(summary))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Processing Error", f"An unexpected error occurred:\n{e}"))
            self.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    def _update_ui_progress(self, current: int, total: int, pct: int, item: ProcessedItem):
        self.progress_bar["value"] = pct
        self.progress_lbl.config(text=f"Processing file {current} of {total} ({pct}%)")
        dest_filename = Path(item.destination_path).name
        status_line = (
            f"[{item.timestamp}] {item.source_file} -> {item.document_type} "
            f"(conf={item.confidence:.0%}) -> Renamed: {dest_filename} [{item.status}]"
        )
        self._log_message(status_line)

    def _toggle_watcher(self):
        if self.watcher_instance and self.watcher_instance.is_running:
            # Stop watcher
            self.watch_btn.config(state=tk.DISABLED, text="Stopping...")
            threading.Thread(target=self._stop_watcher_thread, daemon=True).start()
        else:
            # Start watcher
            in_dir = Path(self.input_var.get().strip())
            out_dir = Path(self.output_var.get().strip())
            if not in_dir.is_dir():
                messagebox.showerror("Error", f"Input directory does not exist:\n{in_dir}")
                return

            overrides = {
                "input": {"directory": str(in_dir)},
                "output": {
                    "directory": str(out_dir),
                    "operation": self.op_var.get().lower(),
                    "create_date_folder": self.date_folder_var.get(),
                },
                "processing": {
                    "confidence_threshold": self.conf_var.get(),
                    "dry_run": self.dry_run_var.get(),
                },
            }
            active_config = load_config(overrides=overrides)

            def on_watched_file(item: ProcessedItem):
                dest_name = Path(item.destination_path).name
                status_line = (
                    f"[{item.timestamp}] [WATCHER] {item.source_file} -> {item.document_type} "
                    f"({item.confidence:.0%}) -> Renamed: {dest_name} [{item.status}]"
                )
                self.after(0, lambda: self._log_message(status_line))

            self.watcher_instance = SourceFolderWatcher(
                config=active_config,
                on_event_callback=on_watched_file,
            )
            try:
                self.watcher_instance.start(process_existing=False)
                self.watch_btn.config(text="Stop Folder Watcher", bg="#c53030")
                self.progress_lbl.config(
                    text=f"Watcher Active — Monitoring {in_dir.name}/ for new files...",
                    fg="#276749",
                )
                ts_now = datetime.datetime.now().strftime("%H:%M:%S")
                self._log_message(f"[{ts_now}] [WATCHER ACTIVE] Monitoring '{in_dir}' for new KYC documents.")
                self._log_message(f"[{ts_now}] Drop supported images (.jpg, .JPG) into folder to trigger automatic sorting.")
            except Exception as e:
                self.watcher_instance = None
                messagebox.showerror("Watcher Error", f"Failed to start filesystem watcher:\n{e}")

    def _stop_watcher_thread(self):
        if self.watcher_instance:
            self.watcher_instance.stop()
            self.watcher_instance = None
        self.after(0, self._on_watcher_stopped)

    def _on_watcher_stopped(self):
        self.watch_btn.config(state=tk.NORMAL, text="Start Folder Watcher", bg="#276749")
        self.progress_lbl.config(text="Watcher stopped. Ready.", fg="#4a5568")
        self._log_message("Source folder event listener stopped.")

    def _on_batch_complete(self, summary: BatchSummary):
        self.start_btn.config(state=tk.NORMAL)
        self.progress_lbl.config(text=f"Batch complete: {summary.processed} files processed in {summary.elapsed_time_sec:.1f}s", fg="#276749")
        self._log_message("\n" + "=" * 50)
        self._log_message("BATCH PROCESSING SUMMARY")
        self._log_message("=" * 50)
        self._log_message(f"Total Files       : {summary.total_files}")
        self._log_message(f"Processed         : {summary.processed}")
        for k, v in sorted(summary.counts_by_type.items()):
            self._log_message(f"{k:<18}: {v}")
        if summary.csv_report_path:
            self._log_message(f"CSV Report Saved  : {summary.csv_report_path}")
        self._log_message("=" * 50)

        messagebox.showinfo(
            "Batch Complete",
            f"Successfully processed {summary.processed} of {summary.total_files} files.\n\n"
            f"CSV Report: {summary.csv_report_path}",
        )


def launch_gui(config_path: Optional[Path] = None):
    """Entry point to launch the graphical desktop interface."""
    app = SorterApp(config_path=config_path)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
