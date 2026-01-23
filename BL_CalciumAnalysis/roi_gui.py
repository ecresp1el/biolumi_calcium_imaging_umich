"""GUI for tracking ROI curation status and launching Napari."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from BL_CalciumAnalysis.roi_processing import process_roi_analysis


@dataclass(frozen=True)
class RecordingEntry:
    name: str
    manifest_path: Path
    motion_corrected_tiff: Path | None
    max_projection_tiff: Path | None
    roi_path: Path

    @property
    def roi_exists(self) -> bool:
        return self.roi_path.exists()

    @property
    def motion_corrected_exists(self) -> bool:
        return self.motion_corrected_tiff is not None and self.motion_corrected_tiff.exists()

    @property
    def max_projection_exists(self) -> bool:
        return self.max_projection_tiff is not None and self.max_projection_tiff.exists()

    @property
    def ready_for_roi(self) -> bool:
        return self.motion_corrected_exists and self.max_projection_exists


def _load_manifest(manifest_path: Path) -> RecordingEntry | None:
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None

    ims_path_raw = payload.get("ims_path")
    ims_path = Path(ims_path_raw) if ims_path_raw else manifest_path.parent
    recording_name = ims_path.stem if ims_path.suffix else ims_path.name
    if not recording_name:
        recording_name = manifest_path.parent.name
    paths = payload.get("paths", {})

    motion_corrected_raw = paths.get("motion_corrected_tiff")
    max_projection_raw = paths.get("max_projection")

    motion_corrected = Path(motion_corrected_raw) if motion_corrected_raw else None
    max_projection = Path(max_projection_raw) if max_projection_raw else None

    roi_path = manifest_path.parent / "rois" / f"{recording_name}_roi_masks_uint16.tif"

    return RecordingEntry(
        name=recording_name,
        manifest_path=manifest_path,
        motion_corrected_tiff=motion_corrected,
        max_projection_tiff=max_projection,
        roi_path=roi_path,
    )


def load_recordings(project_root: Path) -> list[RecordingEntry]:
    manifests = sorted(project_root.rglob("processing_manifest.json"))
    print(f"[roi_gui] Scanning {project_root} for manifests.")
    print(f"[roi_gui] Found {len(manifests)} manifest(s).")
    recordings: list[RecordingEntry] = []
    for manifest in manifests:
        entry = _load_manifest(manifest)
        if entry is None:
            print(f"[roi_gui] Skipping invalid manifest: {manifest}")
            continue
        recordings.append(entry)
        print(
            "[roi_gui] Loaded recording: "
            f"name={entry.name} "
            f"ready={entry.ready_for_roi} "
            f"roi_exists={entry.roi_exists}"
        )
    return recordings


class RoiTrackerApp(ttk.Frame):
    def __init__(self, master: tk.Tk, project_root: Path) -> None:
        super().__init__(master)
        self.project_root = project_root
        self.pending_entries: list[RecordingEntry] = []
        self.completed_entries: list[RecordingEntry] = []
        self.missing_entries: list[RecordingEntry] = []
        self.pending_listbox: tk.Listbox
        self.completed_listbox: tk.Listbox
        self.missing_listbox: tk.Listbox
        self.detail_text: tk.Text
        self.summary_label = ttk.Label(self, text="")
        self._selected_entry: RecordingEntry | None = None
        self._build_layout()
        self.refresh()

    def _build_layout(self) -> None:
        self.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.master.title("ROI Curation Tracker")
        self.master.minsize(720, 480)

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.summary_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        pending_frame = ttk.LabelFrame(self, text="Needs ROI")
        completed_frame = ttk.LabelFrame(self, text="ROI Complete")
        missing_frame = ttk.LabelFrame(self, text="Missing Inputs")
        pending_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        completed_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        missing_frame.grid(row=1, column=2, sticky="nsew")
        pending_frame.columnconfigure(0, weight=1)
        completed_frame.columnconfigure(0, weight=1)
        missing_frame.columnconfigure(0, weight=1)

        self.pending_listbox = tk.Listbox(pending_frame, height=12, exportselection=False)
        self.completed_listbox = tk.Listbox(
            completed_frame,
            height=12,
            exportselection=False,
            selectmode=tk.EXTENDED,
        )
        self.missing_listbox = tk.Listbox(missing_frame, height=12, exportselection=False)

        pending_scroll = ttk.Scrollbar(pending_frame, orient="vertical", command=self.pending_listbox.yview)
        completed_scroll = ttk.Scrollbar(
            completed_frame, orient="vertical", command=self.completed_listbox.yview
        )
        missing_scroll = ttk.Scrollbar(missing_frame, orient="vertical", command=self.missing_listbox.yview)
        self.pending_listbox.configure(yscrollcommand=pending_scroll.set)
        self.completed_listbox.configure(yscrollcommand=completed_scroll.set)
        self.missing_listbox.configure(yscrollcommand=missing_scroll.set)

        self.pending_listbox.grid(row=0, column=0, sticky="nsew")
        pending_scroll.grid(row=0, column=1, sticky="ns")
        self.completed_listbox.grid(row=0, column=0, sticky="nsew")
        completed_scroll.grid(row=0, column=1, sticky="ns")
        self.missing_listbox.grid(row=0, column=0, sticky="nsew")
        missing_scroll.grid(row=0, column=1, sticky="ns")

        pending_frame.rowconfigure(0, weight=1)
        completed_frame.rowconfigure(0, weight=1)
        missing_frame.rowconfigure(0, weight=1)

        self.pending_listbox.bind("<<ListboxSelect>>", lambda _: self._on_select("pending"))
        self.completed_listbox.bind("<<ListboxSelect>>", lambda _: self._on_select("completed"))
        self.missing_listbox.bind("<<ListboxSelect>>", lambda _: self._on_select("missing"))

        detail_frame = ttk.LabelFrame(self, text="Recording Details")
        detail_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.detail_text = tk.Text(detail_frame, height=6, width=70, state="disabled")
        self.detail_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        button_frame.columnconfigure(0, weight=1)

        ttk.Button(button_frame, text="Refresh", command=self.refresh).grid(row=0, column=0, sticky="w")
        ttk.Button(button_frame, text="Open ROI Editor", command=self.open_roi_editor).grid(
            row=0, column=1, padx=8
        )
        ttk.Button(button_frame, text="Open Recording Folder", command=self.open_recording_folder).grid(
            row=0, column=2
        )
        ttk.Button(button_frame, text="Process ROI Traces", command=self.process_roi_traces).grid(
            row=0, column=3, padx=8
        )

    def refresh(self) -> None:
        entries = load_recordings(self.project_root)
        self.pending_entries = [entry for entry in entries if entry.ready_for_roi and not entry.roi_exists]
        self.completed_entries = [entry for entry in entries if entry.roi_exists]
        self.missing_entries = [entry for entry in entries if not entry.ready_for_roi]

        print(
            "[roi_gui] Refresh summary: "
            f"total={len(entries)} "
            f"pending={len(self.pending_entries)} "
            f"completed={len(self.completed_entries)} "
            f"missing_inputs={len(self.missing_entries)}"
        )

        self.pending_listbox.delete(0, tk.END)
        self.completed_listbox.delete(0, tk.END)
        self.missing_listbox.delete(0, tk.END)

        for entry in self.pending_entries:
            self.pending_listbox.insert(tk.END, entry.name)
        for entry in self.completed_entries:
            self.completed_listbox.insert(tk.END, entry.name)
        for entry in self.missing_entries:
            self.missing_listbox.insert(tk.END, entry.name)

        summary = (
            f"Found {len(entries)} recording(s) • "
            f"Needs ROI: {len(self.pending_entries)} • "
            f"ROI Complete: {len(self.completed_entries)} • "
            f"Missing Inputs: {len(self.missing_entries)}"
        )
        self.summary_label.configure(text=summary)

        if not entries:
            self._set_detail_text(
                "No recordings found. Make sure you selected the project root that contains "
                "processing_manifest.json files."
            )
        else:
            self._set_detail_text("Select a recording to view details.")
        self._selected_entry = None

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text)
        self.detail_text.configure(state="disabled")

    def _on_select(self, list_name: str) -> None:
        if list_name == "pending":
            selection = self.pending_listbox.curselection()
            entries = self.pending_entries
            self.completed_listbox.selection_clear(0, tk.END)
            self.missing_listbox.selection_clear(0, tk.END)
        elif list_name == "completed":
            selection = self.completed_listbox.curselection()
            entries = self.completed_entries
            self.pending_listbox.selection_clear(0, tk.END)
            self.missing_listbox.selection_clear(0, tk.END)
        else:
            selection = self.missing_listbox.curselection()
            entries = self.missing_entries
            self.pending_listbox.selection_clear(0, tk.END)
            self.completed_listbox.selection_clear(0, tk.END)

        if not selection:
            return

        index = selection[0]
        entry = entries[index]
        self._selected_entry = entry
        motion_corrected = entry.motion_corrected_tiff or Path("<missing>")
        max_projection = entry.max_projection_tiff or Path("<missing>")
        status = "✅ complete" if entry.roi_exists else "⚠️ missing"
        inputs_status = "ready" if entry.ready_for_roi else "missing inputs"
        detail = (
            f"Recording: {entry.name}\n"
            f"Manifest: {entry.manifest_path}\n"
            f"Motion-corrected: {motion_corrected}\n"
            f"Max projection: {max_projection}\n"
            f"ROI file: {entry.roi_path}\n"
            f"ROI status: {status}\n"
            f"Inputs status: {inputs_status}"
        )
        self._set_detail_text(detail)

    def open_roi_editor(self) -> None:
        entry = self._selected_entry
        if entry is None:
            messagebox.showinfo("ROI Editor", "Select a recording first.")
            return
        if not entry.ready_for_roi:
            messagebox.showwarning(
                "ROI Editor",
                "This recording is missing the motion-corrected movie or max projection.",
            )
            return

        entry.roi_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "BL_CalciumAnalysis.napari_roi_cli",
            "--movie",
            str(entry.motion_corrected_tiff),
            "--max-projection",
            str(entry.max_projection_tiff),
            "--roi",
            str(entry.roi_path),
            "--save-roi",
            str(entry.roi_path),
            "--strict",
        ]
        subprocess.Popen(command)

    def open_recording_folder(self) -> None:
        entry = self._selected_entry
        if entry is None:
            messagebox.showinfo("Open Folder", "Select a recording first.")
            return

        folder = entry.manifest_path.parent
        try:
            subprocess.Popen(["open", str(folder)])
        except FileNotFoundError:
            messagebox.showerror("Open Folder", "Could not open folder on this platform.")

    def _get_selected_completed_entries(self) -> list[RecordingEntry]:
        selections = self.completed_listbox.curselection()
        if not selections:
            return []
        return [self.completed_entries[idx] for idx in selections]

    def process_roi_traces(self) -> None:
        entries = self._get_selected_completed_entries()
        print("[roi_gui] Process ROI Traces clicked.")
        if not entries:
            if not self.completed_entries:
                messagebox.showinfo("Process ROI Traces", "No recordings with completed ROIs found.")
                return
            proceed = messagebox.askyesno(
                "Process ROI Traces",
                "No completed recordings selected. Process all completed recordings?",
            )
            if not proceed:
                return
            entries = self.completed_entries

        print(f"[roi_gui] Processing {len(entries)} recording(s).")
        failures: list[str] = []
        for entry in entries:
            print(f"[roi_gui] Processing entry: {entry.name}")
            print(f"[roi_gui] Manifest: {entry.manifest_path}")
            print(f"[roi_gui] ROI path: {entry.roi_path}")
            if not entry.roi_exists:
                failures.append(f"{entry.name}: ROI file missing.")
                continue
            try:
                print("[roi_gui] Starting ROI analysis (generate_movies=False).")
                process_roi_analysis(entry.manifest_path, entry.roi_path)
                print(f"[roi_gui] Finished ROI analysis: {entry.name}")
            except Exception as exc:
                print(f"[roi_gui] ROI analysis failed: {entry.name}")
                print(traceback.format_exc())
                failures.append(f"{entry.name}: {exc}")

        if failures:
            messagebox.showerror(
                "Process ROI Traces",
                "Some recordings failed to process:\n" + "\n".join(failures),
            )
        else:
            messagebox.showinfo("Process ROI Traces", "ROI trace extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a GUI to track ROI curation status.")
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Root directory containing recording folders with processing_manifest.json files.",
    )
    args = parser.parse_args()

    if not args.project_root.exists():
        raise FileNotFoundError(f"Project root not found: {args.project_root}")

    print(f"[roi_gui] Launching ROI tracker for: {args.project_root}")
    root = tk.Tk()
    app = RoiTrackerApp(root, args.project_root)
    root.mainloop()


if __name__ == "__main__":
    main()
