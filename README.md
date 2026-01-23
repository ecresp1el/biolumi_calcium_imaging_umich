
# Biolumi Calcium Imaging Analysis

📌 Overview

This project provides a structured approach for analyzing calcium imaging data. It ensures reproducibility, organization, and automation by setting up a standardized directory structure and enabling image processing via the command line.

🛠️ Installation & Setup

## Pipeline overview (what happens, in order)
1. **Preprocess (Imaris → TIFF + motion correction)**  
   `preprocess_cli` converts `.ims` recordings to motion-corrected TIFF stacks and generates projection images. A `processing_manifest.json` file is saved with full paths to every generated asset.
2. **Manual ROI annotation (Napari)**  
   `napari_roi_cli` opens the motion-corrected movie and max projection, then writes a 3D ROI label TIFF (T, Y, X) that matches the movie shape.
3. **ROI analysis (trace extraction + plots)**  
   `roi_processing.process_roi_analysis` reads the manifest + ROI labels, writes ROI trace CSVs, ΔF/F CSVs, and trace plots into a `roi_analysis/` directory. Optional movies and projection TIFFs can be generated when requested.
4. **ROI curation tracker (GUI)**  
   `roi_gui` helps track which recordings are missing ROI labels and can trigger ROI analysis for completed recordings.

1️⃣ Set Up the Conda Environment

Ensure you have Miniconda or Anaconda installed.

Run the following command to create and activate the environment:

2️⃣ Set Up the Project Directory Structure

Run the setup script to create a standardized directory tree for storing image data:

You'll be prompted to enter:

Project directory name (default: biolumi_project)

Number of groups (default: 2)

Number of recordings per group (default: 2)

This will create a structure like:

```
project_root/
   recording_1/
      processing_manifest.json
      raw/
      motion_corrected/
      projections/
      roi/
      roi_analysis/
   recording_2/
      ...
```

> **Note:** `processing_manifest.json` is the single source of truth for where each output lives. It is updated after ROI analysis with a `roi_analysis` section that records the outputs described below.

3️⃣ Process Images Using CLI

After setting up the structure, you can process images.

🔹 Process a Single Image

🔹 Process All Images in a Directory

4️⃣ Running Tests

To verify that everything works correctly, run:

🧠 Manual ROI Annotation with Napari

Use the Napari helper script to open a motion-corrected TIFF movie and a max
projection image, then draw/save manual ROI labels (3D labels matching the
movie time axis). When you close Napari, the CLI prints ROI count, shape, dtype,
and the output path.

Example usage:

```
python -m BL_CalciumAnalysis.napari_roi_cli \
  --movie /path/to/motion_corrected.tif \
  --max-projection /path/to/max_projection.tif \
  --roi /path/to/roi_masks_uint16.tif \
  --save-roi /path/to/roi_masks_uint16.tif \
  --strict
```

🧪 Preprocess an Imaris movie (motion correction + projections)

Use the preprocessing CLI to convert an Imaris `.ims` recording (or every `.ims`
file in a directory) to a TIFF stack, run CaImAn motion correction, and save
max/avg/std projections plus a manifest JSON in a recording-specific folder.

Example usage:

```
python -m BL_CalciumAnalysis.preprocess_cli \
  --ims "/Volumes/Manny4TBUM/12_3_2025/test_gcamp_dreadd_dtom_Confocal - Green_2025-12-03_2.ims" \
  --output-root "/path/to/project_root"
```

Process all `.ims` files in a directory:

```
python -m BL_CalciumAnalysis.preprocess_cli \
  --ims "/Volumes/Manny4TBUM/12_5_2025/2025-12-05" \
  --output-root "/path/to/project_root"
```

🖥️ ROI Curation Tracker (GUI)

Use the GUI to track which recordings still need ROI work and launch Napari
directly for a selected recording. The GUI reads `processing_manifest.json`
files created by preprocessing.

Example usage:

```
python -m BL_CalciumAnalysis.roi_gui \
  --project-root "/path/to/project_root"
```

Recordings missing the motion-corrected movie or max projection will appear in
the “Missing Inputs” list so you can identify preprocessing issues quickly.

## Manifest ground truth (fields + data types)
The preprocessing step writes a `processing_manifest.json` file inside each recording folder.
Below is the expected structure and data types for the fields used downstream:

- `paths` (object)
  - `raw_tiff` (string): Path to the raw TIFF stack (T, Y, X) extracted from the `.ims`.
  - `motion_corrected_tiff` (string): Path to the motion-corrected TIFF stack (T, Y, X).
  - `max_projection` (string): Path to max-projection TIFF (Y, X) from the motion-corrected movie.
  - `avg_projection` (string): Path to avg-projection TIFF (Y, X).
  - `std_projection` (string): Path to std-projection TIFF (Y, X).
- `roi_analysis` (object, added after ROI analysis)
  - See “ROI analysis outputs” below for each field, file type, and contents.

## ROI analysis outputs (every file generated in `roi_analysis/`)
When `process_roi_analysis` runs, it creates (or updates) a `roi_analysis/` folder
alongside the manifest. Each file is listed below with its contents and data type:

1. **`*_raw_vs_mc_withtext.mp4`** (MP4 video, optional)  
   Side-by-side movie of the raw stack and the motion-corrected stack. Each frame is
   normalized independently (1st–99th percentile) and labeled “ORIGINAL” and
   “MOTION CORRECTED.”
2. **`*_MCMOVIE_uint16.tif`** (TIFF stack, optional)  
   Motion-corrected movie normalized and saved as uint16 for visualization.
3. **`*_MAXPROJ.tif`** (TIFF image, optional)  
   Max projection of the uint16 motion-corrected movie.
4. **`*_AVGPROJ.tif`** (TIFF image, optional)  
   Mean projection of the uint16 motion-corrected movie.
5. **`*_STDPROJ.tif`** (TIFF image, optional)  
   Standard-deviation projection of the uint16 motion-corrected movie.
6. **`*_static_roi_labels.tif`** (TIFF image, uint16)  
   A 2D label image (Y, X) created by collapsing the ROI labels across time
   (`max` across T). Each ROI has a unique integer ID; background is 0.
7. **`*_roi_traces.csv`** (CSV)  
   Raw fluorescence traces. Each column is an ROI ID and each row is a frame.
   Values are the mean uint16 intensity inside each ROI, per frame.
8. **`*_roi_dff.csv`** (CSV)  
   ΔF/F traces computed from the raw traces using a baseline percentile (default 10th).
9. **`*_roi_traces.png`** (PNG)  
   Plot of raw ROI traces with vertical offsets for readability (one line per ROI).
10. **`*_roi_dff.png`** (PNG)  
   Plot of ΔF/F traces (one line per ROI).
11. **`*_MC_ROI_TRACE_GRID_OUTLINES.mp4`** (MP4 video, optional)  
   Diagnostic movie grid that shows: motion-corrected frame, ROI overlay, a growing
   ΔF/F plot, and per-ROI ΔF/F traces with an inset ROI crop and outline.
12. **`movies_skipped.txt`** (text file, optional)  
   Written when movies/projections are disabled. States which movie outputs were skipped.

🚀 Future Enhancements

Automate metadata collection.

Add parallel processing for large datasets.

Enhance logging and error handling.

This project is designed to be scalable, efficient, and user-friendly.Feel free to contribute or suggest improvements! 🚀
