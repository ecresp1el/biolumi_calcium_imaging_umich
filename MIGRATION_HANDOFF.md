# Calcium imaging workflow: migration handoff

## Purpose

Build a new, focused repository that lets a non-technical user reproduce the
validated calcium-imaging workflow from Imaris input to ROI movies. The legacy
repository remains read-only reference material until the new workflow passes
acceptance tests on representative data.

## Canonical workflow to preserve

```text
Imaris .ims
  -> TIFF conversion + motion correction + projections
  -> manual ROI annotation in Napari
  -> adaptive sliding-F0, dF/F, smoothing, peaks, QC PDF, and CSVs
  -> ROI-outline + time-locked trace MP4
```

### Image data type requirement

The streamlined workflow writes all generated imaging assets as `uint16` in
the original 0–65535 intensity range: raw movie, motion-corrected movie, and
max/average/standard-deviation projections. Motion correction is calculated in
floating point internally, then rounded and clipped before writing; it does not
apply display normalization or intensity rescaling.

This project should *not* carry forward the experimental DREADD comparison
scripts, red-ROI comparison logic, historical notebooks, duplicate analysis
routes, or project-specific hard-coded paths unless a later decision explicitly
requires them.

## Legacy source map

| New capability | Legacy source | Keep now? | Notes |
| --- | --- | --- | --- |
| Imaris conversion and motion correction | `BL_CalciumAnalysis/pipeline.py`, `preprocess_cli.py` | Yes | Requires CaImAn; lock parameters after representative-data review. |
| Manual ROI drawing | `BL_CalciumAnalysis/napari_roi_cli.py` | Yes | Preserve 2D/3D ROI validation and saving. |
| Trace extraction and standard outputs | `BL_CalciumAnalysis/contracted_signal_extraction.py` | Yes | Adopt this as the one analysis route. |
| ROI/trace MP4 | `custom_analysis_scripts/dreadd_stim_roi_movie.py` | Yes, refactor | Remove DREADD name and hard-coded project root. |
| Older `roi_processing.py` outputs | `BL_CalciumAnalysis/roi_processing.py` | No, initially | Overlaps with the contracted pipeline; retain only as reference. |
| ROI tracker GUI | `BL_CalciumAnalysis/roi_gui.py` | Later | Add only after the command-line workflow is reliable. |
| DREADD/red-ROI validation scripts | `custom_analysis_scripts/dreadd_stim_*validation*.py` | No | Experiment-specific analysis, not core processing. |
| Notebooks | `notebooks/` | No | Historical evidence only; no production dependency. |

## Current evidence and constraints

- The preserved code implements all four required stages and records processing
  locations in `processing_manifest.json`.
- The dedicated `napari-env` works now: Python 3.10.19, Napari 0.4.19, CaImAn,
  numerical/plotting libraries, and FFmpeg import or execute successfully.
- The repository-local `.conda` environment is not usable for this pipeline
  because it lacks `matplotlib`.
- The historical data roots were on `/Volumes/Manny4TBUM/`; that drive is not
  mounted now. An end-to-end claim remains pending a representative recording.
- The legacy repo has no automated test suite, so migration requires new tests
  before the legacy implementation is changed or discarded.

## Read-only readiness audit — 2026-07-31

The `Manny4TBUM` volume is mounted. The following checks were performed without
modifying any imaging data.

| Workflow stage | Result | Evidence |
| --- | --- | --- |
| Source Imaris input | Pass | Two original `.ims` inputs named by the chemogenetic manifests exist under `/Volumes/Manny4TBUM/chem_dreadd_sorted_files/`. |
| Preprocessing | Pass | Both chemogenetic recording manifests resolve to existing raw TIFF, motion-corrected TIFF, max, mean, and standard-deviation projections. Motion-correction settings are retained in each manifest. |
| Napari ROI annotation | Pass | Both recordings contain one 3D uint16 ROI label stack matching the movie spatial dimensions: 24 ROIs in the “after” recording and 17 in the “before” recording. |
| Adaptive extraction, smoothing, peaks, and QC | Pass | Both recordings contain sliding dF/F and smoothed dF/F CSVs, two peak-count CSVs, and a non-empty QC PDF. ROI identifiers equal the trace columns. The 995 trace frames are expected: the legacy code excludes the first five frames for its 1-s bleach-fit window at 5 fps. |
| ROI/traces MP4 | Pass | Both recordings contain non-empty H.264 MP4s. Each decoded movie has 995 frames and valid dimensions (1200 x 608). |
| Existing runtime | Pass | `napari-env` imports Napari, CaImAn, SciPy, pandas, TIFF/plotting libraries, and can run FFmpeg. |
| Fresh environment recreation | Pass | On 2026-07-31, a disposable Apple-Silicon Conda environment was created from `env/environment.yml`. It installed 494 packages and passed imports for Python 3.10.20, Napari 0.4.19.post1, CaImAn 1.13.0, NumPy, SciPy, pandas, TIFF/ImageIO, matplotlib, h5py, scikit-image, and statsmodels. FFmpeg 8.1.2 and the preprocessing/extraction CLI help commands also ran successfully. |
| New streamlined repository | Stages 1–2 complete; Stage 3 pilot complete | `organoid_calcium_imaging_workflow` has a tested environment, manifest, Imaris reading, uint16 preprocessing/projections, Napari 2D ROI annotation, ROI validation, and one adaptive-F0 pilot analysis. MP4 generation remains pending. |

The two completed chemogenetic recordings are valid candidates for the Stage 0
reference fixture. The “after” recording has 24 ROIs and seven surviving MP4
outputs; the “before” recording has 17 ROIs and six. Mac Finder metadata files
named `._*` were excluded from the output checks.

## Proposed new repository contract

New repository name: `organoid_calcium_imaging_workflow`.

The user-facing interface should be one command (and later a simple launcher):

```bash
organoid-calcium-workflow run /path/to/recording.ims --output /path/to/project
```

It pauses after preprocessing for manual ROIs, then resumes with:

```bash
organoid-calcium-workflow analyze /path/to/project/recording/processing_manifest.json
organoid-calcium-workflow movie /path/to/project/recording/processing_manifest.json
```

The project must produce this stable recording layout:

```text
project/
  recording_name/
    processing_manifest.json
    raw/
    motion_corrected/
    projections/
    rois/
    analysis/
      traces/
      qc/
      movies/
```

Every command should read locations from the manifest, never from a hard-coded
drive path. Commands must fail with an actionable message if an expected input
is absent or has a mismatched shape.

### Fresh-input and scratch-output rule

Preprocessing reads only from an immutable source-only input tree and writes
only to a separate disposable scratch root. The source keeps its natural
experimental hierarchy and contains `.ims` files plus nearby metadata text.
Generated output preserves source-relative parent folders:

```text
scratch_root/<source-relative-parent>/<ims-stem>/
  raw/movie_raw.tif
  motion_corrected/movie_motion_corrected.tif
  projections/{max,average,std}_projection.tif
  processing_manifest.json
```

Preprocessing assumes Imaris resolution level 0, calcium channel 0, and 2D or
max-collapsible Z data. A `ready_for_roi` manifest is the handoff point to
manual ROI annotation; preprocessing does not start trace analysis or movies.

## Multi-stage handoff plan

### Stage 0 — Freeze and select reference data

**Goal:** establish a trustworthy baseline without editing legacy code.

1. Reconnect the `Manny4TBUM` volume.
2. Identify one small representative `.ims` recording and its completed legacy
   output folder (ROI TIFF, analysis CSVs/PDF, and at least one MP4).
3. Copy only that recording and outputs into a controlled migration fixture
   location; do not commit raw imaging data to Git.
4. Record acquisition frame rate, Imaris channel selections, motion-correction
   settings, and any manual decisions made during ROI drawing.
5. Create checksums and a concise expected-output inventory.

**Exit criterion:** one fixture has a documented input-to-output chain.

### Stage 1 — Make the environment reproducible — Complete (2026-07-31)

**Goal:** replace environment assumptions with a tested, single setup path.

The legacy environment definition is `env/environment.yml`. It has been tested
successfully as written. It specifies a compatible Python 3.10 Napari/CaImAn
stack plus FFmpeg, ImageIO, TIFF, SciPy, pandas, matplotlib, scikit-image,
h5py, and statsmodels. A full package-version lock file is **not** required for
this project.

#### User setup instructions

The new repository will contain this same file at its top level as
`environment.yml`. A first-time user will run:

```bash
cd /path/to/organoid_calcium_imaging_workflow
conda env create --file environment.yml
conda activate organoid-calcium-workflow
```

The current legacy setup uses the same commands with its current file and name:

```bash
cd /Users/ecrespo/Documents/github_project_folder/biolumi_calcium_imaging_umich
conda env create --file env/environment.yml
conda activate napari-env
```

#### What Stage 1 tests

Stage 1 is an environment-only check; it does not run a real movie, open a
Napari window, or change any imaging data. It must verify that a *newly created*
environment can:

1. import Python, Napari, CaImAn, NumPy, SciPy, pandas, tifffile, ImageIO,
   matplotlib, h5py, scikit-image, and statsmodels;
2. run `ffmpeg -version` for MP4 generation;
3. load the preprocessing CLI and adaptive-extraction CLI with `--help`;
4. run the non-GUI environment diagnostic, which reports numerical-library
   versions and verifies the LOWESS implementation used by the analysis.

The clean-install test should use a disposable prefix so it never changes a
user's existing environment:

```bash
TEST_ROOT=$(mktemp -d /tmp/calcium-stage1.XXXXXX)
conda env create --prefix "$TEST_ROOT/organoid-calcium-workflow" --file environment.yml --yes
"$TEST_ROOT/organoid-calcium-workflow/bin/python" -c \
  'import napari, caiman, numpy, scipy, pandas, tifffile, imageio, matplotlib, h5py, skimage, statsmodels; print("Environment imports: OK")'
"$TEST_ROOT/organoid-calcium-workflow/bin/ffmpeg" -version
```

While the legacy repository is the reference implementation, also run its
non-GUI compatibility checks with the fresh interpreter:

```bash
LEGACY_ROOT=/Users/ecrespo/Documents/github_project_folder/biolumi_calcium_imaging_umich
PYTHONPATH="$LEGACY_ROOT" "$TEST_ROOT/organoid-calcium-workflow/bin/python" \
  -c 'from pathlib import Path; from BL_CalciumAnalysis.contracted_signal_extraction import validate_environment; print(validate_environment(Path("env/environment.yml")))'
PYTHONPATH="$LEGACY_ROOT" "$TEST_ROOT/organoid-calcium-workflow/bin/python" \
  -m BL_CalciumAnalysis.preprocess_cli --help
PYTHONPATH="$LEGACY_ROOT" "$TEST_ROOT/organoid-calcium-workflow/bin/python" \
  -m BL_CalciumAnalysis.contracted_signal_extraction --help
```

Remove the disposable environment using `conda env remove --prefix
"$TEST_ROOT/organoid-calcium-workflow" --yes` after the test succeeds.

#### Result of this handoff audit

**Pass (2026-07-31).** A clean temporary environment was created from
`env/environment.yml`. It installed 494 packages, imported the complete
dependency set, ran FFmpeg, and loaded both legacy CLI entry points. The
temporary environment was removed after the test.

**Exit criterion:** the new repository's `environment.yml` passes the same
clean-install checks before Stage 2 code is added.

### Stage 2 — Create the small processing core — Complete (2026-08-02)

**Goal:** migrate only preprocessing, manifest handling, and ROI validation.

1. Start the new repository with a package layout, command-line entry points,
   README, license, `.gitignore`, and no notebooks or experiment scripts.
2. Port Imaris reading, TIFF conversion, motion correction, and projections.
3. Preserve the manifest concept but define and validate its schema.
4. Port the Napari ROI viewer as a single `annotate` command; save 2D labels by
   default and support 3D labels only when needed.
5. Add synthetic TIFF/ROI fixtures and tests for manifest generation, output
   locations, and ROI/movie shape compatibility.

**Result:** The new repository provides `preprocess-root`, `annotate`, and
`validate-roi` commands. A real, isolated Gaillard pilot was preprocessed to
uint16 TIFFs, annotated with three 2D Napari ROIs, and validated against the
motion-corrected movie. The source-only input was not modified.

#### Stage 2 operator instructions

Run these commands from the new repository, substituting the scratch recording
directory produced by Stage 1 preprocessing:

```bash
cd /path/to/organoid_calcium_imaging_workflow
conda activate organoid-calcium-workflow

RECORDING=/path/to/disposable_scratch/<source-relative-parent>/<ims-stem>
PYTHONPATH=src python -m organoid_calcium_imaging_workflow.cli annotate \
  --manifest "$RECORDING/processing_manifest.json"
```

In Napari, select `roi_labels`, draw each ROI with a unique nonzero label
number, and close the window. Closing writes
`$RECORDING/rois/roi_labels.tif` automatically. Then run:

```bash
PYTHONPATH=src python -m organoid_calcium_imaging_workflow.cli validate-roi \
  --movie "$RECORDING/motion_corrected/movie_motion_corrected.tif" \
  --roi "$RECORDING/rois/roi_labels.tif"
```

**Exit criterion:** `validate-roi` reports `ROI validation passed` and a
nonzero ROI count. No imaging file in the source-only tree is changed.

#### Manual-mask import acceptance test — Implementation complete; real-data test pending

**Goal:** support a mask drawn outside this workflow without redrawing it or
changing its original file.

1. The new repository now provides `add-manual-masks`, which accepts a
   recording manifest and an existing label TIFF.
2. Require an integer-valued 2D label TIFF whose `(Y, X)` exactly matches the
   recording's motion-corrected movie. The analysis path uses this 2D label
   form.
3. Copy, never move or overwrite, the supplied original into
   `rois/imported/<original-name>.tif`; write the selected active copy as
   `rois/roi_labels.tif` only after validation succeeds.
4. Record the original path, imported-copy path, checksum, label count, and
   image shape in `processing_manifest.json`.
5. Refuse a dimension mismatch or non-integer mask. Do not resize, shift,
   interpolate, or otherwise attempt to align a mask automatically.
6. The command passed synthetic tests: it copies the mask, leaves the original
   byte-for-byte unchanged, writes provenance to the manifest, rejects a
   mismatched shape, and refuses to replace existing active labels unless the
   user provides `--replace-active`.
7. Real-data audit (2026-08-03): two independently created Gaillard masks were
   located. The Day 110 mask is a 2D `uint32` 512 x 512 label TIFF; it exactly
   matches the spatial dimensions of its 720-frame original TIFF and is a
   candidate once its matching MGEO recording has been preprocessed into a
   scratch manifest. The Day 121 mask is a 3D `uint32` 360 x 996 x 1020 label
   stack that matches its original movie but is intentionally rejected by the
   current 2D analysis contract. Neither MGEO recording has yet been
   preprocessed under the new scratch layout, so neither mask was imported. If
   a mask was drawn on an uncorrected or differently sized movie, stop for
   manual alignment review rather than importing it.

#### Operator command for an externally made mask

```bash
PYTHONPATH=src python -m organoid_calcium_imaging_workflow.cli add-manual-masks \
  --manifest "$RECORDING/processing_manifest.json" \
  --mask /path/to/existing_manual_labels.tif
```

The command stops if `rois/roi_labels.tif` already exists. Only after checking
the intended replacement should the operator add `--replace-active`.

**Remaining exit criterion:** preprocess the Day 110 MGEO recording into a
fresh scratch root, then import its matching 2D mask with the command above.
It must pass ROI validation and run through analysis while the original mask
remains untouched.

### Stage 3 — Migrate exactly one analysis route — Pilot implemented

**Goal:** preserve the adaptive extraction that worked, without duplicate logic.

1. The new repository now contains only one analysis route: raw ROI extraction,
   adaptive-percentile F0, ΔF/F, one-second smoothing, peaks, CSV outputs, and
   a raw-intensity-plus-QC plot. It does not expose the retired duplicate
   analysis routes.
2. The adaptive F0 settings currently encoded are a 30-s window, activity
   fraction 0.3, and low/high percentiles 10/10. Frame rate is supplied
   explicitly per recording; the Gaillard pilot used its metadata-derived 4
   fps.
3. The pilot output folder is deliberately restricted to raw traces, adaptive
   F0, percentile-used, ΔF/F, smoothed ΔF/F, peak table, and QC PNG. Obsolete
   sliding-F0 filenames were removed from the scratch pilot.
4. Synthetic tests cover adaptive-F0 shapes and percentile behavior; the
   current suite passes eight tests.

**Remaining exit criterion:** review the pilot traces scientifically, decide
whether the configured 10/10 percentile range should remain fixed or vary with
activity, then compare the accepted output against a completed reference
recording before declaring Stage 3 complete.

### Stage 4 — Migrate the MP4 output

**Goal:** make the final visual deliverable dependable and configurable.

1. Port only the generic components of `dreadd_stim_roi_movie.py`.
2. Read movie, ROI, dF/F, and frame rate through the manifest and analysis
   configuration; remove the DREADD project constant.
3. Retain configurable outline thickness, top-K selection, frame stride, and
   playback speed, with clear defaults.
4. Add a tiny movie fixture test that verifies an MP4 is produced and a manual
   visual acceptance check for colors, outlines, time marker, and labels.

**Exit criterion:** a non-technical user can generate a playable ROI-and-trace
movie with one command after analysis.

### Stage 5 — Non-technical usability layer

**Goal:** make the validated command-line workflow easy to operate safely.

1. Add a desktop launcher or minimal GUI that presents the four stages,
   remembers the selected project directory, shows progress, and displays the
   next required action.
2. Provide a preflight panel that checks the environment, disk space, input
   readability, and ROI compatibility before a long run.
3. Add plain-language errors and a `report` command that links every produced
   file.
4. Write a short illustrated “first recording” guide and a troubleshooting
   page.

**Exit criterion:** a colleague who did not write the code completes the
fixture workflow from the guide without shell-level debugging.

### Stage 6 — Handoff, release, and retirement decision

**Goal:** make the new repository the source of truth.

1. Run the complete new pipeline on the representative recording and compare
   its outputs with the legacy reference inventory.
2. Tag a release, export a pinned environment lock file, and archive a
   provenance report with the fixture checksums and software versions.
3. Keep the legacy repository unchanged and clearly label it as archived
   reference code.
4. Decide separately whether any red-ROI or cross-recording comparison modules
   merit a future extension.

**Exit criterion:** the new repo reproduces the agreed reference workflow,
documentation, and release artifacts without reliance on legacy paths.

## Decisions to make before implementation

1. Confirm the new repository name and whether it should be public or private.
2. Choose the representative recording once the external drive is mounted.
3. Confirm whether red-channel ROI support belongs in the first release; the
   recommended answer is **no** unless it is needed for every experiment.
4. Confirm whether “non-technical” means a guided command-line launcher first
   or a full GUI in the first release; the recommended sequence is CLI first,
   GUI after the core passes acceptance tests.

## Immediate next action

Reconnect `Manny4TBUM` and identify one completed recording directory. We will
inventory its manifest, ROI labels, analysis files, and MP4s to create the
Stage 0 reference fixture before writing the new repository.
