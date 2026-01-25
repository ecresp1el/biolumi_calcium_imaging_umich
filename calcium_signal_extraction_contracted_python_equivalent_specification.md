# Calcium Signal Extraction Pipeline
## Python-Equivalent, Contracted Specification (Code-Free)

---

### Purpose of this Document
This document defines a **drop‑in preprocessing specification** for calcium signal extraction that reproduces a MATLAB‑based pipeline described in the literature, while allowing **surgical insertion** into an existing Python analysis stack **without altering downstream plotting, spike detection, or data contracts**.

This is intentionally **code‑free**. All implementations are described *conceptually*, with explicit references to the **Python libraries and functions** that serve as direct equivalents to the MATLAB operations used in the paper.

This document is suitable for:
- A secondary README
- A Word document (copy‑paste compatible)
- A methods‑alignment contract for a new repository

---

## Governing Design Rule (Authoritative)

**Photobleaching correction must occur before any per‑ROI slow baseline estimation or ΔF/F computation.**

Photobleaching and slow baseline drift represent **distinct physical processes** and must not be conflated.

---

## High‑Level Logical Progression (Locked Order)

1. Raw fluorescence loading (unchanged)
2. Rough spike detection (masking only)
3. Identification of silent periods
4. Global photobleaching baseline estimation (bi‑exponential)
5. Bleaching correction via division
6. Per‑ROI slow baseline estimation (robust LOWESS)
7. ΔF/F computation (unchanged contract)
8. Spike detection and amplitude logic (unchanged)

Downstream analysis assumes **only ΔF/F** and is therefore unaffected by steps 2–6.

---

## Step‑by‑Step Specification

### 1. Rough Spike Detection (Mask‑Only Stage)

**Purpose**  
To identify transient calcium events *solely* for the purpose of excluding them from photobleaching estimation.

**Key Characteristics**
- Permissive detection
- No amplitude accuracy required
- Output is a boolean mask only

**MATLAB Reference**  
Implicit (preprocessing step prior to bleaching correction)

**Python Equivalent (Conceptual)**
- Z‑score based transient detection
- Implemented using numerical standardization and thresholding

**Python Libraries**
- NumPy
- SciPy (statistics utilities)

**Contract Check**
☐ This mask is **never reused** downstream

---

### 2. Silent Period Identification

**Purpose**  
Define fluorescence segments suitable for estimating photobleaching without spike contamination.

**Definition**
Silent periods are frames not flagged during rough spike detection.

**Python Equivalent (Conceptual)**
- Boolean inversion of spike mask

**Contract Check**
☐ Silent periods span the full recording duration

---

### 3. Global Photobleaching Baseline Estimation

**Purpose**  
Estimate slow, multiplicative fluorescence decay due to photobleaching.

**Model Used in Paper**
A two‑component exponential decay:

B(t) = A₁·exp(−t/τ₁) + A₂·exp(−t/τ₂) + C

**Critical Properties**
- Fitted only on silent periods
- Estimated **once per recording**
- Shared across all ROIs

**MATLAB Reference**
Custom exponential fitting (not explicitly named)

**Python Equivalent (Conceptual)**
- Nonlinear least‑squares fitting of a bi‑exponential model

**Python Libraries**
- SciPy (`optimize.curve_fit`)

**Contract Checks**
☐ Baseline is global (not ROI‑specific)  
☐ Fit excludes spike‑contaminated frames

---

### 4. Photobleaching Correction

**Purpose**  
Remove multiplicative decay while preserving relative signal amplitudes.

**Operation**
Raw fluorescence is **divided** by the bleaching baseline.

**Why Division (Not Subtraction)**
Photobleaching scales signal magnitude proportionally; subtraction would distort amplitudes.

**MATLAB Reference**
“Bleaching baseline was then divided from the entire recording.”

**Python Equivalent (Conceptual)**
- Element‑wise division using array broadcasting

**Contract Checks**
☐ Same bleaching baseline applied to all ROIs  
☐ Occurs before slow baseline estimation

---

### 5. Slow Baseline Estimation (Per‑ROI)

**Purpose**  
Remove slow baseline fluctuations while preserving calcium transients.

**MATLAB Reference**
`smooth(..., 'rlowess', 100 frames)`

**Key Parameters**
- Robust LOWESS smoothing
- Window:(~2.4 seconds)
- Down‑weights outliers

**Python Equivalent (Conceptual)**
- Robust LOWESS smoothing using locally weighted regression

**Python Libraries**
- statsmodels (LOWESS implementation)

**Contract Checks**
☐ Applied **after** bleaching correction  
☐ Operates independently per ROI

---

### 6. ΔF/F Computation (Locked Contract)

**Definition**
ΔF/F = (F_corrected − F_baseline) / median(F_baseline)

**MATLAB Reference**
Explicitly stated in the paper

**Python Equivalent (Conceptual)**
- Array subtraction
- Median normalization

**Contract Checks**
☐ Formula identical to original pipeline  
☐ Downstream code unchanged

---

### 7. Spike Detection and Amplitude Analysis

**Status**
Intentionally **unchanged**.

**Preserved Behaviors**
- Z‑score thresholding
- Exponential tail subtraction when applicable

**Contract Check**
☐ Operates strictly on ΔF/F

---

## Validation Checklist (Required)

### Structural Integrity
☐ No downstream function signatures changed  
☐ ΔF/F array shape unchanged  
☐ Plotting code runs without modification

### Numerical Consistency
☐ Event amplitudes preserved (excluding bleaching trend)  
☐ Spike counts stable within expected tolerance

### Visual Sanity Checks
☐ Bleaching‑corrected traces show flattened slow decay  
☐ LOWESS baseline preserves transient peaks

### Explicit Failure Modes
☐ No ROI‑specific bleaching fits  
☐ No bleaching correction after ΔF/F


## Environment & Dependency Verification (Mandatory Preflight)

Before executing **any step** in this pipeline, the runtime environment **must be validated**. This is a hard requirement because numerical smoothing, nonlinear fitting, and statistical utilities are **library‑version sensitive**.

### Environment Contract

- The repository **must** contain an explicit `environment.yml` (or equivalent lock file).
- All packages required for each processing stage **must be declared** before that stage is allowed to run.
- No implicit dependencies are permitted.

### Required Capability Mapping

Each logical stage depends on the following *capabilities* (not implementations):

1. Numerical array operations
2. Statistical standardization
3. Nonlinear least‑squares optimization
4. Robust locally weighted regression (LOWESS)
5. Median‑based normalization

If **any capability is missing or version‑incompatible**, execution must halt **before data modification**.

### Verification Checklist (Run Before Processing)

☐ Environment resolves without solver errors  
☐ Numerical library versions are recorded (NumPy / SciPy)  
☐ LOWESS implementation supports robust iteration  
☐ Optimization library supports constrained nonlinear fitting  
☐ Environment hash is logged with outputs

Failure at this stage invalidates downstream comparisons.

---

## Surgical Insertion Contract (Non‑Destructive Requirement)

This pipeline modification is defined as a **surgical insertion**, not a refactor.

### Non‑Destructive Definition

- No downstream function signatures may change
- No plotting functions may be edited
- No analysis code may be reordered
- Only **one new upstream artifact** is introduced: *bleaching‑corrected fluorescence*

All downstream code must be able to operate on **either**:

- Legacy ΔF/F streams (baseline pipeline)
- Bleaching‑corrected ΔF/F streams (new pipeline)

without modification.

---

## Dual‑Stream Plotting Requirement

All plotting and re‑analysis must support **side‑by‑side comparability**.

### Required Plot Modes

A) **Legacy Mode**  
Plots generated using the original ΔF/F pipeline

B) **Corrected Mode**  
Plots generated using bleaching‑corrected ΔF/F

The plotting interface must allow seamless substitution of data streams.

### Plotting Contract

- Identical axes
- Identical normalization
- Identical aggregation logic
- Identical visual encodings

If a plot cannot be generated in both modes, the pipeline is considered **broken**.

---

## Compatibility & Breakpoint Identification

If outputs diverge, the pipeline **must identify the earliest incompatible stage**.

### Authorized Breakpoints

1. Photobleaching baseline estimation  
   *Expected difference*: removal of slow decay

2. Slow baseline estimation  
   *Risk*: window mis‑scaling or over‑flattening

3. ΔF/F normalization  
   *Risk*: median shift

4. Spike amplitude computation  
   *Risk*: altered peak‑to‑baseline contrast

No other divergence points are acceptable.

---

## Failure Diagnosis Protocol

If comparability fails, testing proceeds in this order:

1. Compare raw fluorescence (must match exactly)
2. Compare bleaching baseline (shape only)
3. Compare corrected fluorescence (ratio stability)
4. Compare slow baseline (peak preservation)
5. Compare ΔF/F distributions
6. Compare spike amplitudes

Failure at any step **must stop validation** and be logged.

---

## Validation Matrix (Pass/Fail)

☐ Environment verified before execution  
☐ Legacy plots reproducible  
☐ Corrected plots reproducible  
☐ Dual‑stream substitution works  
☐ Divergences localized and explained  
☐ No silent numerical drift

---

## Final Statement

This specification defines a **controlled, testable, and reversible insertion** of photobleaching correction into an existing calcium imaging pipeline.

All changes are:
- Environment‑validated
- Contract‑bound
- Dual‑stream comparable
- Failure‑diagnosable

Any deviation from this protocol invalidates quantitative comparisons and must be resolved before adoption.

---

## Implementation Audit (Current Code Path)

- **Bleaching correction**: Disabled (bleaching baseline set to ones; no division applied). Legacy fields remain for compatibility but contain placeholders.
- **Spike masking (diagnostic only)**: Per‑ROI local z‑scores (rolling median/MAD, 5 s window, z>2), time‑based expansion (±2.5 s), union across ROIs; not used to modulate bleaching.
- **Sliding F0**: Per‑ROI sliding window (default ~1 s, derived from FPS, can be overridden) with fixed percentile at the 10th (low=high=10) for all frames; yields F0(t) and dF/F = (F−F0)/F0.
- **Peak detection**: Per‑ROI peaks on sliding‑F0 dF/F with a static threshold mean+2·std over the full trace (z≈2), positive‑going only.
- **QC PDF (contract)**: Per‑ROI pages with three panels: (1) raw + sliding F0 (10th percentile); (2) percentile used (constant 10); (3) sliding‑F0 dF/F with z=2 threshold and detected peaks.
- **Outputs**: Contract CSVs for raw traces, (placeholder) bleaching baseline, slow baselines, sliding F0, sliding percentiles, sliding dF/F, per‑ROI peaks, per‑ROI peak counts; QC PDF as above.

---

## How to Run (Current Implementation)

```
# Batch (all recordings under project root)
python -m BL_CalciumAnalysis.contracted_signal_extraction \
  --project-root "/path/to/project_root" \
  --fps 5

# Single recording
python -m BL_CalciumAnalysis.contracted_signal_extraction \
  --manifest "/path/to/processing_manifest.json" \
  --roi "/path/to/roi_masks_uint16.tif" \
  --fps 5
```

Outputs are written to `roi_analysis_contract/` alongside the manifest:
- `*_contract_qc.pdf` — 3-panel pages per ROI (raw+F0, percentile, dF/F+peaks)
- `*_roi_sliding_f0.csv`, `*_roi_sliding_pct.csv`, `*_roi_sliding_dff.csv`
- `*_roi_peaks.csv`, `*_roi_peak_counts.csv`
- ROI1 debug: `*_roi1_sliding_f0_debug.{csv,png}`, `*_roi1_peaks.csv`
- Legacy placeholders: bleaching baseline CSV/JSON (set to ones, “disabled”)

---

## Function I/O (Key Routines)

- `compute_sliding_f0_adaptive(trace, fps, target_window_s=50, max_fraction=0.4, activity_fraction=0.3, low_percentile=5, high_percentile=50)`  
  Returns `(f0, percentile_used, window_frames)` per frame for one ROI trace using adaptive percentiles in a time-based sliding window.

- `process_contract_analysis(manifest_path, roi_path, output_dir=None, config=None)`  
  Reads movie + ROIs from the manifest, writes all contract outputs to `roi_analysis_contract/` (sliding F0/dF/F/peaks, QC PDF, legacy placeholders). Bleaching is disabled; spike masks are diagnostic only.

- `save_qc_pdf(out_path, roi_ids, raw_traces, sliding_f0, sliding_pct, sliding_dff, roi_peak_thresholds, roi_peaks_by_roi)`  
  Generates the 3-panel PDF pages for each ROI using the already-computed arrays—no recomputation or alternative methods are performed.
