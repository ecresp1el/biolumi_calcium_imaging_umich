# Reproducible napari environment (macOS)

This repository provides a **deterministic, macOS-safe napari environment**
tested on both **Apple silicon (M1/M2/M3)** and **Intel Macs**.

---

## System requirements

- macOS (Intel or Apple silicon)
- Conda distribution:
  - Miniconda (recommended)
  - Miniforge (recommended for labs)
  - Anaconda (works, but larger)

> ⚠️ Windows and Linux require separate environment files.

---

## Installation (fresh environment)

From the directory containing `napari-macos.yml`:

```bash
conda env create -f napari-macos.yml