# QO3-LAIC-FIO: Multi-Parameter Earthquake Regime Detection System

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

QO3-LAIC-FIO is a comprehensive earthquake regime detection system that integrates:

- **FIO (Fractal Information Ontology)**: b-value, CV, entropy, SID indicators
- **LAIC Model**: Lithosphere-Atmosphere-Ionosphere Coupling index
- **Custom Precursor Data**: Radon, GNSS, groundwater, OLR, and any CSV time series
- **Honest Evaluation**: Leak-free features, bootstrap CI, publication-ready LaTeX tables

## Citation

If you use this software in your research, please cite:

```bibtex
@software{chechelnitsky_qo3_2025,
  author       = {Chechelnitsky, Igor},
  title        = {{QO3-LAIC-FIO: Multi-Parameter Earthquake Regime Detection System}},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## Installation

```bash
git clone https://github.com/ichechelnitsky/qo3-laic-fio.git
cd qo3-laic-fio
pip install -r requirements.txt
```

## Quick Start

```bash
# Basic usage
python src/qo3_fio_ultimate.py --region japan

# Multiple regions with evaluation
python src/qo3_fio_ultimate.py --regions global,japan,cascadia,israel

# With custom precursor data
python src/qo3_fio_ultimate.py --region california --custom_dir data/examples

# With Telegram notifications
python src/qo3_fio_ultimate.py --region tohoku --telegram --daily_report
```

## Features

### FIO Indicators

| Parameter | Formula | Physical Meaning | Precursor Signal |
|-----------|---------|------------------|------------------|
| **b-value** | log₁₀(e)/(M̄-Mc) | Stress distribution | b < 0.9 |
| **CV** | σ(ΔT)/μ(ΔT) | Temporal clustering | CV → 1 |
| **Entropy** | -Σ pᵢ log₂ pᵢ | Information content | H ↓ |
| **SID** | 1 - H/H_bg | Information deficit | SID ↑ |

### LAIC Coupling Index

Integrates multi-layer anomalies:
- Seismic stress (b < 0.9)
- Geomagnetic storm (Kp ≥ 5)
- Ionospheric TEC anomaly
- Solar activity (F10.7 > 150)
- Custom precursors (radon, GNSS, etc.)

### Custom Data Support

Place CSV files in `data/examples/` or `--custom_dir`:

| Type | Pattern | Format |
|------|---------|--------|
| Radon | `radon_*.csv` | date,value,station |
| Groundwater | `groundwater_*.csv` | date,level,temperature,station |
| GNSS | `gnss_*.csv` | date,east,north,up,station |
| Temperature | `lst_*.csv` | date,value,lat,lon |
| OLR | `olr_*.csv` | date,value,lat,lon |
| Custom | `custom_*.csv` | date,value,... |

## Scientific Foundation

### Physical Basis

The system is grounded in the **Cascadia-Tohoku analogy** for fault mechanics:

- **b-value decrease** → Fault locking due to fluid expulsion
- **CV convergence to 1** → Transition from stochastic to deterministic behavior
- **SID increase** → Seismic quiescence (information compression)

### Mathematical Foundation

The FIO framework is supported by:
- Extension of Sebestyen's theorem to unbounded operators
- Proper handling of heavy-tailed distributions
- Information-theoretic formulation of regime detection

### Key References

1. Aki, K. (1965). Maximum likelihood estimate of b. Bull. Earthq. Res. Inst., 43, 237-239.
2. Utsu, T. (1965). A method for determining b value. Geophys. Bull. Hokkaido Univ., 13, 99-103.
3. Pulinets, S., & Ouzounov, D. (2011). LAIC model. Nat. Hazards Earth Syst. Sci., 11, 3247-3256.

## Output

### Markdown Report
```
| Region | Risk | Probability | b-value | CV | SID | LAIC |
|--------|------|-------------|---------|-----|-----|------|
| Japan | 🔴 HIGH | 72% | 0.78 | 1.12 | 0.35 | 0.75 |
```

### LaTeX Tables (Publication-Ready)
```latex
\begin{table}[t]
\caption{Model comparison: PR-AUC with 95\% bootstrap CI.}
\begin{tabular}{lcccc}
\toprule
Model & PR-AUC & 95\% CI & Skill \\
\midrule
Baseline (rate-only) & 0.573 & [0.528, 0.618] & 0.260 \\
QO3 Full (FIO + LAIC) & 0.582 & [0.538, 0.628] & 0.275 \\
\bottomrule
\end{tabular}
\end{table}
```

### JSON Forecast
```json
{
  "region": "Japan",
  "probability": 72.3,
  "risk_level": "HIGH",
  "indicators": {
    "b_value": 0.78,
    "cv": 1.12,
    "sid": 0.35,
    "laic_index": 0.75
  }
}
```

## Available Regions

| Region | Mc | Target M* |
|--------|-----|----------|
| global | 4.5 | 6.0 |
| japan | 2.5 | 5.0 |
| tohoku | 2.0 | 4.5 |
| california | 2.5 | 4.5 |
| cascadia | 2.5 | 5.0 |
| turkey | 3.0 | 4.5 |
| israel | 2.0 | 4.0 |
| indonesia | 4.0 | 5.5 |
| chile | 4.0 | 5.5 |
| mediterranean | 3.0 | 4.5 |
| iran | 3.5 | 5.0 |
| himalaya | 4.0 | 5.5 |
| alaska | 3.5 | 5.0 |
| newzealand | 3.0 | 5.0 |

## Data Sources

### Built-in (Automatic)
- **USGS**: Real-time earthquake catalog (<5 min latency)
- **NOAA SWPC**: Kp index (<1 min latency)
- **NOAA**: Solar flux F10.7 (<1 day latency)

### Custom (User-Provided)
- Radon monitoring stations
- GNSS networks (Nevada Geodetic Lab, UNAVCO)
- Groundwater wells (USGS)
- Satellite thermal data (MODIS, Landsat)
- OLR data (NOAA NCEP)

## License

CC-BY-NC-ND-4.0  License. See [LICENSE](LICENSE) for details.

## Author

**Igor Chechelnitsky**
- ORCID: [0009-0007-4607-1946](https://orcid.org/0009-0007-4607-1946)
- Location: Ashkelon, Israel

## Acknowledgments

This work builds upon foundational research in statistical seismology and
the LAIC model. Special thanks to the seismological community for open
data access through USGS, NOAA, and other agencies.
