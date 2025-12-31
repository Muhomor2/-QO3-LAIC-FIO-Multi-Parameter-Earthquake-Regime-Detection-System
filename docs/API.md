# QO3-LAIC-FIO API Reference

## Command Line Interface

```bash
python src/qo3_fio_ultimate.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--region` | str | global | Single region name |
| `--regions` | str | | Multiple regions (comma-separated) |
| `--target_mag` | float | varies | Target magnitude |
| `--horizon` | int | 7 | Forecast horizon (days) |
| `--days` | int | 365 | History days to load |
| `--custom_dir` | str | custom_data | Custom data directory |
| `--telegram` | flag | | Enable Telegram notifications |
| `--daily_report` | flag | | Send daily summary |
| `--output` | str | README.md | Markdown output file |
| `--latex_output` | str | results.tex | LaTeX output file |
| `--list_regions` | flag | | List available regions |

### Examples

```bash
# Basic usage
python src/qo3_fio_ultimate.py --region japan

# Custom target
python src/qo3_fio_ultimate.py --region tohoku --target_mag 4.5 --horizon 7

# Multiple regions
python src/qo3_fio_ultimate.py --regions global,japan,cascadia

# With custom data
python src/qo3_fio_ultimate.py --region california --custom_dir my_data/

# Full monitoring
python src/qo3_fio_ultimate.py --regions global,japan,israel \
    --telegram --daily_report
```

## Python API

### FIOEstimators

```python
from qo3_fio_ultimate import FIOEstimators

fio = FIOEstimators()

# b-value
b = fio.b_value_aki_utsu(magnitudes, mc=2.5)

# CV
cv = fio.cv_interevent(timestamps_ns)

# Entropy
H = fio.shannon_entropy(magnitudes)

# SID
sid = fio.seismic_information_deficit(H, H_background)
```

### QO3FeatureBuilder

```python
from qo3_fio_ultimate import QO3FeatureBuilder, Config

config = Config(
    region_name="Japan",
    lat_min=24, lat_max=46,
    lon_min=122, lon_max=150,
    magnitude_completeness=2.5,
    target_magnitude=5.0,
    forecast_horizon=7
)

builder = QO3FeatureBuilder(config)
features = builder.build_matrix(events, kp_data=kp, solar_data=solar)
```

### EventDeduplicator

```python
from qo3_fio_ultimate import EventDeduplicator

dedup = EventDeduplicator(time_sec=30, dist_km=30, mag_diff=0.3)
clean_events = dedup.deduplicate(merged_catalog)
```

### ModelEvaluator

```python
from qo3_fio_ultimate import ModelEvaluator

evaluator = ModelEvaluator()

# PR-AUC
pr_auc = evaluator.pr_auc(y_true, y_score)

# Bootstrap CI
ci_lo, ci_hi = evaluator.bootstrap_ci(y_true, y_score, evaluator.pr_auc)

# Skill score
skill = evaluator.skill_score(pr_auc, baseline_rate)
```

### LaTeXGenerator

```python
from qo3_fio_ultimate import LaTeXGenerator

latex = LaTeXGenerator()

# Results table
table = latex.results_table(results)

# Setup table
setup = latex.experimental_setup_table(config, "USGS", "2020-2023", "2024")
```

## Output Formats

### Markdown (README.md)
Human-readable forecast summary with emoji risk levels.

### JSON (forecast.json)
Machine-readable format for integration:
```json
{
  "region": "Japan",
  "date": "2025-12-31",
  "probability": 72.3,
  "risk_level": "HIGH",
  "target_magnitude": 5.0,
  "horizon_days": 7,
  "indicators": {
    "b_value": 0.78,
    "cv": 1.12,
    "sid": 0.35,
    "laic_index": 0.75
  }
}
```

### LaTeX (results.tex)
Publication-ready tables with bootstrap CI.
