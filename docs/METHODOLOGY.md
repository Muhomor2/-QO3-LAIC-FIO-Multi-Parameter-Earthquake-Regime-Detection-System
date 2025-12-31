# QO3-LAIC-FIO Methodology

## 1. Theoretical Foundation

### 1.1 Fractal Information Ontology (FIO)

The FIO framework treats earthquake sequences as information-processing systems
where regime transitions manifest as changes in statistical descriptors.

#### b-value Estimation

The Gutenberg-Richter law states:
```
log₁₀(N) = a - bM
```

We use the Aki-Utsu maximum likelihood estimator with Shi-Bolt bias correction:
```
b = log₁₀(e) / (M̄ - (Mc - 0.05))
```

Physical interpretation:
- b ≈ 1.0: Normal tectonic state
- b < 0.9: Stress accumulation (precursor signal)
- b > 1.2: Stress release (aftershock regime)

#### Coefficient of Variation (CV)

The CV of inter-event times measures temporal clustering:
```
CV = σ(ΔT) / μ(ΔT)
```

Physical interpretation:
- CV > 1: Clustered (aftershocks, fluid-rich)
- CV = 1: Poisson (random baseline)
- CV < 1: Quasi-periodic (locked fault)

The transition CV → 1 from above indicates system rigidification.

#### Seismic Information Deficit (SID)

The SID quantifies "seismic quiescence" through entropy reduction:
```
SID = 1 - H(t) / H_background
```

where H is the Shannon entropy of the magnitude distribution.

### 1.2 LAIC Coupling Model

The Lithosphere-Atmosphere-Ionosphere Coupling (LAIC) model posits that:

1. Tectonic stress accumulation → Rock micro-fracturing
2. Micro-fracturing → Radon gas release
3. Radon → Air ionization
4. Ionization → Electric field perturbations
5. Electric fields → Ionospheric TEC anomalies

The LAIC index integrates multiple layers:
```
LAIC = (seismic_stress + geomag_storm + TEC_anomaly + solar_anomaly) / 4
```

### 1.3 Cascadia-Tohoku Analogy

Both subduction zones share:
- Oceanic-continental convergence
- M9+ potential
- Multi-century recurrence intervals
- Evidence of fault locking

The physical mechanism for b-value decrease:
- Pore fluid expulsion from fault zone
- Increased effective friction
- Suppression of small events
- Stress concentration

## 2. Data Processing

### 2.1 Event Deduplication

Multi-catalog merging requires deduplication to avoid:
- Inflated event counts
- Corrupted b-value estimates
- Biased CV calculations

Deduplication rule: Collapse events within
- Time: ±30 seconds
- Distance: ±30 km
- Magnitude: ±0.3

### 2.2 Leak-Free Target Construction

The target variable must avoid look-ahead bias:
```python
future_max = max_mag.shift(-1).rolling(horizon).max()
target = (future_max >= target_magnitude).astype(int)
```

The `shift(-1)` ensures today's events are not included in the target.

### 2.3 Feature Engineering

All features are computed using only past data:
- Rolling windows look backward
- No future information leakage
- Proper train/test temporal split

## 3. Evaluation Protocol

### 3.1 Primary Metric: PR-AUC

For imbalanced classification (rare large events), PR-AUC is preferred over ROC-AUC.

### 3.2 Uncertainty Quantification

95% bootstrap confidence intervals (1000 resamples):
```python
for _ in range(1000):
    idx = rng.choice(n, size=n, replace=True)
    values.append(metric(y_true[idx], y_score[idx]))
ci = [quantile(values, 0.025), quantile(values, 0.975)]
```

### 3.3 Skill Score

Normalized improvement over baseline:
```
Skill = (PR-AUC - baseline_rate) / (1 - baseline_rate)
```

## 4. Limitations

### 4.1 What This System Does
- Detects statistical regime changes
- Ranks relative risk periods
- Integrates multiple precursor types

### 4.2 What This System Does NOT Do
- Predict specific earthquakes (time, location, magnitude)
- Provide deterministic forecasts
- Guarantee precursor detection

### 4.3 Epistemic Uncertainty
- Physical mechanisms are plausible but not proven
- Statistical associations may not be causal
- Performance varies by region and target magnitude

## References

1. Aki, K. (1965). Maximum likelihood estimate of b. Bull. Earthq. Res. Inst.
2. Utsu, T. (1965). A method for determining b value. Geophys. Bull. Hokkaido Univ.
3. Shi, Y., & Bolt, B. A. (1982). The standard error of the magnitude-frequency b value. BSSA.
4. Pulinets, S., & Ouzounov, D. (2011). LAIC model. Nat. Hazards Earth Syst. Sci.
5. Heki, K. (2011). Ionospheric electron enhancement preceding the 2011 Tohoku earthquake. GRL.
