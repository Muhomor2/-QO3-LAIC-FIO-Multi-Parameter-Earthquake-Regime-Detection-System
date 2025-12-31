# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-31

### Added
- Complete integration of FIO (Fractal Information Ontology) indicators
- LAIC (Lithosphere-Atmosphere-Ionosphere Coupling) index
- Event deduplication for multi-catalog support
- Leak-free feature engineering with proper target construction
- Bootstrap confidence intervals for PR-AUC
- LaTeX table generation for publications
- Custom precursor data support (radon, GNSS, groundwater, OLR)
- Telegram notification system
- Zenodo-ready package structure

### Changed
- Refactored codebase into unified system
- Improved b-value estimation with Shi-Bolt bias correction
- Enhanced CV calculation with proper timestamp handling

### Fixed
- Target leakage issue (shift(-1) before rolling)
- Timestamp comparison bugs
- NaN handling in feature computation

## [1.0.0] - 2025-12-01

### Added
- Initial release with basic QO3 functionality
- USGS data loader
- Basic b-value and CV calculations
