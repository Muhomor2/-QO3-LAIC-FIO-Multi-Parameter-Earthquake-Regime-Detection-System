# Contributing to QO3-LAIC-FIO

Thank you for your interest in contributing to QO3-LAIC-FIO!

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Include detailed description and steps to reproduce

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`python -m pytest tests/`)
5. Commit with clear messages
6. Push and create a Pull Request

### Adding New Precursor Types
To add support for new precursor data types:
1. Add pattern to `CustomDataLoader.DATA_TYPES`
2. Update `CUSTOM_DATA_GUIDE.md`
3. Add example CSV to `data/examples/`

### Scientific Contributions
- Literature references should be added to `.zenodo.json`
- Physical interpretations should cite primary sources
- New indicators require validation documentation

## Code Style
- Follow PEP 8 guidelines
- Add type hints for function signatures
- Document functions with docstrings

## Contact
- Author: Igor Chechelnitsky (ORCID: 0009-0007-4607-1946)
