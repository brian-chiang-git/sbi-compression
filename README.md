
# Repository Structure

This project is organized as a Python package that can be imported directly and also used through a command-line interface.

## Proposed Structure

```
sbi_compression/
    src/
        sbi_compression/
            __init__.py
            api.py
            cli.py
            methods/
                base.py                  # compressor api
                utils.py                 # utilities e.g. find data dimensions
                linear/
                    pca.py               # JAX PCA implementation
                    cca.py               # JAX CCA implementation
                    moped.py             # JAX MOPED implementation
                ml/
                    autoencoder.py       # Equinox autoencoder implementation   
                    flow_compressor.py   # Equinox ?
            data/
                loaders.py
                transforms.py
            eval/
                metrics.py
                validation.py
            io/
                save_load.py
            config/
                schema.py
    tests/
        test_api.py
        test_cli.py
        methods/
            linear/
            ml/
    examples/
        quickstart.py
    docs/
    README.md
    pyproject.toml
```

## How It Maps To The Project

### Importable package

Users should be able to import `sbi_compression` and call the high-level API from `api.py`.

### CLI support

`cli.py` should expose commands such as `fit`, `transform`, `evaluate`, and batch-style workflows.

### Mixed methods

`methods/base.py` defines a shared compressor interface, while `methods/linear/` and `methods/ml/` contain the concrete implementations.

### Reproducibility

`config/schema.py` and `io/save_load.py` should handle saved model metadata, configuration, and reloadable artifacts.

## Core Interface

Each compressor should expose the same basic operations:

```python
fit(X, y=None)
transform(X)
inverse_transform(Z)  # optional
save(path)
load(path)
```

Both the Python API and the CLI should use these same methods.

## Packaging Notes

In `pyproject.toml`, keep:

- a `build-system` section
- project metadata
- optional dependency groups for `ml`, `dev`, and `docs`
- a `project.scripts` entry for the CLI, for example `sbi-compress`

## Public API

In `__init__.py`, expose only the stable public surface:

- high-level classes and functions from `api.py`
- a version string

Avoid exporting every internal implementation detail.

## Testing Strategy

- unit tests for each method implementation
- contract tests to verify every compressor follows the same interface
- CLI integration tests for command behavior and file output
- small synthetic cosmology fixtures for deterministic tests

## Suggested Next Steps

1. Create the `src/sbi_compression` package skeleton around the existing `__init__.py`.
2. Add a minimal `api.py` and `cli.py` wired through `project.scripts` in `pyproject.toml`.
3. Add one linear method and one ML method under a shared base interface.
4. Add `test_api.py` and `test_cli.py` first, then extend method-specific tests.