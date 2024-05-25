# Model Training for Autosink Project

[🇬🇧](README.md) | [🇰🇷](README.kr.md) | [🇨🇳](README.zh-CN.md)

The environment is based on MacOS and Linux.

## `Makefile`

The `Makefile` has the following functionalities.

### `make lint`

- To use the `.vscode` settings, install the `pylint` extension.
- Overrides options specified in the `pyproject.toml` file in the linter's default settings to lint the code.

### `make format`

- The formatter uses Google's `yapf`.
- Overrides options specified in the `pyproject.toml` file in the default settings of the `yapf` formatter to format the code.
- To use the `.vscode` settings, install the `yapf` extension.

### `make test`

- Tests use `unittest`.
- Supports both `test_*.py` and `*_test.py` patterns.
- The test files must be connected to `__init__.py` up to the location where the test files exist.

### `make publish`

- Write the `~/.pypirc` file as follows.
    ```
    [pypi]
    username = __token__
    password = pypi-something # Obtain and write your personal API token.
    ```
- Running this command will push the package to the PyPI public registry using `flit`.
- The package uploaded under the previously specified name `myproject` (alias) will be available for anyone worldwide to install and use with `python3 -m pip install myproject`.