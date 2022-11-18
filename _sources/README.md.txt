# Docs builds

The docs are built with Sphinx. To build a local copy, run ``make html``.

Otherwise, the readthedocs servers will build the docs. CASA imports are mocked with Sphinx ``autodoc_mock_imports`` because the CASA install requires ``libgfortran`` to be installed and apparently that's difficult with RTD.