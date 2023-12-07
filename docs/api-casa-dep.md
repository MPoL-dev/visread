# Visread + casatools API

Visread tools with a `casatools` dependency.

Before getting started with the *visread* package, we recommend becoming familiar with the [casatools](https://casadocs.readthedocs.io/en/stable/api/casatools.html) provided as part of CASA. You can see some of our highlights in the tutorials and in {ref}`casatools-api-label`.

In this context, the *visread* routines are designed to perform additional processing to the products extracted from measurement sets using casatools.

The philosophy of this packaged is to augment the casatools functionality, and is meant to be used in coordination with it. The reason is that every measurement set seems to be slightly different (depending on the type of observation, how it was calibrated, and whether the user performed any additional processing) and therefore it's difficult to create many general purpose, fully abstracted routines.

## Process CASA

```{eval-rst}
.. automodule:: visread.process_casa
```

## Scatter

```{eval-rst}
.. automodule:: visread.scatter_casa
```

## Visualization

```{eval-rst}
.. automodule:: visread.visualization_casa
```
