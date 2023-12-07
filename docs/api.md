# Visread API

Before getting started with the *visread* package, we recommend becoming familiar with the [casatools](https://casadocs.readthedocs.io/en/stable/api/casatools.html) provided as part of CASA. You can see some of our highlights in the tutorials and in {ref}`casatools-api-label`.

In this context, the *visread* routines are designed to perform additional processing to the products extracted from measurement sets using casatools.

The philosophy of this packaged is to augment the casatools functionality, and is meant to be used in coordination with it. The reason is that every measurement set seems to be slightly different (depending on the type of observation, how it was calibrated, and whether the user performed any additional processing) and therefore it's difficult to create many general purpose, fully abstracted routines.

## Process tools

The routines in the `process` module are generally designed to process quantities extracted using casatools into alternative formats for plotting and analysis.

```{eval-rst}
.. automodule:: visread.process
```

## Utils

```{eval-rst}
.. automodule:: visread.utils
```

## Scatter

The routines in the `utils` module are designed to calculate the scatter of the visibility residuals.

```{eval-rst}
.. automodule:: visread.scatter
```

## Visualization

The routines in the `visualization` module are designed to plot various diagnostic plots for the visibility residuals.

```{eval-rst}
.. automodule:: visread.visualization
```
