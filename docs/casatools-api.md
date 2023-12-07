(casatools-api-label)=

# Casatools API

Over the past few ALMA cycles, the documentation for the [casatools](https://casadocs.readthedocs.io/en/stable/api/casatools.html) has improved tremendously, obviating the need for many of the original *visread* routines. Therefore, we have removed that duplicate functionality from the visread package. Instead, we highlight several of the casatools that we use on a regular basis. Following the tutorials, we recommend that you familiarize yourself with these tools and then use *visread* conversion routines as needed. Be aware that there are many [casatools](https://casadocs.readthedocs.io/en/stable/api/casatools.html) available beyond those listed here, too.

## msmetadata

As its name implies, the [msmetadata](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.msmetadata.html#casatools.msmetadata) tool is a very useful tool for getting metadata-like quantities from a measument set. We have found it very useful for obtaining things like number and characteristics of spectral windows (`datadescids`), channel frequencies, times of observations, antenna baselines, and more.

For example, to get an array of the channel frequencies (in Hz) from a measurement set, follow

```
import casatools
msmd = casatools.msmetadata()
msmd.open("my.ms")
swp_id = 0
chanfreqs = msmd.chanfreqs(spw_id)
msmd.done()
```

## ms

In our opinion, the [ms](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html) tool is the workhorse tool for obtaining data from the measurement set. We commonly use something like the following workflow

```
import casatools
ms = casatools.ms()
ms.open("my.ms")
swp_id = 0
# narrow down the selection to just the spw under consideration
ms.selectinit(spw_id)
# further narrow the selection to the shortest baselines
ms.select({"uvdist":[0.0, 100.0]}) # [m]

# get channel info, baselines, visibility data, etc in a dictionary
d = ms.getdata(['axis_info', 'data', 'uvw'])
print(d['uvw'].shape)

ms.reset() # clear selections
ms.done()
```

Making the appropriate selection before querying your data can really reduce the memory requirements and runtime of any operation. This can really make a difference when working with large measurement sets.

If your spectral windows are different shapes, you will find the `ms.selectinit` operation to be necessary to read one spectral window at a time.

## table

Compared to `ms`, the [table](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.table.html) tool is a more direct way to access the sub-tables within a measurement set. If you are working with a measurement set that only contains a single spectral window, you may find these routines useful. In practice, though, we tend to prefer a combination of the `msmetadata` and `ms` tools because of their ability to handle measurement sets with heterogeneous spectral windows.
