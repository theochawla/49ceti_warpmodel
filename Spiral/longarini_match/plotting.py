import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import bettermoments as bm
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

m1_elias = fits.open('longarini_2021_M1.fits')
m1_elias_data = m1_elias["PRIMARY"].data

fig, ax = plt.subplots()
#plt.imshow(inc23nowarpm0_data, vmin=0, vmax=2, extent=[0,25.6,0,25.6])
plt.imshow(m1_elias_data/1000, origin="lower", cmap="seismic", vmin=-3,vmax=3)
#plt.xlim(150,350)
#plt.ylim(150,350)
plt.colorbar(label="km/s")
plt.savefig("longarini_copy.png")
plt.show()
