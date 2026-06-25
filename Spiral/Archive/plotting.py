# Plotting tools
#
# Cristiano Longarini
#
# Units:
# - distances in au
# - mass in msun
# - velocities in km/s
#

import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import simps
from astropy import constants as const
from scipy.interpolate import griddata
#from giggle_functions import myfunctions
import giggle_functions as giggle
from scipy import ndimage as ndimage

G = 4.30091e-3 * 206265 



#Parameters
ms = 1 #star mass
md = 0.35 #disc mass
p = -1.5 #surface density
ap = 13*np.pi/180 #pitch angle
m = 2 #azimuthal wavenumber
beta = 5 #cool
incl = np.pi/6 #inclination of the disc towards the line of sight


r = np.linspace(1,100,500)
phi = np.linspace(-np.pi,np.pi,360)
gr, gphi = np.mgrid[1:100:500j, -np.pi:np.pi:360j] #rin:rout:resolution
gx, gy = np.mgrid[-100:100:400j,-100:100:400j]
car = np.linspace(-100,100,400)
grid_angle = 0*gx
g_r = (gx**2+gy**2)**(0.5)
    
m1c = giggle.momentoneC(gx, gy, ms, md , p, m, 1, beta, 1, 100, ap, incl, 0)
m1k = giggle.momentoneC(gx, gy, ms, 0 , p, m, 1, beta, 1, 100, ap, incl, 0)
for i in range(len(car)):
    for j in range(len(car)):
        grid_angle[i,j] = math.atan2(car[i], car[j])
spir = giggle.perturbed_sigma(g_r, grid_angle, p, 1, 100, md, beta, m, ap,0)
#Spiral surface density
for i in range(len(car)):
    for j in range(len(car)):
        if(g_r[i,j] > 100):
            spir[i,j] = -np.inf


#Rotation of the disc
matrix_y_deproject = [[np.cos((incl)), 0],[0, 1]]
matrix_y_deproject = np.asarray(matrix_y_deproject)
matrix_y_deproject = np.linalg.inv(matrix_y_deproject)

m1crot = ndimage.affine_transform(m1c, matrix_y_deproject, 
                    offset=(-30,-0),order=1)
m1krot = ndimage.affine_transform(m1k, matrix_y_deproject, 
                    offset=(-30,-0),order=1)
spir_rot = ndimage.affine_transform(spir, matrix_y_deproject, 
                    offset=(-30,-0),order=1)


#plot1
fig, (ax) = plt.subplots(1,1,figsize=(8,8))
plt.imshow(m1crot, cmap='seismic', vmin = -3, vmax = 3, origin='lower')
plt.text(150+70,222,'20', size=17)
plt.text(150+95,250,'40', size=17)
plt.text(150+120,275,'60', size=17)
plt.text(150+150,302,'80', size=17)
plt.text(150+180,330,'100', size=17)
plt.text(405,195, '$0$', size=17)
plt.text(187.5,380, '$\pi/2$', size=17)
plt.text(-20,195, r'$\pi$', size=17)
plt.text(187.5,-0, r'$3\pi/2$', size=17)
plt.axis('off')
plt.colorbar(label=r'$v_{obs}$ [km/s]')
plt.savefig('p1.png', dpi=300)


#plot2
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,5))
ax1.contour(m1crot, [0], colors='mediumblue')
ax1.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax1.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax1.text(150,222,'20', size=15)
ax1.text(150-28,250,'40', size=15)
ax1.text(150-28-25,275,'60', size=15)
ax1.text(150-28-25-27,302,'80', size=15)
ax1.text(150-28-25-27-28,330,'100', size=15)
ax1.text(420,195, '$0$', size=17)
ax1.text(175,390, '$\pi/2$', size=17)
ax1.text(-30,195, r'$\pi$', size=17)
ax1.text(175,-10, r'$3\pi/2$', size=17)
ax1.set_xlim(-5,403)
ax1.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax1.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax1.text(100,-60,r'$v_{obs} = 0 $ km/s', size=22)
ax1.axis('off')
ax1.axis('equal')

ax2.contour(m1crot, [1], colors='mediumblue')
ax2.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax2.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax2.text(150,222,'20', size=15)
ax2.text(150-28,250,'40', size=15)
ax2.text(150-28-25,275,'60', size=15)
ax2.text(150-28-25-27,302,'80', size=15)
ax2.text(150-28-25-27-28,330,'100', size=15)
ax2.text(420,195, '$0$', size=17)
ax2.text(175,390, '$\pi/2$', size=17)
ax2.text(-30,195, r'$\pi$', size=17)
ax2.text(175,-10, r'$3\pi/2$', size=17)
ax2.set_xlim(-5,403)
ax2.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax2.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax2.text(100,-60,r'$v_{obs} = 1 $ km/s', size=22)
ax2.axis('off')
ax2.axis('equal')

ax3.contour(m1crot, [2], colors='mediumblue')
ax3.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax3.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax3.text(150,222,'20', size=15)
ax3.text(150-28,250,'40', size=15)
ax3.text(150-28-25,275,'60', size=15)
ax3.text(150-28-25-27,302,'80', size=15)
ax3.text(150-28-25-27-28,330,'100', size=15)
ax3.text(420,195, '$0$', size=17)
ax3.text(175,390, '$\pi/2$', size=17)
ax3.text(-30,195, r'$\pi$', size=17)
ax3.text(175,-10, r'$3\pi/2$', size=17)
ax3.set_xlim(-5,403)
ax3.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax3.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax3.text(100,-60,r'$v_{obs} = 2 $ km/s', size=22)
ax3.axis('off')
ax3.axis('equal')

ax4.contour(m1crot, [3], colors='mediumblue')
ax4.plot(200*np.cos(phi) + 200, 172*np.sin(phi)+ 198, c='black', lw=1)
ax4.plot(200/5*np.cos(phi) + 200, 172/5*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 2 *np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 2*np.cos(phi) + 200, 172/5 * 2*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 3*np.cos(phi) + 200, 172/5 * 3*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.plot(200/5 * 4*np.cos(phi) + 200, 172/5 * 4*np.sin(phi)+ 198, c='black', lw=0.25)
ax4.text(150,222,'20', size=15)
ax4.text(150-28,250,'40', size=15)
ax4.text(150-28-25,275,'60', size=15)
ax4.text(150-28-25-27,302,'80', size=15)
ax4.text(150-28-25-27-28,330,'100', size=15)
ax4.text(420,195, '$0$', size=17)
ax4.text(175,390, '$\pi/2$', size=17)
ax4.text(-30,195, r'$\pi$', size=17)
ax4.text(175,-10, r'$3\pi/2$', size=17)
ax4.set_xlim(-5,403)
ax4.plot(np.linspace(0,1,10) * 0 + 200, np.linspace(0,1,10)*344 + 26, lw=0.5, c='black')
ax4.plot(np.linspace(0,1,10)*400 + 1, np.linspace(0,1,10) * 0 + 200, lw=0.5, c='black')
ax4.text(100,-60,r'$v_{obs} = 3 $ km/s', size=22)
ax4.axis('off')
ax4.axis('equal')
plt.savefig('p2.png', dpi=300)


#plot3
fig, (ax) = plt.subplots(1,1,figsize=(8,8))

plt.imshow(np.log(spir_rot), cmap='jet', vmin = -12.5, vmax = -8., origin='lower')
plt.text(150+70,222,'20', size=17)
plt.text(150+95,250,'40', size=17)
plt.text(150+120,275,'60', size=17)
plt.text(150+150,302,'80', size=17)
plt.text(150+180,330,'100', size=17)
plt.text(405,195, '$0$', size=17)
plt.text(187.5,380, '$\pi/2$', size=17)
plt.text(-20,195, r'$\pi$', size=17)
plt.text(187.5,-0, r'$3\pi/2$', size=17)
plt.axis('off')
plt.colorbar(label=r'$v_{obs}$ [km/s]')
plt.savefig('p3.png', dpi=300)



#plot4
fig, (ax) = plt.subplots(1,1,figsize=(8,8))

plt.imshow(-m1crot +m1krot, cmap='seismic', vmin = -0.8, vmax = 0.8, origin='lower')
plt.colorbar(label=r'$v_{obs}$ [km/s]')

ax.text(150+70,222,'20', size=17)
ax.text(150+95,250,'40', size=17)
ax.text(150+120,275,'60', size=17)
ax.text(150+150,302,'80', size=17)
ax.text(150+180,330,'100', size=17)

ax.text(405,195, '$0$', size=17)
ax.text(187.5,390, '$\pi/2$', size=17)
ax.text(-20,195, r'$\pi$', size=17)
ax.text(187.5,-10, r'$3\pi/2$', size=17)
ax.set_xlim(-5,403)
ax.axis('off')
plt.savefig('p4.png', dpi = 300)
