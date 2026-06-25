import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const

import matplotlib.gridspec as gridspec

import cmocean

import cmcrameri.cm as cmc
#color_map = cmc.vik

plt.rcParams["font.size"] = 18

def incline(y, z, inc):
    inc_ = np.deg2rad(inc)

    cosi = np.cos(inc_)
    sini = np.sin(inc_)

    y_f =  y*cosi - z*sini
    z_f =  y*sini + z*cosi
    return y_f, z_f
    
def matrix_mine(x, y, z, warp, twist, inc_, PA_):
    #print(warp)
    warp = warp[:, None, None]
    twist = twist[:, None, None]

    cosPA = np.cos(PA_)
    sinPA = np.sin(PA_)
    
    cosi = np.cos(inc_)
    sini = np.sin(inc_)

    '''making sure twist occurs before warp'''

    cosw = np.cos(warp)
    sinw = np.sin(warp)

    cost = np.cos(twist)
    sint = np.sin(twist)
    
    y_w = y*cosw - z*sinw
    z_w = y*sinw + z*cosw
    
    x_t = x*cost - y_w*sint
    y_t = x*sint + y_w*cost

    x_pa = x_t*cosPA - y_t*sinPA
    y_pa = x_t*sinPA + y_t*cosPA

    #self.z_w_max = np.max(z_w)
    #self.x_pa = x_pa
    #self.y_pa = y_pa
    #self.z_w = z_w

    #self.a_w, self.p_w = cart2pol(x_pa, y_pa)
    #self.zf_w = np.linspace(-z_w_max, z_w_max, self.nzc)
    #self.pf_w = np.linspace(-z_w_max, z_w_max, self.nzc)
    
    #y_f =  y_pa*cosi - z_w*sini
    #z_f =  y_pa*sini + z_w*cosi


    return x_pa, y_pa, z_w


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def w_func(a, r0, dr, r, type):
    #r0 = self.w_r0
    #dr = self.w_dr

    '''same general function for warp & twist, just need to specify which param to use'''
    #if type == "w":
        #a = self.w_i

    #elif type == "pa":
        #a = self.pa
    
    
    r0 = 1.0 if r0 is None else r0
    dr = 1.0 if dr is None else dr
    return np.radians(a / (1.0 + np.exp(-(r0 - r) / (0.1*dr)))) 

def w_func_global(value, glob, r0, dr, r, type):
    #a = value + glob
    #r0 = self.w_r0
    #dr = self.w_dr

    '''same general function for warp & twist, just need to specify which param to use'''
    #if type == "w":
        #a = self.w_i

    #elif type == "pa":
        #a = self.pa
    
    
    r0 = 1.0 if r0 is None else r0
    dr = 1.0 if dr is None else dr

    list = np.radians(glob - value / (1.0 + np.exp(-(r0 - r) / (0.1*dr))))
    #print(list)
    return np.radians(glob - value / (1.0 + np.exp(-(r0 - r) / (0.1*dr)))) 
    a = value + glob
    #r0 = self.w_r0
    #dr = self.w_dr

    '''same general function for warp & twist, just need to specify which param to use'''
    #if type == "w":
        #a = self.w_i

    #elif type == "pa":
        #a = self.pa
    
    
    r0 = 1.0 if r0 is None else r0
    dr = 1.0 if dr is None else dr
    return np.radians(a / (1.0 + np.exp(-(r0 - r) / (0.1*dr)))) 


af = np.linspace(20,300,100)
zf = np.linspace(-30,30,20)
#zf_warp = np.linspace(zmin+1,self.zmax-1,nzc-2)
pf = np.linspace(0,2*np.pi,400)

w_i = 15
w_pa = 0

inc = 80
pa = 180

r0 = 100
dr = 200

warp_i  = w_func(w_i, r0, dr, af, type="w")

twist_i = w_func(w_pa,r0, dr, af, type="pa")


pcf,acf,zcf = np.meshgrid(pf,af,zf)
xi, yi = pol2cart(acf, pcf)

x_w, y_w, z_w = matrix_mine(xi, yi, zcf, warp_i, twist_i, np.deg2rad(inc), np.deg2rad(pa))

r_w, p_w = cart2pol(x_w, y_w)

ty, tz = incline(yi, zcf, inc)

ty_w, tz_w = incline(y_w, z_w, inc)


G = 1.3271244002e11 * u.km**3 / (u.solMass * u.second**2)
star_mass = 2

vel_kep = np.sqrt(G.value*star_mass/(acf*u.au.to(u.km)))*(np.cos(pcf))*np.sin(np.deg2rad(inc))




#wi_list = [5, 10, 15]
pa_list = [30, 60, 90, 120, 150, 180]

#ty, tz = incline(yi, zcf, inc)
fig = plt.figure(figsize=(15,10), constrained_layout=True)
fig.suptitle("Warp Velocity Residuals", fontsize=26, y=1.05)
gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

#for i in range(len(wi_list)):
for j in range(len(pa_list)):
    warp_i  = w_func(w_i, r0, dr, af, type="w")
    twist_i = w_func(w_pa,r0, dr, af, type="pa")

    x_w, y_w, z_w = matrix_mine(xi, yi, zcf, warp_i, twist_i, np.deg2rad(inc), np.deg2rad(pa_list[j]))
    ty_w, tz_w = incline(y_w, z_w, inc)

    warp_global_incl = w_func_global(w_i, inc, r0, dr, af, type="w")
    vel_w = np.sqrt(G.value*star_mass/(r_w*u.au.to(u.km)))*(np.cos(p_w))*np.sin(warp_global_incl[:,None,None])

    vel_resid = vel_w + vel_kep

    

    if j < 3:
        ax = fig.add_subplot(gs[0, j], label=f"ax{j}")
        ax.set_xticks([])
    else:
        ax = fig.add_subplot(gs[1, j-3], label=f"ax{j}")
        ax.set_xlabel("X (au)")

    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    pcm = ax.pcolor(x_w[:,:,10], ty_w[:,:,10], vel_resid[:,:,10], cmap=cmocean.cm.balance)

    if j == 2 or j == 5:
        fig.colorbar(pcm, ax=ax, label="Velocity (km/s)", fraction=0.046, pad=0.04, shrink=0.8)

    if j == 0 or j == 3:
        ax.set_ylabel("Y (au)")
    else:
        ax.set_yticks([])

    ax.set_title(r"$\gamma=${}$^\circ$".format(pa_list[j]))
    ax.set_ylim(-200, 200)
    #plt.tight_layout()

plt.savefig("warp_vel_resids.png", pad_inches=0.5, bbox_inches='tight')
plt.show()