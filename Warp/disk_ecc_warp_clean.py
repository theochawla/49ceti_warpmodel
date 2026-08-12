import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from astropy import constants as const
from scipy.special import ellipk,ellipe
from scipy.integrate import trapz

from scipy.interpolate import LinearNDInterpolator as interpnd

import time

'''method from Andres Zuleta et al. 2024
paper: ui.adsabs.harvard.edu/abs/2024A%26A...692A..56Z/abstract
github repo: https://github.com/andres-zuleta/eddy/tree/warp_rf'''

'''The general method:
In set-structure:
-create 3D position cubes in cylindrical coordinates
-Calculate density, temp, and velocity
-Create cartesian 3D position grids
-Apply 3D warp (rotation matrix in cartesian coordinates)
-Translate back to cylindrical coordinates
-Interpolate density, temp, and velocity back on to warped cylindrical grid


In set rt grid:
-define r & phi grids for sky (image) plane
-translate to cartesian coordinates
-using this structure (because it has the same dimensions as the original disk)
apply a warp with a rotational matrix
-rotate disk into sky plane using tdisky and tdiskz
-interpolate grids from set-structure onto tdisky & tdiskz grids
'''



'''
Defining some helpful functions for applying warp & coordinate switching
straightforward coordinate switching but i did steal it from here;
https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates'''

def plot(x, y, z, title, label=None, ylim=None):
    plt.pcolor(x, y, z, cmap="RdBu_r")
    plt.title(title)
    plt.colorbar(label=label)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

def imshow(x, title, label=None):
    plt.imshow(x, cmap="RdBu_r")
    plt.title(title)
    plt.colorbar(label=label)
    plt.show()

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


'''
This function applies warp and twist gradually over a range of annuli dr, centered at annulus r0
'''
def w_func(self, r, type):
    r0 = self.w_r0
    dr = self.w_dr

    '''same general function for warp & twist, just need to specify which param to use'''
    if type == "w":
        a = self.w_i

    elif type == "pa":
        a = self.pa
    '''
    print("a " + str(a))
    print("r0 " + str(r0))
    print("dr " + str(dr))
    print("r max" + str(np.max(r)))
    print("r min" + str(np.min(r)))
    '''
    r0 = 1.0 if r0 is None else r0
    dr = 1.0 if dr is None else dr
    return np.radians(a / (1.0 + np.exp(-(r0 - r) / (0.1*dr))))

'''
Original matrix from (Zuleta, 2024) with 2D inputs and outputs
'''
def apply_matrix2d_d(p0, warp, twist, inc_, PA_):
    inc_ = 0
    x = p0[:, :, 0]
    y = p0[:, :, 1]
    z = p0[:, :, 2]
    '''
    plt.pcolor(x, y, z)
    plt.title("z apply matrix input")
    plt.colorbar()
    plt.show()
    '''

    warp = warp[:, None]
    #print("warp.shape" + str(warp.shape))
    twist = twist[:, None]
    #print("twist.shape" + str(twist.shape))

    cosw = np.cos(warp)
    sinw = np.sin(warp)

    cost = np.cos(twist)
    sint = np.sin(twist)

    cosPA = np.cos(PA_)
    sinPA = np.sin(PA_)

    cosi = np.cos(inc_)
    sini = np.sin(inc_)

    xp = x*(-sinPA*sint*cosi + cosPA*cost) + y*((-sinPA*cosi*cost - sint*cosPA)*cosw + sinPA*sini*sinw) + z*(-(-sinPA*cosi*cost - sint*cosPA)*sinw + sinPA*sini*cosw)
    yp = x*(sinPA*cost + sint*cosPA*cosi) + y*((-sinPA*sint + cosPA*cosi*cost)*cosw - sini*sinw*cosPA) + z*(-(-sinPA*sint + cosPA*cosi*cost)*sinw - sini*cosPA*cosw)
    zp = x*sini*sint + y*(sini*cost*cosw + sinw*cosi) + z*(-sini*sinw*cost + cosi*cosw)

    return np.moveaxis([xp, yp, zp], 0, 2)

'''
My matrix with 3D inputs and outputs
'''

def matrix_mine(x, y, z, warp, twist, inc_, PA_):

    warp = warp[:, None, None]
    #print("warp.shape" + str(warp.shape))
    twist = twist[:, None, None]
    #print("twist.shape" + str(twist.shape))

    cosw = np.cos(warp)
    sinw = np.sin(warp)
    

    cost = np.cos(twist)
    sint = np.sin(twist)

    cosPA = np.cos(PA_)
    sinPA = np.sin(PA_)

    cosi = np.cos(inc_)
    sini = np.sin(inc_)

    xp = x*(-sinPA*sint*cosi + cosPA*cost) + y*((-sinPA*cosi*cost - sint*cosPA)*cosw + sinPA*sini*sinw) + z*(-(-sinPA*cosi*cost - sint*cosPA)*sinw + sinPA*sini*cosw)
    yp = x*(sinPA*cost + sint*cosPA*cosi) + y*((-sinPA*sint + cosPA*cosi*cost)*cosw - sini*sinw*cosPA) + z*(-(-sinPA*sint + cosPA*cosi*cost)*sinw - sini*cosPA*cosw)
    zp = y*(sini*cost*cosw + sinw*cosi) + z*(-sini*sinw*cost + cosi*cosw)

    return xp, yp, zp


'''
I tried warping the rt grid as well, and it required indexing the grids slightly differently
'''
def matrix_mine_rt(x, y, z, warp, twist, inc_, PA_):

    warp = warp[None, :, None]
    #print("warp.shape" + str(warp.shape))
    twist = twist[None, :, None]
    #print("twist.shape" + str(twist.shape))

    cosw = np.cos(warp)
    sinw = np.sin(warp)
    

    cost = np.cos(twist)
    sint = np.sin(twist)

    cosPA = np.cos(PA_)
    sinPA = np.sin(PA_)

    cosi = np.cos(inc_)
    sini = np.sin(inc_)

    xp = x*(-sinPA*sint*cosi + cosPA*cost) + y*((-sinPA*cosi*cost - sint*cosPA)*cosw + sinPA*sini*sinw) + z*(-(-sinPA*cosi*cost - sint*cosPA)*sinw + sinPA*sini*cosw)
    yp = x*(sinPA*cost + sint*cosPA*cosi) + y*((-sinPA*sint + cosPA*cosi*cost)*cosw - sini*sinw*cosPA) + z*(-(-sinPA*sint + cosPA*cosi*cost)*sinw - sini*cosPA*cosw)
    zp = y*(sini*cost*cosw + sinw*cosi) + z*(-sini*sinw*cost + cosi*cosw)

    return xp, yp, zp



class Disk:
    'Common class for circumstellar disk structure'
    #Define useful constants
    AU = const.au.cgs.value          # - astronomical unit (cm)
    Rsun = const.R_sun.cgs.value     # - radius of the sun (cm)
    c = const.c.cgs.value            # - speed of light (cm/s)
    h = const.h.cgs.value            # - Planck's constant (erg/s)
    kB = const.k_B.cgs.value         # - Boltzmann's constant (erg/K)
    pc = const.pc.cgs.value          # - parsec (cm)
    Jy = 1.e23                       # - cgs flux density (Janskys)
    Lsun = const.L_sun.cgs.value     # - luminosity of the sun (ergs)
    Mearth = const.M_earth.cgs.value # - mass of the earth (g)
    mh = const.m_p.cgs.value         # - proton mass (g)
    Da = mh                          # - atomic mass unit (g)
    Msun = const.M_sun.cgs.value     # - solar mass (g)
    G = const.G.cgs.value            # - gravitational constant (cm^3/g/s^2)
    rad = 206264.806   # - radian to arcsecond conversion
    kms = 1e5          # - convert km/s to cm/s
    GHz = 1e9          # - convert from GHz to Hz
    mCO = 12.011+15.999# - CO molecular weight
    mHCO = mCO+1.008-0.0005 # - HCO molecular weight
    mu = 2.37          # - gas mean molecular weight
    m0 = mu*mh     # - gas mean molecular opacity
    Hnuctog = 0.706*mu   # - H nuclei abundance fraction (H nuclei:gas)
    sc = 1.59e21   # - Av --> H column density (C. Qi 08,11)
    H2tog = 0.8 # - H2 abundance fraction (H2:gas)

    '''freeze out = 0 for dd parameters'''
    Tco = 0   
    #Tco = 19.    # - freeze out
    sigphot = 0.79*sc   # - photo-dissociation column

    '''Theo -- added warp parameters to init'''
#    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,-1,0,0,[.76,1000],[10,800]],obs=[180,131,300,170],rtg=True,vcs=True,line='co',ring=None):
    def __init__(self,q=-0.5,McoG=0.09,pp=1.,Ain=10.,Aout=1000.,Rc=150.,incl=51.5,
                 Mstar=2.3,Xco=1e-4,vturb=0.01,Zq0=33.9,Tmid0=19.,Tatm0=69.3,
                 handed=-1,ecc=0.,aop=0.,sigbound=[.79,1000],Rabund=[10,800],
                 nr=180,nphi=131,nz=300,zmax=170,rtg=True,vcs=True,line='co',ring=None,w_i=10, w_r0=10,w_dr=10,w_pa=10, rotation=0):
        #not sure if I'm doing this right, but adding some parameters here for warp
        params=[q,McoG,pp,Ain,Aout,Rc,incl,Mstar,Xco,vturb,Zq0,Tmid0,Tatm0,handed,ecc,aop,sigbound,Rabund,w_i,w_r0,w_dr,w_pa, rotation]
        obs=[nr,nphi,nz,zmax]
        #tb = time.clock()
        self.ring=ring
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters

        self.set_structure()  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid()
            self.set_line(line=line,vcs=vcs)
        #tf = time.clock()
        #print("disk init took {t} seconds".format(t=(tf-tb)))

    def set_structure(self):
        #tst=time.clock()
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nac = 180#256             # - number of unique a rings
        amin = self.Ain       # - minimum a [AU]
        amax = self.Aout      # - maximum a [AU]
        e = self.ecc          # - eccentricity
        #nzc = int(2.5*nac)#nac*5
        nzc = self.nz         # - number of unique z points
        '''defining z-array: .1 AU to specified max value in AU. logarithmic. number specified by
        # of annuli'''
        #zmin = .1*Disk.AU      # - minimum z [AU]
        '''Right now, defining z-dimension from -zmax to zmax, instead of top half of disk'''
        zmin = -self.zmax      # - minimum z [AU]
        nfc = self.nphi       # - number of unique f points
        '''putting into linspace for now'''
        af = np.linspace(amin,amax,nac)
        zf = np.linspace(zmin,self.zmax,nzc)
        #zf_warp = np.linspace(zmin+1,self.zmax-1,nzc-2)
        #zf = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)


        pf = np.linspace(0,2*np.pi,self.nphi) #f is with refrence to semi major axis
        ff = (pf - self.aop) % (2*np.pi) # phi values are offset by aop- refrence to sky
        #print("ff" + str(ff))
        rf = np.zeros((nac,nfc))
        for i in range(nac):
            for j in range(nfc):
                rf[i,j] = (af[i]*(1.-e*e))/(1.+e*np.cos(ff[j]))

        '''1d array of z-values as ones'''
        idz = np.ones(nzc)
        idf = np.ones(self.nphi)
        #rcf = np.outer(rf,idz)
        ida = np.ones(nac)
        ##zcf = np.outer(ida,zf)
        ##acf = af[:,np.newaxis]*np.ones(nzc)
        #order of dimensions: a, f, z
        '''meshgrid of z values above midplane'''
        pcf,acf,zcf = np.meshgrid(pf,af,zf)
        fcf = (pcf - self.aop) % (2*np.pi)
    
        #rcf=rf[:,:,np.newaxis]*idz
        rcf = acf

        self.pcf=pcf
        self.acf=acf
        self.zcf=zcf

        '''warp code'''
        '''defining warp, taking parmas from input into Disk'''
        '''defines change in inclination'''
        warp_i  = w_func(self, af, type="w")
        '''defines twist'''
        twist_i = w_func(self, af, type="pa")

        '''defining cartesian system for warp rotation'''
        xi, yi = pol2cart(acf, pcf)
        x_w, y_w, z_w = matrix_mine(xi, yi, zcf, warp_i, twist_i, 0, 0)



        '''calculating velocity structure at the same time as the positional warp'''
        #vel_phi = np.sqrt(Disk.G*self.Mstar/(acf*(1-self.ecc**2.)))*(np.cos(self.aop+fcf)+self.ecc*self.cosaop)
        
        vel_phi = np.sqrt(Disk.G*self.Mstar/(acf*(1-self.ecc**2.)))
        
        vx = -vel_phi*np.sin(pcf)
        vy = vel_phi*np.cos(pcf)
        vz = np.zeros(pcf.shape)

        vx_w, vy_w, vz_w = matrix_mine(vx, vy, vz, warp_i, twist_i, 0, 0)

        '''now, global az rotation of the disk:'''
        rot = np.radians(self.rot)
        trotx = x_w*np.cos(rot)-y_w*np.sin(rot) 
        troty = x_w*np.sin(rot)+y_w*np.cos(rot) 
        
        rotvx = vx_w*np.cos(rot)-vy_w*np.sin(rot) 
        rotvy = vx_w*np.sin(rot)+vy_w*np.cos(rot)

        #plot(trotx[:,:,0], troty[:,:,0], vz_w[:,:,0], "uninclined z velocity after az rotation", label="z vel (cm/s)")

        #self.vx_w = rotvx
        #self.vy_w = rotvy
        #self.vz_w = vz_w
        
        '''calculating velocity structure after global inclination'''
        tvy_rot = (rotvy*np.cos(self.thet) - vz_w*np.sin(self.thet)) 
        tvz_rot = (rotvy*np.sin(self.thet) + vz_w*np.cos(self.thet)) 
        self.vz_w = tvz_rot

        ty_w = (troty*np.cos(self.thet) - z_w*np.sin(self.thet))
        tz_w = (troty*np.sin(self.thet) + z_w*np.cos(self.thet)) 

        #plot(trotx[:,:,150], troty[:,:,150], rotvx[:,:,150], "x velocity", label="x vel (cm/s)")
        #plot(trotx[:,:,150], troty[:,:,150], rotvy[:,:,150], "y velocity", label="y vel (cm/s)")
        #plot(trotx[:,:,150], troty[:,:,150], vz_w[:,:,150], "z velocity", label="z vel (cm/s)")
        
        self.x_grid = trotx
        self.y_grid = troty
        self.z_grid = z_w

        '''converting back to polar coordinates'''
        r_w, p_w = cart2pol(trotx, troty)
        #r_w, p_w = cart2pol(x_w, y_w)
        #p_w = p_w + np.pi 

        '''useful for plotting polar graphs in cart space'''
        self.x_polar_w, self.y_polar_w = pol2cart(r_w[:,:,0], p_w[:,:,0])
        self.x_polar, self.y_polar = pol2cart(acf[:,:,0], pcf[:,:,0])
        
        #plot(self.x_polar_w, self.y_polar_w, z_w[:,:,150], "warp in disk plane (midplane)", label="z [AU]")

        grid = {'nac':nac,'nfc':nfc,'nzc':nzc,'rcf':rcf,'amax':amax,'zcf':zcf}#'ff':ff,'af':af,
        self.grid=grid

        '''using debris disk parameters, setting tmid = tatm, and using 100 AU instead of 150 AU'''

        delta = 1.                # shape parameter
        rcf100=rcf/(100.*Disk.AU)
        rcf100q=rcf100**self.qq

        zq = self.zq0*Disk.AU*rcf100**1.3
        #zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.1
        tmid = self.tmid0*rcf100q
        tatm = tmid
        #tatm = self.tatm0*rcf100q
        tempg = tatm + (tmid-tatm)*np.cos((np.pi/(2*zq))*zcf)**(2.*delta)
        ii = zcf > zq
        tempg[ii] = tatm[ii]

        

        #Type I structure
#        tempg = tmid*np.exp(np.log(tatm/tmid)*zcf/zq)
        ###### this step is slow!!! ######
        #print("temp struct {t}".format(t=time.clock()-tst)

        # Calculate vertical density structure
        # nolonger use exponential tail
        ## Circular:
        #Sc = self.McoG*(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
        #siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        ## Elliptical:
        #asum = (np.power(af,-1*self.pp)).sum()
        rp1 = np.roll(rf,-1,axis=0)
        rm1 = np.roll(rf,1,axis=0)
        #*** Approximations used here ***#
        #siggas = (self.McoG*np.sqrt(1.-e*e))/((rp1-rm1)*np.pi*(1.+e*np.cos(fcf[:,:,0]))*np.power(acf[:,:,0],self.pp+1.)*asum)
        #siggas[0,:] = (self.McoG*np.sqrt(1.-e*e))/((rf[1,:]-rf[0,:])*2.*np.pi*(1.+e*np.cos(ff))*np.power(af[0]*idf,self.pp+1.)*asum)
        #siggas[nac-1,:] = (self.McoG*np.sqrt(1.-e*e))/((rf[nac-1,:]-rf[nac-2,:])*2.*np.pi*(1.+e*np.cos(ff))*np.power(af[nac-1]*idf,self.pp+1.)*asum)
        Sc = self.McoG*(2.-self.pp)/(self.Rc*self.Rc)
        siggas_r = Sc*(acf[:,:,0]/self.Rc)**(-1*self.pp)*np.exp(-1*(acf[:,:,0]/self.Rc)**(2-self.pp))
        #Sc = self.McoG*(2.-self.pp)/((amax**(2-self.pp)-amin**(2-self.pp)))
        #siggas_r = Sc*acf[:,:,0]**(-1*self.pp)
        dsdth = (acf[:,:,0]*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e))/(1+e*np.cos(pcf[:,:,0]))**2
        siggas = ((siggas_r*np.sqrt(1.-e*e))/(2*np.pi*acf[:,:,0]*np.sqrt(1+2*e*np.cos(pcf[:,:,0])+e*e)))*dsdth

        #print("siggas shape: "+ str(siggas.shape))

        ## Add an extra ring
        if self.ring is not None:
            w = np.abs(rcf-self.Rring)<self.Wring/2.
            if w.sum()>0:
                tempg[w] = tempg[w]*(rcdf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)/((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.sig_enhance)


        if 0:
            # check that siggas adds up to Mdisk #
            df=ff[1]-ff[0]
            dA = 0.5*(rp1-rm1)*df*rf
            dA[0,:]=(rf[1,:]-rf[0,:])*rf[0,:]*df
            dA[nac-1,:]=(rf[nac-1,:]-rf[nac-2,:])*rf[nac-1,:]*df
            mcheck=(siggas*dA)
            mcheck=mcheck.sum()
            #print("sig mass check (should be 1)")
            #print(mcheck/self.McoG)

            #dsdth = (acf*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf)+e*e))/(1+e*np.cos(fcf))**2
            dr = af-np.roll(af,1)
            dr[0] = af[0]
            dr = dr[:,np.newaxis]*np.ones(nfc)
            dm = (siggas*dr*acf[:,:,0]*df)
            #dm = (linrho*dA*dsdth*2*np.pi)
#dm[0] = 0
            print('second sig mass check ',dm.sum()/self.McoG)


        self.calc_hydrostatic(tempg,siggas,grid)


        # Check for NANs
        ii = np.isnan(self.rho0)
        if ii.sum() > 0:
            self.rho0[ii] = 1e-60
            print('Beware: removed NaNs from density (#%s)' % ii.sum())
        ii = np.isnan(tempg)
        if ii.sum() > 0:
            tempg[ii] = 2.73
            print('Beware: removed NaNs from temperature (#%s)' % ii.sum())

        #print("nan chekc {t}".format(t=time.clock()-tst))
        # find photodissociation boundary layer from top
        zpht_up = np.zeros((nac,nfc))
        zpht_low = np.zeros((nac,nfc))
        sig_col = np.zeros((nac,nfc,nzc))
        #zice = np.zeros((nac,nfc))
        for ia in range(nac):
            for jf in range (nfc):
                psl = (Disk.Hnuctog/Disk.m0*self.rho0[ia,jf,:])[::-1]
                zsl = self.zmax - (zcf[ia,jf,:])[::-1]
                foo = (zsl-np.roll(zsl,1))*(psl+np.roll(psl,1))/2.
                foo[0] = 0
                nsl = foo.cumsum()
                sig_col[ia,jf,:] = nsl[::-1]*Disk.m0/Disk.Hnuctog
                pht = (np.abs(nsl) >= self.sigbound[0])
                if pht.sum() == 0:
                    zpht_up[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_up[ia,jf] = np.max(self.zmax-zsl[pht])
                #Height of lower column density boundary
                pht = (np.abs(nsl) >= self.sigbound[1])
                if pht.sum() == 0:
                    zpht_low[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_low[ia,jf] = np.max(self.zmax-zsl[pht])
                #used to be a seperate loop
                ###### only used for plotting
                #foo = (tempg[ia,jf,:] < Disk.Tco)
                #if foo.sum() > 0:
                #    zice[ia,jf] = np.max(zcf[ia,jf,foo])
                #else:
                #    zice[ia,jf] = zmin

        '''indexing cylindrical warped structure to interpolate temp, density, and vel grids on to'''
        '''expanding z dimension to account for extension from warp'''
        z_w_max = np.max(z_w)
        self.z_w_max = z_w_max
        zf_w = np.linspace(-z_w_max, z_w_max, nzc)
        self.zf_w = zf_w

        pcf_shift = (pcf - rot) % (2*np.pi)
        p_w_shift = (p_w + rot) % (2*np.pi)

        aind_w = np.interp(r_w.flatten(), af, range(nac), right=nac)
        aind = np.interp(rcf.flatten(), af, range(nac), right=nac)
        #phiind_w = np.interp(p_w.flatten(), pf, range(self.nphi))
        phiind_w_shift = np.interp(p_w_shift.flatten(), pf, range(self.nphi))
        phiind_shift = np.interp(pcf_shift.flatten(), pf, range(self.nphi))
        #phiind = np.interp(pcf.flatten(), pf, range(self.nphi))
        #zind_w = np.interp(z_w.flatten(), zf[:-150],range(nzc-150), right=nzc-150)
        zind_w = np.interp(z_w.flatten(), zf_w,range(nzc), right=nzc)
        zind = np.interp(zcf.flatten(), zf, range(nzc), right=nzc)

        #imshow(self.vz_w[:,:,0], title="natural velocity grid", label="vel [cm/s]")

        #plot(self.x_polar, self.y_polar, self.vz_w[:,:,0], "vel, before spatial incl", label="vel [cm/s]")
        #plot(trotx[:,:,0], ty_w[:,:,0], tz_w[:,:,0], "warped positions, inclined", label="z [cm]", ylim=[-np.max(trotx), np.max(trotx)])
        #plot(trotx[:,:,0], ty_w[:,:,0], self.vz_w[:,:,0], "vel, referenced with warped grid inclined", label="vel [cm/s]", ylim=[-np.max(trotx), np.max(trotx)])

        '''interpolating temp, vel, and density onto warped grid
        vel is calculated directly on warped grid, so needs to be interpolated back to natural grid for RT calculations
        temp & sig_col are calculated on unwarped grid, so need to be interpolated onto warped grid for RT calculations
        '''

        temp_interp = ndimage.map_coordinates(tempg,[[aind_w], [phiind_w_shift], [zind_w]], order=1).reshape(nac,self.nphi, nzc)
        sig_col_interp = ndimage.map_coordinates(sig_col,[[aind_w], [phiind_w_shift], [zind_w]], order=1).reshape(nac,self.nphi, nzc)
        #temp_interp = ndimage.map_coordinates(tempg,[[aind], [phiind_shift], [zind]], order=1).reshape(nac,self.nphi, nzc)
        #sig_col_interp = ndimage.map_coordinates(sig_col,[[aind], [phiind_shift], [zind]], order=1).reshape(nac,self.nphi, nzc)
        vel_interp = ndimage.map_coordinates(self.vz_w,[[aind], [phiind_shift], [zind]], order=1).reshape(nac,self.nphi, nzc)

        #plot(trotx[:,:,0], troty[:,:,0], temp_interp[:,:,150], "temp interp, rotated grid", label="temp [K]")
        #plot(trotx[:,:,0], troty[:,:,0], sig_col_interp[:,:,150], "sig col interp, rotated grid", label="sig col [cm^-2]")
        plot(self.x_polar, self.y_polar, temp_interp[:,:,0], "temp interp, unwarped uninclined coords", label="temp [K]")
        plot(self.x_polar, self.y_polar, sig_col_interp[:,:,0], "sig col interp, unwarped uninclined coords", label="sig col [cm^-2]")

        self.sig_col = sig_col_interp
        self.vel = vel_interp
        self.tempg = temp_interp

        plot(self.x_polar, self.y_polar, vel_interp[:,:,0], "vel interp, before spatial incl", label="vel [cm/s]")

        '''
        szpht = zpht
        #zpht = scipy.signal.medfilt(zpht,kernel_size=7) #smooth it

        # find height where CO freezes out
        # only used for ploting
        zice = np.zeros(nrc)
        for ir in range(nrc):
            foo = (tempg[ir,:] < Disk.Tco)
            if foo.sum() > 0:
                zice[ir] = np.max(zcf[ir,foo])
            else:
                zice[ir] = zmin
        '''
        self.af = af
        #self.ff = ff
        #self.rf = rf
        self.pf = pf
        self.nac = nac
        self.zf = zf
        self.nzc = nzc
        
        #self.Omg0 = Omg#velrot
        self.zpht_up = zpht_up
        self.zpht_low = zpht_low
        self.pcf = pcf  #only used for plotting can remove after testing
        self.rcf = rcf  #only used for plotting can remove after testing


    def set_rt_grid(self):


        
        #tst=time.clock()
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        #Smin = 1*Disk.AU                 # offset from zero to log scale
        #if self.thet > np.arctan(self.Aout/self.zmax):
        #    Smax = 2*self.Aout/self.sinthet
        #else:
        #    Smax = 2.*self.zmax/self.costhet       # los distance through disk
        #Smid = Smax/2.                    # halfway along los
        #ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center
        #sky coordinates
        #R = np.logspace(np.log10(self.Ain*(1-self.ecc)),np.log10(self.Aout*(1+self.ecc)),self.nr)
        R = np.linspace(0,self.Aout*(1+self.ecc),self.nr) #******* not on cluster*** #
        '''as in disk_ecc to run consistency test'''
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #print("nr " + str(self.nr))

        #print("nr " + str(self.nr))
        #print("nphi " + str(self.nphi))
        #phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #phi = np.linspace(0, 2*np.pi, self.nphi)
        #print("nphi " + str(self.nphi))
        #z_l = self.zf
        #z_l = np.arange(self.nz)/self.nz*(-self.zmax)+s/2.

        #z_l = np.linspace(self.zmax,-self.zmax,self.nz)
        z_l = np.linspace(self.z_w_max,-self.z_w_max,self.nz)
        #print("nz " + str(self.nz))


        '''trying meshgridding first before converting to polar to match
        set_structure method'''

        R_mesh, phi_mesh, Z = np.meshgrid(R, phi, z_l)
        #print("grid shape " + str(phi_mesh.shape))
        #fcf = (pcf - self.aop) % (2*np.pi)
        #phi_mesh_aop = (phi_mesh - self.aop) % (2*np.pi)

        warp_rt  = w_func(self, R, type="w")
        twist_rt = w_func(self, R, type="pa")


        #total_inc = warp_rt + self.thet


        '''leaving X & Y here as sky coordinates, but since right now they are the same
        dimension as disk coordinates, I am going to use them as the basis for the warp'''
        Xw, Yw = pol2cart(R_mesh, phi_mesh)

        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()


        '''applying warp
        necessary for S'''

        X_w, Y_w, Z_w = matrix_mine_rt(Xw, Yw, Z, warp_rt, twist_rt,0,0)

        '''calculating velocity directly here instead of interpolating from set_structure'''

        #vel_phi = np.sqrt(Disk.G*self.Mstar/(R_mesh*(1-self.ecc**2.)))

        
        #vx = -vel_phi*np.sin(phi_mesh)
        #vy = vel_phi*np.cos(phi_mesh)
        #vz = np.zeros(phi_mesh.shape)

        #vx_w, vy_w, vz_w = matrix_mine_rt(vx, vy, vz, warp_rt, twist_rt, 0, 0)

        '''now, global az rotation of the disk:'''
        rot = np.radians(self.rot)
        #trotx = X_w*np.cos(rot)-Y_w*np.sin(rot) 
        #troty = X_w*np.sin(rot)+Y_w*np.cos(rot) 
        
        #rotvx = vx_w*np.cos(rot)-vy_w*np.sin(rot) 
        #rotvy = vx_w*np.sin(rot)+vy_w*np.cos(rot)

        #self.vx_w = rotvx
        #self.vy_w = rotvy
        #self.vz_w = vz_w


        #plot(trotx[:,:,150], troty[:,:,150], rotvx[:,:,150], "x velocity", label="x vel (cm/s)")
        #plot(trotx[:,:,150], troty[:,:,150], rotvy[:,:,150], "y velocity", label="y vel (cm/s)")
        #plot(trotx[:,:,150], troty[:,:,150], vz_w[:,:,150], "z velocity", label="z vel (cm/s)")


        #plot(X_w[:,:,150], Y_w[:,:,150], Z_w[:,:,150], "Warped grid", label="z [AU]")

        rot = np.radians(self.rot)

        #X_nonw_rot = X*np.cos(rot)-Y*np.sin(rot)
        #Y_nonw_rot = X*np.sin(rot)+Y*np.cos(rot)

        #X_rot = X_w*np.cos(rot)-Y_w*np.sin(rot) 
        #Y_rot = X_w*np.sin(rot)+Y_w*np.cos(rot)

        #plot(X_rot[:,:,150], Y_rot[:,:,150], Z_w[:,:,150], "Warp before incl", label="z cm", ylim=[-np.max(X), np.max(X)])
 

        nac = 500#256             # - number of unique a ring
        #nzc = int(5*nac)#nac*5           # - number of unique z points

        if np.abs(self.thet) > np.arctan(self.Aout*(1+self.ecc)/self.zmax):
            #zsky_w = np.abs(Z_w/self.sinthet)
            #zsky_w = Z_w/self.sinthet
            zsky_max = np.abs(2*self.Aout*(1+self.ecc)/self.sinthet)
            print("weird zsky_w True")
        else:
            #zsky_w = (Z_w/self.costhet)
            #zsky_w = Z_w
            #zsky = (Z/self.costhet)
            zsky_max = 2*(self.z_w_max/self.costhet)
            #zsky = Z
            print("normal zsky_w True")


        '''I create a warped and unwarped grid here
        for something to compare against '''

        zsky = np.arange(self.nz)/self.nz*(-zsky_max)+zsky_max/2.

        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet

        #tdiskY_w = (Y_rot*self.costhet - zsky_w*self.sinthet)
        #tdiskZ_w = (Y_rot*self.sinthet + zsky_w*self.costhet)

        #tdiskY = (Y*self.costhet - zsky*self.sinthet)
        #tdiskZ = (Y*self.sinthet + zsky*self.costhet)

        #tdiskY = (Y*self.costhet - zsky_w*self.sinthet)
        #tdiskZ = (Y*self.sinthet + zsky_w*self.costhet)

        #tdiskY_rot = (Y_nonw_rot*self.costhet - zsky*self.sinthet)
        #tdiskZ_rot = (Y_nonw_rot*self.sinthet + zsky*self.costhet)

        #tvy = (self.vy_w*self.costhet - self.vz_w*self.sinthet)
        #tvz = (self.vy_w*self.sinthet + self.vz_w*self.costhet)

        #tvel = tvz


        #tvy_rot = (rotvy*np.cos(inc) - vwz*np.sin(inc)) 
        #tvz_rot = (rotvy*np.sin(inc) + vwz*np.cos(inc)) 

        #plot(X_rot[:,:,150], tdiskY_w[:,:,150], tdiskZ_w[:,:,150], "Warp on sky", label="z [AU]", ylim=[-np.max(X), np.max(X)])
        
        #total_inc_full_grid = total_inc[np.newaxis, :, np.newaxis]*np.ones((self.nphi, self.nr, self.nz))

        '''I have not modified S'''
        if (self.thet<np.pi/2) & (self.thet>0):
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax-tdiskZ))
            #S = (self.zmax-tdiskZ)/self.costhet
            S = (Z_w[:,:,0][:,:,np.newaxis]*np.ones((self.nphi, self.nr, self.nz))-Z_w)/self.costhet
            #S[(theta_crit<self.thet)] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<self.thet)])/self.sinthet)
            #S[(theta_crit<self.thet)] = 0
            mask = (theta_crit < self.thet)
            S[mask] = ((Y_w[:,:,0][:,:,np.newaxis]*np.ones((self.nphi, self.nr, self.nz)) - Y_w)[mask]) / np.sin(self.thet)
            #mask = (theta_crit < total_inc_full_grid)
            #S[mask] = ((self.Aout*(1+self.ecc) + tdiskY[mask]) / np.sin(total_inc_full_grid[mask]))
        elif self.thet>np.pi/2:
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax+tdiskZ))  
            S = -(self.zmax+tdiskZ)/self.costhet
            S[(theta_crit<(np.pi-self.thet))] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<(np.pi-self.thet))])/self.sinthet)
        elif (self.thet<0) & (self.thet>-np.pi/2):
            theta_crit = np.arctan((self.Aout*(1+self.ecc)-tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<np.abs(self.thet))] = -((self.Aout*(1+self.ecc)-tdiskY[(theta_crit<np.abs(self.thet))])/self.sinthet)

        imshow(S[:,:,0], title="S bottom of disk")
        imshow(S[:,:,-1], title="S top of disk")

        tr = np.sqrt(X**2+tdiskY**2)
        #tr_w = np.sqrt(X_rot**2+tdiskY_w**2)
  
        #tphi = np.arctan2(tdiskY,X_w.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))%(2*np.pi)
        #tphi = np.arctan2(tdiskY,X)%(2*np.pi)
        tphi = np.arctan2(tdiskY,X)%(2*np.pi)
        #tphi_shift = (tphi - rot)%(2*np.pi)
        #tphi_rot = np.arctan2(tdiskY_rot,X_nonw_rot)%(2*np.pi)
        #tphi_w = np.arctan2(tdiskY_w,X_rot)%(2*np.pi)

        #imshow(tdiskY_w[:,:,0], title="Y warped grid", label="Y")
        #imshow(X_rot[:,:,0], title="X warped grid", label="X")
        #imshow(np.arctan2(tdiskY_w[:,:,0],X_rot[:,:,0]), title="single slice phi warped grid")
        #tphi_w = np.arctan2(X_w,tdiskY_w)%(2*np.pi)

        #print("tdiskY_w[:,:,0]:"+ str(tdiskY_w[:,:,0]))
        #print("X_rot[:,:,0]:"+ str(X_rot[:,:,0]))

        #tr_w, phi_w = cart2pol(X_rot, tdiskY_w)

        #tr_rot, phi_rot = cart2pol(X_nonw_rot, tdiskY_rot)

        #tphi_w = (phi_w)%(2*np.pi)

        #imshow(tphi_w[:,:,0], title="phi warped grid", label="phi [rad]")

        ###### should be real outline? requiring a loop over f or just Aout(1+ecc)######
        notdisk = (tr > self.Aout*(1.+self.ecc)) | (tr < self.Ain*(1-self.ecc))  # - individual grid elements not in disk
        isdisk = (tr>self.Ain*(1-self.ecc)) & (tr<self.Aout*(1+self.ecc)) & (np.abs(tdiskZ)<self.zmax)
        S -= S[isdisk].min() #Reset the min S to 0
        #xydisk =  tr[:,:,0] <= self.Aout*(1.+self.ecc)+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        self.r = tr
        self.phi = tphi

        zind = np.interp(np.abs(tdiskZ).flatten(),self.zf_w,range(self.nzc)) #zf,nzc
        #zind_rot = np.interp(np.abs(tdiskZ_rot).flatten(),self.zf_w,range(self.nzc))
        #zind_w = np.interp(np.abs(tdiskZ_w).flatten(),self.zf_w,range(self.nzc))
        
        phiind = np.interp(tphi.flatten(),self.pf,range(self.nphi))
        #phiind_shift = np.interp(tphi_shift.flatten(),self.pf,range(self.nphi))
        #phiind_rot = np.interp(tphi_rot.flatten(),self.pf,range(self.nphi))
        #phiind_w = np.interp(tphi_w.flatten(),self.pf,range(self.nphi))

        aind = np.interp((tr.flatten()*(1+self.ecc*np.cos(tphi.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)
        #aind_w = np.interp((tr_w.flatten()*(1+self.ecc*np.cos(tphi_w.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)
        #aind_rot = np.interp((tr_rot.flatten()*(1+self.ecc*np.cos(phi_rot.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)

        #imshow(aind_w.reshape(self.nphi,self.nac, self.nzc)[:,:,0], title="a index warped grid", label="a index")
        #imshow(phiind_w.reshape(self.nphi, self.nac, self.nzc)[:,:,0], title="phi index warped grid", label="phi index")
        #imshow(zind_w.reshape(self.nphi, self.nac, self.nzc)[:,:,0], title="z index warped grid", label="z index")

        tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        #tT_w = ndimage.map_coordinates(self.tempg,[[aind_w],[phiind_w],[zind_w]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)

        tvel = ndimage.map_coordinates(self.vel,[[aind],[phiind],[zind]],order=1,).reshape(self.nphi,self.nr,self.nz)
        #tvel = ndimage.map_coordinates(tvz,[[aind],[phiind_shift],[zind]],order=1,).reshape(self.nphi,self.nr,self.nz)
        #tvel_w = ndimage.map_coordinates(tvz,[[aind_w],[phiind_w],[zind_w]],order=1,).reshape(self.nphi,self.nr,self.nz)
        #tvel_rot = ndimage.map_coordinates(tvz,[[aind_rot],[phiind_rot],[zind_rot]],order=1,).reshape(self.nphi,self.nr,self.nz)
        #plot(X[:,:,150], tdiskY[:,:,150], tvel[:,:,150], "warped velocities", label="vel (cm/s)", ylim=[-np.max(X), np.max(X)])
        #tvel = tvel_w

        plot(X[:,:,150], tdiskY[:,:,150], tvel[:,:,150], title="final velocity structure", label="z vel (cm/s)", ylim=[-np.max(X), np.max(X)])
        
        #plot(X_w[:,:,0], tdiskY_w[:,:,0], tvel_w[:,:,0], title="cartesian vel")

        tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #tsig_col_w = ndimage.map_coordinates(self.sig_col,[[aind_w],[phiind_w],[zind_w]],order=1).reshape(self.nphi,self.nr,self.nz)

        #plot(X_w[:,:,0], tdiskY_w[:,:,0], tsig_col_w[:,:,0], title="cartesian sig col")

        zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        #zpht_up_w = ndimage.map_coordinates(self.zpht_up,[[aind_w],[phiind_w]],order=1).reshape(self.nphi,self.nr,self.nz)
        #zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #zpht_low_w = ndimage.map_coordinates(self.zpht_low,[[aind_w],[phiind_w]],order=1).reshape(self.nphi,self.nr,self.nz)
        tT[notdisk] = 0

        self.sig_col = tsig_col

        #plot(X[:,:,150], tdiskY[:,:,150], tvel[:,:,150], "Velocity on sky, unwarped coords", label="cm/s", ylim=[-np.max(X), np.max(X)])
        #plot(X_rot[:,:,150], tdiskY_w[:,:,150], tvel_w[:,:,150], "Velocity on sky, warped coords", label="cm/s", ylim=[-np.max(X), np.max(X)])


        self.add_mol_ring(self.Rabund[0]/Disk.AU,self.Rabund[1]/Disk.AU,self.sigbound[0]/Disk.sc,self.sigbound[1]/Disk.sc,self.Xco,initialize=True)

        if np.size(self.Xco)>1:
            Xmol = self.Xco[0]*np.exp(-(self.Rabund[0]-tr)**2/(2*self.Rabund[3]**2))+self.Xco[1]*np.exp(-(self.Rabund[1]-tr)**2/(2*self.Rabund[4]**2))+self.Xco[2]*np.exp(-(self.Rabund[2]-tr)**2/(2*self.Rabund[5]**2))
        #else:
        #    Xmol = self.Xco



        #print("image interp {t}".format(t=time.clock()-tst))

        # photo-dissociation
        #zap = (np.abs(tdiskZ) > zpht_up)
        #if zap.sum() > 0:
        #    trhoG[zap] = 1e-18*trhoG[zap]
        #zap = (np.abs(tdiskZ) < zpht_low)
        #if zap.sum()>0:
        #    trhoG[zap] = 1e-18*trhoG[zap]

        #if np.size(self.Xco)<2:
        #    #Inner and outer abundance boundaries
        #    zap = (tr<=self.Rabund[0]) | (tr>=self.Rabund[1])
        #    if zap.sum()>0:
        #        trhoG[zap] = 1e-18*trhoG[zap]

        # freeze out
        zap = (tT <= Disk.Tco)
        if zap.sum() >0:
            self.Xmol[zap] = 1e-8*self.Xmol[zap]
            #trhoG[zap] = 1e-8*trhoG[zap]

        #trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind_unwarped],[phiind_unwarped],[zind_unwarped]],order=1).reshape(self.nphi,self.nr,self.nz)
        trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #trhoH2_w = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind_w],[phiind_w],[zind_w]],order=1).reshape(self.nphi,self.nr,self.nz)
        
        trhoG = trhoH2*self.Xmol
        #trhoG_w = trhoH2_w*self.Xmol
        trhoH2[notdisk] = 0
        trhoG[notdisk] = 0

        #trhoH2_w[notdisk] = 0
        #trhoG_w[notdisk] = 0
        
        '''I could also set these properties to take the warped versions'''
        #self.rhoH2 = trhoH2_w
        self.rhoH2 = trhoH2
        #self.sig_col = tsig_col_w
        self.sig_col = tsig_col
        #self.rhoG = trhoG_w
        self.rhoG = trhoG
        #self.T = tT_w
        self.T = tT

        #self.vel = tvel_w
        self.vel = tvel

        plt.imshow(self.vel[:,:,0])
        plt.title("vel")
        plt.colorbar()

        plt.show()

        # store disk

        '''originally, I think these parameters were X, Y, and tdiskZ?'''
        self.X = X
        self.Y = Y
        #self.X_w = X_w
        #self.X_unwarp = X
        #self.Y_unwarp = Y
        #self.X = X
        #self.Y = tdiskY_unwarped
        #self.tY = tdiskY_nosky
        #self.tZ = tdiskZ_nosky
        #self.tY_unwarp = tdiskY_unwarped
        #self.tZ_unwarp = tdiskZ_unwarped
        self.Z = tdiskZ
        self.S = S
        #self.r = tr
        
        #self.dBV = tdBV
        
        #self.Omg = Omg#Omgy #need to combine omgx,y,z
        
        self.i_notdisk = notdisk
        #self.i_xydisk = xydisk
        #self.rhoH2 = trhoH2 #*** not on cluster ***
        #self.sig_col=tsig_col
        #self.Xmol = Xmol
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        #self.tempg = tempg
        #self.zpht = zpht
        #self.phi = tphi


    '''end modified/added functions'''



    def set_params(self,params):
        'Set the disk structure parameters'
        self.qq = params[0]                 # - temperature index
        self.McoG = params[1]*Disk.Msun     # - gas mass
        self.pp = params[2]                 # - surface density index
        self.Ain = params[3]*Disk.AU        # - inner edge in cm
        self.Aout = params[4]*Disk.AU       # - outer edge in cm
        self.Rc = params[5]*Disk.AU         # - critical radius in cm
        self.thet = math.radians(params[6]) # - convert inclination to radians
        self.Mstar = params[7]*Disk.Msun    # - convert mass of star to g
        self.Xco = params[8]                # - CO gas fraction
        self.vturb = params[9]*Disk.kms     # - turbulence velocity
        self.zq0 = params[10]               # - Zq, in AU, at 150 AU
        self.tmid0 = params[11]             # - Tmid at 150 AU
        self.tatm0 = params[12]             # - Tatm at 150 AU
        self.handed = params[13]            #
        self.ecc = params[14]               # - eccentricity of disk
        self.aop = math.radians(params[15]) # - angle between los and perapsis convert to radians
        self.sigbound = [params[16][0]*Disk.sc,params[16][1]*Disk.sc] #-upper and lower column density boundaries

        '''I think I need to add parameters here for warp model...'''
        self.w_i = params[18]               # - inclination of warp
        self.w_r0 = params[19]*Disk.AU              # - inflection radius
        self.w_dr = params[20]*Disk.AU         # - how many annuli it takes for warp transition
        self.pa = params[21]                # - position angle (how rotated the face-on disk is) of warp
        self.rot = params[22]                # - global azimuthal rotation



        if len(params[17])==2:
            # - inner and outer abundance boundaries
            self.Rabund = [params[17][0]*Disk.AU,params[17][1]*Disk.AU]
        else:
            self.Rabund=[params[17][0]*Disk.AU,params[17][1]*Disk.AU,params[17][2]*Disk.AU,params[17][3]*Disk.AU,params[17][4]*Disk.AU,params[17][5]*Disk.AU]
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        self.cosaop = np.cos(self.aop)
        self.sinaop = np.sin(self.aop)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU # location of ring
            self.Wring = self.ring[1]*Disk.AU # width of ring
            self.sig_enhance = self.ring[2] # surface density enhancement (a multiplicative factor) above the background

        

    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU

      

    def set_line(self,line='co',vcs=True):
        self.line = line
        try:
            if line.lower()[:2]=='co':
                self.m_mol = 12.011+15.999
            elif line.lower()[:4]=='c18o':
                self.m_mol = 12.011+17.999
            elif line.lower()[:4]=='13co':
                self.m_mol = 13.003+15.999
            elif line.lower()[:3] == 'hco':
                self.m_mol = 1.01 + 12.01 + 16.0
            elif line.lower()[:3] == 'hcn':
                self.m_mol = 1.01 + 12.01 + 14.01
            elif line.lower()[:2] == 'cs':
                self.m_mol = 12.01 + 32.06
            elif line.lower()[:3] == 'dco':
                self.m_mol = 12.011+15.999+2.014
            else:
                raise ValueError('Choose a known molecule [CO, C18O, 13CO, HCO, HCO+, HCN, CS, DCO+] for the line parameter')
            #assume it is DCO+
        except ValueError as error:
            raise
        if vcs:
            #temperature and turbulence broadening
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mHCO)*tT+self.vturb**2)
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*self.m_mol)*self.T)) #vturb proportional to cs

        else: #assume line.lower()=='co'
            #temperature and turbulence broadening
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*self.m_mol)*tT+self.vturb**2)
            #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*self.T)) #vturb proportional to cs

        self.dBV=tdBV


    def add_dust_ring(self,Rin,Rout,dtg,ppD,initialize=False):
        '''Add a ring of dust with a specified inner radius, outer radius, dust-to-gas ratio (defined at the midpoint) and slope of the dust-to-gas-ratio'''

        if initialize:
            self.dtg = 0*self.r
            self.kap = 2.3

        w = (self.r>(self.Ain*Disk.AU)) & (self.r<(self.Aout*Disk.AU))
        Rmid = (Rin+Rout)/2.*Disk.AU
        self.dtg[w] += dtg*(self.r[w]/Rmid)**(-ppD)
        self.rhoD = self.rhoH2*self.dtg*2*Disk.mh

    def add_mol_ring(self,Rin,Rout,Sig0,Sig1,abund,alpha=0,initialize=False,just_frozen=False):
        '''Changed Rin to Ain (out)--Theo'''
        ''' Add a ring of fixed abundance, between Rin and Rout (in the radial direction) and Sig0 and Sig1 (in the vertical direction). The abundance is treated as a power law in the radial direction, with alpha as the power law exponent, and normalized at the inner edge of the ring (abund~abund0*(r/Rin)^(alpha))
        disk.add_mol_ring(10,100,.79,1000,1e-4)
        just_frozen: only apply the abundance adjustment to the areas of the disk where CO is nominally frozen out.'''
        if initialize:
            self.Xmol = np.zeros(np.shape(self.r))+1e-18
        if just_frozen:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>self.Ain*Disk.AU) & (self.r<self.Aout*Disk.AU) & (self.T<self.Tco)
        else:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>self.Ain*Disk.AU) & (self.r<self.Aout*Disk.AU)
        if add_mol.sum()>0:
            self.Xmol[add_mol]+=abund*(self.r[add_mol]/(self.Ain*Disk.AU))**(alpha)
        #add soft boundaries
        edge1 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>self.Aout*Disk.AU)
        if edge1.sum()>0:
            self.Xmol[edge1] += abund*(self.r[edge1]/(self.Ain*Disk.AU))**(alpha)*np.exp(-(self.r[edge1]/(self.Aout*Disk.AU))**16)
        edge2 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r<self.Ain*Disk.AU)
        if edge2.sum()>0:
            self.Xmol[edge2] += abund*(self.r[edge2]/(self.Ain*Disk.AU))**(alpha)*(1-np.exp(-(self.r[edge2]/(self.Ain*Disk.AU))**20.))
        edge3 = (self.sig_col*Disk.Hnuctog/Disk.m0<Sig0*Disk.sc) & (self.r>self.Ain*Disk.AU) & (self.r<self.Aout*Disk.AU)
        if edge3.sum()>0:
            self.Xmol[edge3] += abund*(self.r[edge3]/(self.Ain*Disk.AU))**(alpha)*(1-np.exp(-((self.sig_col[edge3]*Disk.Hnuctog/Disk.m0)/(Sig0*Disk.sc))**8.))
        zap = (self.Xmol<0)
        if zap.sum()>0:
            self.Xmol[zap]=1e-18
        if not initialize:
            self.rhoG = self.rhoH2*self.Xmol



    def calc_hydrostatic(self,tempg,siggas,grid):
        nac = grid['nac']
        nfc = grid['nfc']
        nzc = grid['nzc']
        rcf = grid['rcf']
        zcf = grid['zcf']
        midpoint = int(round(nzc/2))
        #zcf_tophalf = zcf[:,:,0:midpoint-1]
        #rcf_tophalf = rcf[:,:,0:midpoint-1]

        #tempg_tophalf = tempg[:,:,midpoint:-1]
        #siggas_tophalf = siggas[:,:,0:midpoint]
        
        print("zcf max " + str(np.max(zcf)))
        print("zcf min " + str(np.min(zcf)))
        #print("zcf tophalf max " + str(np.max(zcf_tophalf)))
        #print("zcf tophalf min " + str(np.min(zcf_tophalf)))
        #dz = (zcf_tophalf - np.roll(zcf_tophalf,1))#,axis=2))
        dz = (zcf - np.roll(zcf,1))#,axis=2))

        print("dz max " + str(np.max(dz)))
        print("dz min " + str(np.min(dz)))
        print("dz shape " + str(dz.shape))


        #compute rho structure
        rho0 = np.zeros((nac,nfc,nzc))
        sigint = siggas

        #compute gravo-thermal constant
        grvc = Disk.G*self.Mstar*Disk.m0/Disk.kB

        #t1 = time.clock()
        #differential equation for vertical density profile
        #dlnT = (np.log(tempg_tophalf)-np.roll(np.log(tempg_tophalf),1,axis=2))/dz
        dlnT = (np.log(tempg)-np.roll(np.log(tempg),1,axis=2))/dz
    
        #dlnp = -1.*grvc*zcf_tophalf/(tempg_tophalf*(rcf_tophalf**2+zcf_tophalf**2)**1.5)-dlnT
        dlnp = -1.*grvc*zcf/(tempg*(rcf**2+zcf**2)**1.5)-dlnT
        #dlnp[:,:,0] = -1.*grvc*zcf_tophalf[:,:,0]/(tempg_tophalf[:,:,0]*(rcf_tophalf[:,:,0]**2.+zcf_tophalf[:,:,0]**2.)**1.5)
        #dlnp[:,:,0] = -1.*grvc*zcf[:,:,0]/(tempg[:,:,0]*(rcf[:,:,0]**2.+zcf[:,:,0]**2.)**1.5)
        dlnp[:,:,midpoint] = -1.*grvc*zcf[:,:,midpoint]/(tempg[:,:,midpoint]*(rcf[:,:,midpoint]**2.+zcf[:,:,midpoint]**2.)**1.5)

        #numerical integration to get vertical density profile
        foo = dz*(dlnp+np.roll(dlnp,1,axis=2))/2.
        foo[:,:,midpoint] = np.zeros((nac,nfc))
        lnp = foo.cumsum(axis=2)

        #normalize the density profile (note: this is just half the sigma value!)
        rho0 = 0.5*((sigint/np.trapz(np.exp(lnp),zcf,axis=2))[:,:,np.newaxis]*np.ones(nzc))*np.exp(lnp)
        #rho0_tophalf = 0.5*((sigint/np.trapz(np.exp(lnp),zcf_tophalf,axis=2))[:,:,np.newaxis]*np.ones(midpoint-1))*np.exp(lnp)
        #rho0[:,:,0:midpoint-1] = rho0_tophalf.flatten()[::-1].reshape(nac,nfc,midpoint-1)
        #rho0[:,:,0:midpoint-1] = rho0_tophalf

        #rho0_bottomhalf = rho0_tophalf.flatten()[::-1].reshape(nac,nfc,midpoint-1)
        #rho0_bottomhalf = rho0_tophalf.flatten()[::-1].reshape(nac,nfc,midpoint-1)
        #rho0[:,:,midpoint:-1] = rho0_bottomhalf

        #t2=time.clock()
        #print("hydrostatic loop took {t} seconds".format(t=(t2-t1)))

        '''from disk.py'''
        
        #gaussian profile 
        #hr = np.sqrt(2*T[0]*rf[ir]**3./grvc)
        #dens = sigint[ir]/(np.sqrt(np.pi)*hr)*np.exp(-(z/hr)**2.)

        #print('Doing hydrostatic equilibrium')
        #t1 = time.clock()
        #for ia in range(nac):
        #    for jf in range(nfc):
        #
        #        #extract the T(z) profile at a given radius
        #        T = tempg[ia,jf]
        #
        #        z=zcf[ia,jf]
        #        #differential equation for vertical density profile
        #        dlnT = (np.log(T)-np.roll(np.log(T),1))/dz[ia,jf]
        #        dlnp = -1*grvc*z/(T*(rcf[ia,jf]**2.+z**2.)**1.5)-dlnT
        #        dlnp[0] = -1*grvc*z[0]/(T[0]*(rcf[ia,jf,0]**2.+z[0]**2.)**1.5)
        #
        #        #numerical integration to get vertical density profile
        #        foo = dz[ia,jf]*(dlnp+np.roll(dlnp,1))/2.
        #        foo[0] = 0.
        #        lnp = foo.cumsum()
        #
        #        #normalize the density profile (note: this is just half the sigma value!)
        #        #print(lnp.shape,grvc.shape,z.shape,T.shape,rcf[ia,jf].shape,dlnT.shape)
        #        dens = 0.5*sigint[ia,jf]*np.exp(lnp)/np.trapz(np.exp(lnp),z)
        #        rho0[ia,jf,:] = dens
        #        #if ir == 200:
        #        #    plt.plot(z/Disk.AU,dlnT)
        #        #    plt.plot(z/Disk.AU,dlnp)
        #t2=time.clock()
        #print("hydrostatic loop took {t} seconds".format(t=(t2-t1)))

        self.rho0=rho0
        #print(Disk.G,self.Mstar,Disk.m0,Disk.kB)
        if 0:
            print('plotting')
            plt.pcolor(rf[:,0,np.newaxis]*np.ones(nzc),zcf[:,0,:],np.log10(rho0[:,0,:]))
            plt.colorbar()
            plt.show()


    def density(self):
        'Return the density structure'
        return self.rho0

    def temperature(self):
        'Return the temperature structure'
        return self.tempg

    def grid(self):
        'Return an XYZ grid (but which one??)'
        return self.grid

    def get_params(self):
        params=[]
        params.append(self.qq)
        params.append(self.McoG/Disk.Msun)
        params.append(self.pp)
        params.append(self.Ain/Disk.AU)
        params.append(self.Aout/Disk.AU)
        params.append(self.Rc/Disk.AU)
        params.append(math.degrees(self.thet))
        params.append(self.Mstar/Disk.Msun)
        params.append(self.Xco)
        params.append(self.vturb/Disk.kms)
        params.append(self.zq0)
        params.append(self.tmid0)
        params.append(self.tatm0)
        params.append(self.handed)
        return params

    def get_obs(self):
        obs = []
        obs.append(self.nr)
        obs.append(self.nphi)
        obs.append(self.nz)
        obs.append(self.zmax/Disk.AU)
        return obs

    def plot_structure(self,sound_speed=False,beta=None,dust=False,rmax=500,zmax=170):
        ''' Plot temperature and density structure of the disk'''



        '''theo: adding some extra plots and print statements to make sure warp structure is correct'''
        #print(self.rcf.shape)
        #print(self.rotation[:,-1,0])

        #plt.scatter(self.rotation[:,:,0], self.rotation[:,:,1], c=self.rotation[:,:,2])
        #plt.show()


        #fig, ax = plt.subplots()
        #plt.scatter(self.rcf[:,:,0], self.pcf[:,:,0], c=self.zcf[:,:,0])
        #ax.set_xlabel("radius")
        #ax.set_ylabel("phi (radians)")
        #plt.show()

        #plt.scatter(self.r_grid, self.f_grid)
        '''end changes'''


        plt.figure()
        plt.rc('axes',lw=2)
        cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,0.1))
        cs2 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[int(self.nphi/2),:,:])+4,np.arange(0,11,0.1))
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        if sound_speed:
            cs = self.r*self.Omg#np.sqrt(2*self.kB/(self.Da*self.mCO)*self.T)
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[0,:,:]/Disk.kms,100,colors='k')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[int(self.nphi/2.),:,:]/Disk.kms,100,colors='k')
            plt.clabel(cs3)
        elif beta is not None:
            cs = np.sqrt(2*self.kB/(self.Da*self.mu)*self.T)
            rho = (self.rhoG+4)*self.mu*self.Da #mass density
            Bmag = np.sqrt(8*np.pi*rho*cs**2/beta) #magnetic field
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[0,:,:]),20)
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[int(self.nphi/2.),:,:]),20)
        elif dust:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[0,:,:]),100,colors='k',linestyles='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[int(self.nphi/2.),:,:]),100,colors='k',linestyles='--')
        else:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[0,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[int(self.nphi/2.),:,:],(20,40,60,80,100,120),colors='k',ls='--')
            plt.clabel(cs3,fmt='%1i')
        plt.colorbar(cs2,label='log n')
        plt.xlim(-1*rmax,rmax)
        plt.ylim(0,zmax)
        plt.xlabel('R (AU)',fontsize=20)
        plt.ylabel('Z (AU)',fontsize=20)
        plt.show()

    def calcH(self,verbose=True):
        ''' Calculate the equivalent of the pressure scale height within our disks. This is useful for comparison with other models that take this as a free parameter. H is defined as 2^(-.5) times the height where the density drops by 1/e. (The factor of 2^(-.5) is included to be consistent with a vertical density distribution that falls off as exp(-z^2/2H^2))'''
        ###### this method does not work with the elliptical disk (must expand to 3d) ######
        nrc = self.nrc
        zf = self.zf
        rf = self.rf
        rho0 = self.rho0

        H = np.zeros(nrc)
        for i in range(nrc):
            rho_cen = rho0[i,0]
            diff = abs(rho_cen/np.e-rho0[i,:])
            H[i] = zf[(diff == diff.min())]/np.sqrt(2.)

        if verbose:
            H100 = np.interp(100*Disk.AU,rf,H)
            psi = (np.polyfit(np.log10(rf),np.log10(H),1))[0]
            #print(H100/Disk.AU)
            #print(psi)
            print('H100 (AU): {:.3f}'.format(H100/Disk.AU))
            print('power law: {:.3f}'.format(psi))

        return H

