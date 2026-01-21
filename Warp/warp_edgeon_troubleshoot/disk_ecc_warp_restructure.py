# Define the Disk class. This class of objects can be used to create a disk structure. Given parameters defining the disk, it calculates the desnity structure under hydrostatic equilibrium and defines the grid used for radiative transfer. This object can then be fed into the modelling code which does the radiative transfer given this structure.

#two methods for creating an instance of this class

# from disk import *
# x=Disk()

# import disk
# x = disk.Disk()

# For testing purposes use the second method. Then I can use reload(disk) after updating the code
import math
#from mol_dat import mol_dat
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from astropy import constants as const
from scipy.special import ellipk,ellipe
from scipy.integrate import trapz

from scipy.interpolate import LinearNDInterpolator as interpnd

#testing time
import time

'''method from Andres Zuleta et al. 2024
paper: ui.adsabs.harvard.edu/abs/2024A%26A...692A..56Z/abstract
github repo: https://github.com/andres-zuleta/eddy/tree/warp_rf'''

'''My attempt at defining an axis rotation (spatial) with warp
I think the best thing would be to integrate this into the set_structure function
goal is to define initial grid, warp it, make it 3d in a way that works with the rest of the code'''

'''straightforward coordinate switching but i did steal it from here;
https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates'''

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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

#    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,-1,0,0,[.76,1000],[10,800]],obs=[180,131,300,170],rtg=True,vcs=True,line='co',ring=None):
    def __init__(self,q=-0.5,McoG=0.09,pp=1.,Ain=10.,Aout=1000.,Rc=150.,incl=51.5,
                 Mstar=2.3,Xco=1e-4,vturb=0.01,Zq0=33.9,Tmid0=19.,Tatm0=69.3,
                 handed=-1,ecc=0.,aop=0.,sigbound=[.79,1000],Rabund=[10,800],
                 nr=180,nphi=131,nz=300,zmax=170,rtg=True,vcs=True,line='co',ring=None,w_i=10, w_r0=10,w_dr=10,w_pa=10):
        #not sure if I'm doing this right, but adding some parameters here for warp
        params=[q,McoG,pp,Ain,Aout,Rc,incl,Mstar,Xco,vturb,Zq0,Tmid0,Tatm0,handed,ecc,aop,sigbound,Rabund,w_i,w_r0,w_dr,w_pa]
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
        nac = 500#256             # - number of unique a rings
        #nrc = 256             # - numver of unique r points
        amin = self.Ain       # - minimum a [AU]
        amax = self.Aout      # - maximum a [AU]
        e = self.ecc          # - eccentricity
        nzc = int(2.5*nac)#nac*5           # - number of unique z points
        '''defining z-array: .1 AU to specified max value in AU. logarithmic. number specified by
        # of annuli'''
        zmin = .1*Disk.AU      # - minimum z [AU]
        #zmin = -self.zmax      # - minimum z [AU]
        nfc = self.nphi       # - number of unique f points
        '''putting into linspace for now to compare to warp model'''
        af = np.linspace(amin,amax,nac)
        zf = np.linspace(zmin,self.zmax,nzc)
        #zf = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)

        #adding this to triple check z-dimension is doing what I think it is
        #print("1d z-array " + str(zf))

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
        #zcf = (np.outer(ida,idf))[:,:,np.newaxis]*zf
        #pcf = (np.outer(ida,pf))[:,:,np.newaxis]*idz
        fcf = (pcf - self.aop) % (2*np.pi)
        '''
        plt.imshow(fcf[:,:,0])
        plt.title("fcf")
        plt.colorbar()
        plt.savefig("warp_fcf.jpg")
        plt.show()
        '''
        #acf = (np.outer(af,idf))[:,:,np.newaxis]*idz
        '''
        plt.imshow(zcf[0,:,:])
        plt.title("zcf")
        plt.colorbar()
        plt.savefig("original_zcf.jpg")
        plt.show()

        plt.imshow(zcf[10,:,:])
        plt.title("zcf p=10 index slice")
        plt.colorbar()
        plt.savefig("original_zcf.jpg")
        plt.show()
        '''
        '''should be 0 grid in shape of radius, phi, z above midplane'''
        rcf=rf[:,:,np.newaxis]*idz
        #print(str(rcf.shape))
        #print("coords init {t}".format(t=time.clock()-tst))

        if 0:
            print('plotting')
            plt.plot((rcf*np.cos(fcf)).flatten(),(rcf*np.sin(fcf)).flatten())
            plt.show()

        # rcf[0][:] = radius at all z for first radial bin
        # zcf[0][:] = z in first radial bin

        # Here introduce new z-grid (for now just leave old one in)

        # Interpolate dust temperature and density onto cylindrical grid
        ###### doesnt seem to be used anywhere ######
        #tf = 0.5*np.pi-np.arctan(zcf/rcf)  # theta values
        #rrf = np.sqrt(rcf**2.+zcf**2)

        # bundle the grid for helper functions
        ###### add angle to grid? ######
        '''nac, nfc, nzc are resolution (int) in each dimension. rcf is 0s grid in 3d. 
        amax is max a (AU), zcf is z meshgrid'''
        grid = {'nac':nac,'nfc':nfc,'nzc':nzc,'rcf':rcf,'amax':amax,'zcf':zcf}#'ff':ff,'af':af,
        self.grid=grid

        #print("grid {t}".format(t=time.clock()-tst))
        #define temperature structure
        # use Dartois (03) type II temperature structure
        ###### expanding to 3D should not affect this ######
        '''using debris disk parameters, setting tmid = tatm, and using 100 AU instead of 150 AU'''

        delta = 1.                # shape parameter
        rcf100=rcf/(100.*Disk.AU)
        rcf100q=rcf100**self.qq
        
        '''# zq0 = Zq, in AU, at 150 AU (????)'''
        '''zq should be 3d and scalled by 150 AU...???'''
        zq = self.zq0*Disk.AU*rcf100**1.3
        #zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.1
        tmid = self.tmid0*rcf100q
        tatm = tmid
        #tatm = self.tatm0*rcf100q
        tempg = tatm + (tmid-tatm)*np.cos((np.pi/(2*zq))*zcf)**(2.*delta)

        '''ii is 3d boolean grid of z values above some critical value'''
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
        dsdth = (acf[:,:,0]*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e))/(1+e*np.cos(fcf[:,:,0]))**2
        siggas = ((siggas_r*np.sqrt(1.-e*e))/(2*np.pi*acf[:,:,0]*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e)))*dsdth

        #print("siggas shape: "+ str(siggas.shape))
        '''
        plt.imshow(siggas)
        plt.title("siggas")
        plt.colorbar()
        plt.savefig("nowarp_siggas.jpg")
        plt.show()
        '''

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

        #print("hydro done {t}".format(t=time.clock()-tst))
        #Calculate radial pressure differential
        ### nolonger use pressure term ###
        #Pgas = Disk.kB/Disk.m0*self.rho0*tempg
        #dPdr = (np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)
        #print(dPdr[:5,0,0],dPdr[200:205,0,500])
        #dPdr = 0#(np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)


        #Calculate velocity field
        #Omg = np.sqrt((dPdr/(rcf*self.rho0)+Disk.G*self.Mstar/(rcf**2+zcf**2)**1.5))
        #w = np.isnan(Omg)
        #if w.sum()>0:
        #    Omg[w] = np.sqrt((Disk.G*self.Mstar/(rcf[w]**2+zcf[w]**2)**1.5))

        #https://pdfs.semanticscholar.org/75d1/c8533025d0a7c42d64a7fef87b0d96aba47e.pdf
        #Lovis & Fischer 2010, Exoplanets edited by S. Seager (eq 11 assuming m2>>m1)
        self.vel = np.sqrt(Disk.G*self.Mstar/(acf*(1-self.ecc**2.)))*(np.cos(self.aop+fcf)+self.ecc*self.cosaop)

        '''
        plt.imshow(self.vel[:,:,0])
        plt.title("vel")
        plt.colorbar()
        plt.savefig("nowarp_vel.jpg")
        plt.show()
        '''
        ###### Major change: vel is linear not angular ######
        #Omk = np.sqrt(Disk.G*self.Mstar/acf**3.)#/rcf
        #velrot = np.zeros((3,nac,nfc,nzc))
        #x,y velocities with refrence to semimajor axis (f)
        #velx = (-1.*Omk*acf*np.sin(fcf))/np.sqrt(1.-self.ecc**2)
        #vely = (Omk*acf*(self.ecc+np.cos(fcf)))/np.sqrt(1.-self.ecc**2)
        #x,y velocities with refrence to sky (phi) only care about Vy on sky
        #velrot[0] = self.cosaop*vel[0] - self.sinaop*vel[1]
        #velrot = self.sinaop*velx + self.cosaop*vely

        # Check for NANs
        ### nolonger use Omg ###
        #ii = np.isnan(Omg)
        #Omg[ii] = Omk[ii]
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
        '''
        print("zpht_up min " +str(np.min(zpht_up)))
        print("zpht_up max " +str(np.max(zpht_up)))
        print("zpht_up mean " +str(np.mean(zpht_up)))

        plt.imshow(self.rho0[:,:,0])
        plt.title("rho0 bottom slice")
        plt.colorbar()
        plt.show()

        plt.imshow(self.rho0[:,:,-1])
        plt.title("rho0 top slice")
        plt.colorbar()
        plt.show()
        
        plt.imshow(sig_col[:,:,0])
        plt.title("sig_col bottom slice")
        plt.colorbar()
        plt.show()

        plt.imshow(sig_col[:,:,-1])
        plt.title("sig_col top slice")
        plt.colorbar()
        plt.show()
        '''
        self.sig_col = sig_col
        '''
        print("sig_col min " +str(np.min(sig_col)))
        print("sig_col max " +str(np.max(sig_col)))
        print("sig_col_up mean " +str(np.mean(sig_col)))
        '''
        #szpht = zpht
        #print("Zpht {t} seconds".format(t=(time.clock()-tst)))

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
        self.tempg = tempg
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
        '''do i need the -1 or did I add that... I could put in aop here, but 
        I'm going to rely on twist_i to define rotation.'''
        #phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        #phi = np.linspace(0, 2*np.pi, self.nphi)
        #print("nphi " + str(self.nphi))
        #z_l = self.zf
        #z_l = np.arange(self.nz)/self.nz*(-self.zmax)+s/2.
        z_l = np.linspace(-self.zmax,self.zmax,self.nz)
        #print("nz " + str(self.nz))


        '''trying meshgridding first before converting to polar to match
        set_structure method'''

        R_mesh, phi_mesh, Z = np.meshgrid(R, phi, z_l)
        #print("grid shape " + str(phi_mesh.shape))
        #fcf = (pcf - self.aop) % (2*np.pi)
        #phi_mesh_aop = (phi_mesh - self.aop) % (2*np.pi)
        '''
        plt.pcolor(R_mesh[:,:,0], phi_mesh[:,:,0], Z[:,:,0])
        plt.title('unwarped rt_polar grid')
        plt.colorbar()
        plt.show()
        '''
        '''converting to xy'''

        warp_rt  = w_func(self, R, type="w")
        twist_rt = w_func(self, R, type="pa")

        X, Y = pol2cart(R_mesh, phi_mesh)

        print("X_0 " + str(X[0,0,0]))
        print("X_-1 " + str(X[0,-1,0]))

        print("Y_0 " + str(Y[0,0,0]))
        print("Y_-1 " + str(Y[0,-1,0]))

        print("X shape " +str(X.shape))
        print("Y shape " +str(Y.shape))
        print("Z shape " +str(Z.shape))



        #print("Rmax & min " + str(np.max(R)) + str(np.min(R)))
        #print("phi max & min " + str(np.max(phi)) + str(np.min(phi)))

        #phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)


        '''warp code'''
        #inc=0
        #pa=0

        #r_grid = self.acf
        #f_grid = self.pcf
        #z_grid = self.zcf

        #r_i = acf[:,0,0]  #1d array of radius values

        '''defining warp, taking parmas from input into Disk'''
        '''defines change in inclination'''
        #warp_i  = w_func(self, R, type="w")
        '''defines twist'''
        #twist_i = w_func(self, R, type="pa")

        '''may actually have to also interpolate x and y warp grids onto these ones...'''
        #X = (np.outer(R,np.cos(phi)))
        #Y = (np.outer(R,np.sin(phi)))
        #X = (np.outer(R,np.cos(phi))).transpose()
        #Y = (np.outer(R,np.sin(phi))).transpose()
        #Z = np.zeros(X.shape)
        #x_l, y_l = pol2cart(R,phi)
        #z_l = self.zf

        '''need 3d inputs, so I'm using meshgrid'''

        #X,Y,Z = np.meshgrid(x_l, y_l, z_l)
        #X, Y = pol2cart(R_mesh, phi_mesh)


        #inc_obs = np.deg2rad(0)
        #PA_obs = np.deg2rad(pa)
        #PA_obs = np.deg2rad(0)

        '''need cartesian system for warp rotation'''
        #xi = acf[:,:,0] * np.cos(pcf[:,:,0])
        #print(xi.shape)
        #yi = acf[:,:,0] * np.sin(pcf[:,:,0])

        '''what if I just use original grid...?'''



        '''reshaping to play nice with rotational matrix'''
        '''
        plt.pcolor(X[:,:,0], Y[:,:,0], Z[:,:,0])
        plt.title('zcf')
        plt.colorbar()
        plt.show()
        '''
        #points_i = np.moveaxis([X, Y, Z], 0, 2)
        #print("points_i.shape" + str(points_i.shape))

        '''applying warp via rotational matrix'''
        #rotation = apply_matrix2d_d(points_i, warp_i, twist_i, inc_obs, PA_obs)
        #velocity = apply_matrix2d_d(vkep_i, warp_i, twist_i, inc_obs, PA_obs)
        #self.rotation = rotation
        '''now we have warped disk, rotation is a a stack of 3 2d array with[:,:,0]=x coord, [:,:,1]=y coord, [:,;,2]=z coord'''
        X_w, Y_w, Z_w = matrix_mine_rt(X, Y, Z, warp_rt, twist_rt,0,0)

        '''
        plt.pcolor(X_w[:,:,0], Y_w[:,:,0], Z_w[:,:,0])
        plt.title("z after warp")
        plt.colorbar()
        plt.show()
        '''
        '''now to make 3d grid:'''

        '''saving these because I think I can use them in rt grid'''
        #self.x_rt_grid = rotation[:,:,0].transpose()
        #self.y_rt_grid = rotation[:,:,1].transpose()
        #self.z_rt_grid = rotation[:,:,2].transpose()

        '''keeping z grid, adding each point in zf to each slice
        this doesn't perfectly replicate geoemetry of warp: 
        I think we would need to apply a rotational matrix to each z slice'''

        nac = 500#256             # - number of unique a ring
        #nzc = int(5*nac)#nac*5           # - number of unique z points

        '''trying to do whole z grid here. May run into problem interpolating onto negative
        z values for rho0 and vel'''
        #zf = np.linspace(-self.zmax, self.zmax, self.nz)

        '''translating x &y grids back to polar coordinates'''

        #x_full_rt_grid = self.x_rt_grid[:,:,np.newaxis]*np.ones(self.nz)
        #y_full_rt_grid = self.y_rt_grid[:,:,np.newaxis]*np.ones(self.nz)
        #z_full_rt_grid = self.z_rt_grid[:,:,np.newaxis] + zf
        '''
        plt.pcolor(self.x_rt_grid, self.y_rt_grid, z_full_rt_grid[:,:,0])
        plt.title("first z slice (rt grid)")
        plt.colorbar()
        plt.show()


        plt.pcolor(self.x_rt_grid, self.y_rt_grid,z_full_rt_grid[:,:,-1])
        plt.title("last z slice (rt grid)")
        plt.colorbar()
        plt.show()
        '''
        
        '''converting back to polar coordinates'''
        #r_grid, p_grid = cart2pol(X_w, Y_w)
        #self.r_grid = r_grid
        #self.p_grid = p_grid


        '''making r & phi grid 3d along z axis'''
        #r_full_grid = r_grid[:,:, np.newaxis]+np.ones(len(zf))
        #f_full_grid = f_grid[:,:, np.newaxis]+np.ones(len(zf))

        #x_full_grid = self.x_grid*np.ones(len(zf))
        #y_full_grid = self.y_grid*np.ones(len(zf))

        '''just going to try this; fcf right now is from -pi to pi, try adding pi to match unedited fcf'''

        #foo = np.floor(self.nz/2)

        #S_old = np.concatenate([Smid+Smin-10**(np.log10(Smid)+np.log10(Smin/Smid)*np.arange(foo)/(foo)),Smid-Smin+10**(np.log10(Smin)+np.log10(Smid/Smin)*np.arange(foo)/(foo))])
        #S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin #*** not on cluster**

        #print("grid {t}".format(t=time.clock()-tst))
        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        #S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        # arrays in [phi,r,s] on sky coordinates
        

        '''interped z grid'''
        #zinterp = interpnd((np.ravel(self.x_grid), np.ravel(self.y_grid)), np.ravel(self.z_grid))
        #Z = zinterp(np.ravel(X), np.ravel(Y))

        #Use a rotation matrix to transform between radiative transfer grid and physical structure grid

        '''try to remove z-mirroring '''
        #if np.abs(self.thet) > np.arctan(self.Aout*(1+self.ecc)/self.zmax):
            #zsky_max = np.abs(2*self.Aout*(1+self.ecc)/self.sinthet)
        #else:
            #zsky_max = 2*(self.zmax/self.costhet)

        '''z_mirror happens here. The problem is that it uses a 1d (not 2d) array to do iit
        because it assumes azimuthal symmetry...'''
        #zsky = np.arange(self.nz)/self.nz*(-zsky_max)+zsky_max/2.
        #print("zsky shape" + str(zsky.shape))
        #print("zsky min" +str(np.min(zsky)))
        #print("zsky max" +str(np.max(zsky)))

        '''trying new z definition using 2d grid. need to think about how to 
        project z axis onto sky.'''

        #if np.abs(self.thet) > np.arctan(self.Aout*(1+self.ecc)/self.zmax):
            #diskZ = Z*self.sinthet
        #else:
            #diskZ = Z*self.costhet

        #tdiskZ = diskZ.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)+zsky*self.costhet


        '''maybe this is where the z mirroring is happening?'''
        '''maybe if I interpolate z grid onto this resolution and them add zsky...'''
        
        #tdiskZ_og = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        #print("tdiskZ shape" + str(tdiskZ.shape))
        #print("tdiskZ min " + str(np.min(tdiskZ)))
        #print("tdiskZ max " + str(np.max(tdiskZ)))

        '''
        plt.pcolor(X, Y, tdiskZ_og[:,:,0])
        plt.title("tdiskZ original cart")
        plt.colorbar()
        plt.show()
        '''

        #X_w = self.x_grid
        #Y_w = self.y_grid
        #Z_w = self.z_grid



        #tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet
        '''based on rotational matrix'''

        '''no idea if this will work but trying Kevin's zsky manipulation on Z_w'''

        if np.abs(self.thet) > np.arctan(self.Aout*(1+self.ecc)/self.zmax):
            #zsky_w = np.abs(Z_w/self.sinthet)
            zsky_w = Z_w/self.sinthet
            print("werid zsky_w True")
        else:
            zsky_w = (Z_w/self.costhet)
            zsky = (Z/self.costhet)
            print("normal zsky_w True")


        plt.pcolor(X_w[:,:,0], Y_w[:,:,0], Z_w[:,:,0])
        plt.colorbar()
        plt.title("Warp, disk coordinates")
        plt.show()

        plt.pcolor(X_w[:,:,0], Y_w[:,:,0], zsky_w[:,:,0])
        plt.colorbar()
        plt.title("warp zsky")
        plt.show()
        #tdiskY = (Y_w*self.costhet - Z_w*self.sinthet)
        #tdiskZ = (Y_w*self.sinthet + Z_w*self.costhet)
        #no way this works

        tdiskY = (-Y_w*self.costhet + zsky_w*self.sinthet)
        tdiskZ = (-Y_w*self.sinthet - zsky_w*self.costhet)

        tdiskY_unwarped = (-Y*self.costhet + zsky*self.sinthet)
        tdiskZ_unwarped = (-Y*self.sinthet - zsky*self.costhet)

        plt.pcolor(X_w[:,:,0], tdiskY[:,:,0], tdiskZ[:,:,0])
        plt.pcolor(X_w[:,:,-1], tdiskY[:,:,-1], tdiskZ[:,:,-1])
        plt.colorbar()
        plt.title("Warp on sky")
        plt.xlim(-4.5e15, 4.5e15)
        plt.ylim(-4.5e15, 4.5e15)
        plt.show()

        plt.pcolor(X[:,:,0], tdiskY_unwarped[:,:,0], tdiskZ_unwarped[:,:,0])
        plt.pcolor(X[:,:,-1], tdiskY_unwarped[:,:,-1], tdiskZ_unwarped[:,:,-1])
        plt.colorbar()
        plt.title("No Warp on sky")
        plt.xlim(-4.5e15, 4.5e15)
        plt.ylim(-4.5e15, 4.5e15)
        plt.show()

        tdiskY_nosky = (Y_w*self.costhet - Z_w*self.sinthet)
        tdiskZ_nosky = (Y_w*self.sinthet + Z_w*self.costhet)
        '''
        plt.pcolor(X_w[:,:,-1], tdiskY_nosky[:,:,-1], tdiskZ_nosky[:,:,-1])
        plt.pcolor(X_w[:,:,0], tdiskY_nosky[:,:,0], tdiskZ_nosky[:,:,0])
        plt.colorbar()
        plt.title("Warp on sky, og tdiskZ")
        plt.xlim(-4.5e15, 4.5e15)
        plt.ylim(-4.5e15, 4.5e15)
        plt.show()
        '''
        '''
        plt.imshow(tdiskY[:,:,0])
        plt.title("tdiskY[:,:,0]")
        plt.colorbar()
        plt.show()

        plt.imshow(tdiskY[:,:,-1])
        plt.title("tdiskY[:,:,-1]")
        plt.colorbar()
        plt.show()
        '''
        plt.imshow(tdiskZ[:,:,0])
        plt.title("tdiskZ[:,:,0]")
        plt.colorbar()
        plt.show()

        plt.imshow(tdiskZ[:,:,-1])
        plt.title("tdiskZ[:,:,-1]")
        plt.colorbar()
        plt.show()
        '''

        del_tdiskY = tdiskY[:,:,-1]-tdiskY[:,:,0]
        plt.imshow(del_tdiskY)
        plt.colorbar()
        plt.title("disk on sky thickness")
        plt.show()

        print("del_tdiskY 1 " + str(del_tdiskY[0,0]))
        print("del_tdiskY 2 " + str(del_tdiskY[10,10]))
        print("del_tdiskY 3 " + str(del_tdiskY[100,100]))
        
        '''
        #tdiskZ = (Z.repeat(self.nz).reshape(self.nphi,self.nr,self.nz) - tdiskY*self.sinthet)*self.costhet
        #print("tdiskZ shape" + str(tdiskZ.shape))
        print("tdiskZ min " + str(np.min(tdiskZ)))
        print("tdiskZ max " + str(np.max(tdiskZ)))
        '''
        plt.pcolor(tdiskY[:,:,0], tdiskZ[:,:,0], X_w[:,:,0])
        plt.pcolor(tdiskY[:,:,-1], tdiskZ[:,:,-1], X_w[:,:,0])
        plt.colorbar()
        plt.title("sky grid yz plane")
        plt.show()

        plt.pcolor(X_w[:,:,-1], tdiskY[:,:,-1], tdiskZ[:,:,-1])
        plt.title("Disk in sky plane")
        plt.xlim(-5e15, 5e15)
        plt.ylim(-5e15, 5e15)
        #plt.show()
    

        plt.pcolor(X_w[:,:,0], tdiskY[:,:,0], tdiskZ[:,:,0])
        #plt.xlim(-5e15, 5e15)
        #plt.ylim(-5e15, 5e15)
        plt.colorbar()
        plt.show()
        '''
        
        if (self.thet<np.pi/2) & (self.thet>0):
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<self.thet)] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<self.thet)])/self.sinthet)
        elif self.thet>np.pi/2:
            theta_crit = np.arctan((self.Aout*(1+self.ecc)+tdiskY)/(self.zmax+tdiskZ))  
            S = -(self.zmax+tdiskZ)/self.costhet
            S[(theta_crit<(np.pi-self.thet))] = ((self.Aout*(1+self.ecc)+tdiskY[(theta_crit<(np.pi-self.thet))])/self.sinthet)
        elif (self.thet<0) & (self.thet>-np.pi/2):
            theta_crit = np.arctan((self.Aout*(1+self.ecc)-tdiskY)/(self.zmax-tdiskZ))
            S = (self.zmax-tdiskZ)/self.costhet
            S[(theta_crit<np.abs(self.thet))] = -((self.Aout*(1+self.ecc)-tdiskY[(theta_crit<np.abs(self.thet))])/self.sinthet)

        
        #S = np.ones(tdiskZ.shape)
        top_zslice = tdiskZ[:,:,-1]

        #S = tdiskZ - top_zslice[:,:,np.newaxis]*np.ones(self.nz)
        #S = (self.zmax-tdiskZ)/self.costhet
        #print("zmax "  + str(self.zmax))
        #print("tdiskZmax " + str(np.max(tdiskZ)))
        '''
        plt.imshow(S[:,:,0])
        plt.title("S bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(S[:,:,-1])
        plt.title("S top of disk")
        plt.colorbar()
        plt.show()
        '''
        '''for passing through disk face:
        I want to use shape of ellipse in [:,:,-1] plane as a mask
        x, y coordinates that fall outside of mask will pass through edge of disk'''
        
        '''To define ellipse:'''
        #x_center = np.mean(X_w[:,:,-1].ravel())
        #y_center = np.mean(tdiskY[:,:,-1].ravel())


        #S[(X_w<X_w[:,:,0])&(X_w>X_w[:,:,0])]
        
        # transform grid to disk coordinates
        #tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S
        #if self.thet > np.arctan(self.Aout/self.zmax):
        #    tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tdiskY = ytop - self.sinthet*S + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        #tr = np.sqrt(X_w.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        #tr = np.sqrt(X_w.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        tr = np.sqrt(X_w**2+tdiskY**2)
        tr_unwarped = np.sqrt(X**2+tdiskY_unwarped**2)
        #tphi = np.arctan2(tdiskY,X_w.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))%(2*np.pi)
        tphi = np.arctan2(tdiskY,X_w)%(2*np.pi)
        tphi_unwarped = np.arctan2(tdiskY_unwarped,X)%(2*np.pi)

        '''
        plt.imshow(tdiskZ[:,:,0])
        plt.title("tdiskZ")
        plt.colorbar()
        plt.show()


        plt.imshow(tphi[:,:,0])
        plt.title("tphi")
        plt.colorbar()
        plt.savefig("warp_tphi.jpg")
        plt.show()

        plt.imshow(tr[:,:,0])
        plt.title("tr")
        plt.colorbar()
        plt.savefig("warp_tr.jpg")
        plt.show()
        '''
        ###### should be real outline? requiring a loop over f or just Aout(1+ecc)######
        notdisk = (tr > self.Aout*(1.+self.ecc)) | (tr < self.Ain*(1-self.ecc))  # - individual grid elements not in disk
        isdisk = (tr>self.Ain*(1-self.ecc)) & (tr<self.Aout*(1+self.ecc)) & (np.abs(tdiskZ)<self.zmax)
        S -= S[isdisk].min() #Reset the min S to 0
        #xydisk =  tr[:,:,0] <= self.Aout*(1.+self.ecc)+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        self.r = tr




        #print("new grid {t}".format(t=time.clock()-tst))
        # Here include section that redefines S along the line of sight
        # (for now just use the old grid)

        # interpolate to calculate disk temperature and densities
        #print('interpolating onto radiative transfer grid')
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        # x is xy plane, y is z axis
        ###### rf is 2d, zf is still 1d ######
        #xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        #yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        #indices in structure arrays of coordinates in transform grid`


        zf_new = np.linspace(-self.zmax, self.zmax, self.nzc)
        z_w_skymax = np.max(zsky_w)
        zf_sky = np.linspace(-z_w_skymax, z_w_skymax, self.nzc)

        zind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        zind_unwarped = np.interp(np.abs(tdiskZ_unwarped).flatten(),self.zf,range(self.nzc)) #zf,nzc
        #print("nzc " + str(self.nzc))

        #print("flattened tdiskZ first 10 " + str(tdiskZ[0:10]))
        #print("flattened tdiskZ last 10 " + str(tdiskZ[-10:-1]))

        print("zf first 10 " + str(self.zf[0:10]))
        print("zf last 10 " + str(self.zf[-10:-1]))
        
        #zind = np.interp(tdiskZ.flatten(),zf_new,range(self.nzc)) #zf,nz

        zind_shaped = zind.reshape(tdiskZ.shape)
        zind_unwarped_shaped = zind_unwarped.reshape(tdiskZ_unwarped.shape)

        plt.imshow(zind_shaped[:,:,0])
        plt.title("zind shaped")
        plt.colorbar()
        plt.show()

        plt.imshow(zind_unwarped_shaped[:,:,0])
        plt.title("zind unwarped shaped")
        plt.colorbar()
        plt.show()
        #zind = np.interp(tdiskZ.flatten(),z_l,range(self.nz)) #zf,nzc
        #zind = np.interp(tdiskZ.flatten(),self.zf,range(self.nzc)) #zf,nzc
        #zind = np.interp(tdiskZ_nosky.flatten(),self.zf,range(self.nzc)) #zf,nzc
        
        print("zind max " + str(np.max(zind)))
        print("zind min " + str(np.min(zind)))
        print("zind len " + str(len(zind)))
        
        #zind = self.zf
        phiind = np.interp(tphi.flatten(),self.pf,range(self.nphi))
        phiind_unwarped = np.interp(tphi_unwarped.flatten(),self.pf,range(self.nphi))

        phiind_shaped = phiind.reshape(tphi.shape)
        phiind_unwarped_shaped = phiind_unwarped.reshape(tphi_unwarped.shape)


        plt.imshow(phiind_shaped[:,:,0])
        plt.title("phiind shaped")
        plt.colorbar()
        plt.show()

        plt.imshow(phiind_unwarped_shaped[:,:,0])
        plt.title("phiind unwarped shaped")
        plt.colorbar()
        plt.show()
        
        #print("phiind max " + str(np.max(phiind)))
        #print("phiind min " + str(np.min(phiind)))
        #print("phiind len " + str(len(phiind)))
    
        #phiind = self.pf
        aind = np.interp((tr.flatten()*(1+self.ecc*np.cos(tphi.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)
        aind_unwarped = np.interp((tr_unwarped.flatten()*(1+self.ecc*np.cos(tphi_unwarped.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)
        #aind = np.interp(tr.flatten(),self.af,range(self.nac),right=self.nac)

        aind_shaped = aind.reshape(tr.shape)
        aind_unwarped_shaped = aind_unwarped.reshape(tr_unwarped.shape)

        plt.imshow(aind_shaped[:,:,0])
        plt.title("aind shaped")
        plt.colorbar()
        plt.show()

        plt.imshow(aind_unwarped_shaped[:,:,0])
        plt.title("aind unwarped shaped")
        plt.colorbar()
        plt.show()

        
        print("aind max " + str(np.max(aind)))
        print("aind min " + str(np.min(aind)))
        print("aind len " + str(len(aind)))
        
        #aind = self.af

        #print("index interp {t}".format(t=time.clock()-tst))
        ###### fixed T,Omg,rhoG still need to work on zpht ######
        
        #tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        #print("tempg shape " + str(self.tempg.shape))
        #tT = ndimage.map_coordinates(self.tempg,[[phiind],[aind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        #tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(tdiskZ.shape)
        '''trying calculating t grids with unwarped coordinates, using warped coordinates to reference them'''
        tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)

        '''
        
        plt.pcolor(X_w[:,:,0], tdiskY_nosky[:,:,0], tT[:,:,0])
        plt.pcolor(X_w[:,:,-1], tdiskY_nosky[:,:,-1], tT[:,:,-1])
        plt.colorbar()
        plt.title("tT, top and bottom no sky")
        plt.show()
        '''
        plt.pcolor(X_w[:,:,0], tdiskY[:,:,0], tT[:,:,0])
        plt.pcolor(X_w[:,:,-1], tdiskY[:,:,-1], tT[:,:,-1])
        plt.colorbar()
        plt.title("tT, top and bottom")
        plt.show()
        #print("tT shape " + str(tT.shape))
        #Omgx = ndimage.map_coordinates(self.Omg0[0],[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgs
        #Omg = ndimage.map_coordinates(self.Omg0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgy
        #tvel = ndimage.map_coordinates(self.vel,[[phiind],[aind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #tvel = ndimage.map_coordinates(self.vel,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        
        plt.imshow(tT[:,:,0])
        plt.title("tT bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(tT[:,:,-1])
        plt.title("tT top of disk")
        plt.colorbar()
        plt.show()
        
        tvel = ndimage.map_coordinates(self.vel,[[aind],[phiind],[zind]],order=1,).reshape(self.nphi,self.nr,self.nz)

        plt.pcolor(X_w[:,:,0], tdiskY[:,:,0], tvel[:,:,0])
        plt.pcolor(X_w[:,:,-1], tdiskY[:,:,-1], tvel[:,:,-1])
        plt.title("tvel, top and bottom")
        plt.show()
        
        plt.imshow(tvel[:,:,0])
        plt.title("tvel bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(tvel[:,:,-1])
        plt.title("tvel top of disk")
        plt.colorbar()
        plt.show()
        
        #Omgz = np.zeros(np.shape(Omgy))
        #trhoG = Disk.H2tog*self.Xmol/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        #trhoH2 = trhoG/self.Xmol #** not on cluster**
        #zpht = np.interp(tr.flatten(),self.rf,self.zpht).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        '''
        print("in rt self.sig_col min " + str(np.min(self.sig_col)))
        print("in rt self.sig_col max " + str(np.max(self.sig_col)))
        print("in rt self.sig_col mean " + str(np.mean(self.sig_col)))

        plt.imshow(self.sig_col[:,:,0])
        plt.title("sig_col in rt bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(self.sig_col[:,:,500])
        plt.title("sig_col in rt top of disk 500")
        plt.colorbar()
        plt.show()

        plt.imshow(self.sig_col[:,:,100])
        plt.title("sig_col in rt top of disk 100")
        plt.colorbar()
        plt.show()

        plt.imshow(self.sig_col[:,:,25])
        plt.title("sig_col in rt top of disk 25")
        plt.colorbar()
        plt.show()

        plt.imshow(self.sig_col[:,:,-1])
        plt.title("sig_col in rt top of disk")
        plt.colorbar()
        plt.show()
        '''

        #tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(tdiskZ.shape)

        #zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        #zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz)
        tT[notdisk] = 0
        self.sig_col = tsig_col


        plt.imshow(zpht_up[:,:,0])
        plt.title("zpht_up bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(zpht_up[:,:,-1])
        plt.title("zpht_up top of disk")
        plt.colorbar()
        plt.show()
        
        
        plt.imshow(self.sig_col[:,:,0])
        plt.title("tsig_col bottom of disk")
        plt.colorbar()
        plt.show()

        plt.imshow(self.sig_col[:,:,-1])
        plt.title("tsig_col top of disk")
        plt.colorbar()
        plt.show()

        plt.pcolor(X_w[:,:,0], tdiskY[:,:,0], tsig_col[:,:,0])
        plt.pcolor(X_w[:,:,-1], tdiskY[:,:,-1], tsig_col[:,:,-1])
        plt.title("Cart tsig top & bottom")
        plt.colorbar()
        plt.show()
        

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
        trhoG = trhoH2*self.Xmol
        trhoH2[notdisk] = 0
        trhoG[notdisk] = 0
        self.rhoH2 = trhoH2

        #print("zap {t}".format(t=time.clock()-tst))
        #temperature and turbulence broadening
        #moved this to the set_line method
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs


        if 1:
            print('plotting')
            plt.figure(1)
            plt.subplot(211)
            plt.pcolor(np.log10(tr[0,:,:]),tdiskZ[0,:,:],np.log10(trhoG[0,:,:]))
            plt.colorbar()
            #plt.subplot(212)
            #plt.pcolor(self.rf[:,0,np.newaxis]*np.ones(256*5),(self.zf[:,np.newaxis]*np.ones(256)).T,np.log10(self.rho0[:,0,:])) #need to expand rf and zf to same dimensions as tempg
            #plt.colorbar()
            plt.show()

        # store disk

        '''originally, I think these parameters were X, Y, and tdiskZ?'''
        #self.X = X_w
        #self.Y = Y_w
        #self.X_unwarp = X
        #self.Y_unwarp = Y
        self.X = X
        self.Y = tdiskY_unwarped
        #self.tY = tdiskY
        #self.tY_unwarp = tdiskY_unwarped
        #self.tZ_unwarp = tdiskZ_unwarped
        self.Z = tdiskZ_unwarped
        self.S = S
        #self.r = tr
        self.T = tT
        #self.dBV = tdBV
        self.rhoG = trhoG
        #self.Omg = Omg#Omgy #need to combine omgx,y,z
        self.vel = tvel
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

