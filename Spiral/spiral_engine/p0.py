'''theo comment: modifying for 49 Ceti parameters'''

from astropy import units as u
import numpy as np
# names for subdirectories
CASA_DATA='CASA_DATA'
DMR_OUTPUTS='DMR_OUTPUTS'
MCMC_OUTPUTS='MCMC_OUTPUTS'
# prefixes for dmr-generated files 
DMR_PREFIX='DMR_RES'
INPUT_MODEL_DMR=f'{DMR_PREFIX}_PYTHON_model.fits'  
INPUT_RESID_DMR=f'{DMR_PREFIX}_PYTHON_resid.fits'

#  SOME REMINDERS
#  STOP! DID YOU RUN VAR VIS AND ADD WEIGHTS? 
#  STOP! DID YOU MAKE SURE TO RENAME YOUR FILES AFTER YOU RAN VAR VIS?
#  STOP! DID YOU CHANGE THE CSV FILE NAME?
#  STOP! DID YOU EXCLUDE ALL FLAGGED CHANNELS FROM YOUR CHI SQR?

#################################################

# ONLY edit VARIABLES below this line
# DO NOT try running dmr files with any additional print statements as dmr-script.sh read variables form stdout (i.e stuff printed in terminal)
# if you really need to add print statements do so in the 'playground' at the bottom of this file 
DISKNAME='49_Ceti'
IMFILE=f'{CASA_DATA}/SB_dirty.fits'
DATAFILE=f'{CASA_DATA}/sb_varvised.uvfits'
CSV_FILE=f'{DISKNAME}_2_24_MCMC.csv'
#################################################

# MUST FOLLOW convention below: f'{CASA_DATA}/<filename>.uvfits': f'{CASA_DATA}/<filename>.fits',
# NOTE: ALL .uvfits and .fits FILES MUST BE ON THE SAME VELOCITY GRID (i.e they should have the same velocities -- up to the ten-thousandths --
#       for all channels) 
ADDITIONAL_OBS={
 f'{CASA_DATA}/lb_spw0_trimmed.uvfits':f'{CASA_DATA}/LB_spw0_dirty.fits',
 f'{CASA_DATA}/lb_spw1_trimmed.uvfits':f'{CASA_DATA}/LB_spw1_dirty.fits',
 f'{CASA_DATA}/lb_spw2_trimmed.uvfits':f'{CASA_DATA}/LB_spw2_dirty.fits',
 f'{CASA_DATA}/lb_spw3_trimmed.uvfits':f'{CASA_DATA}/LB_spw3_dirty.fits',
 f'{CASA_DATA}/MB_spw0_varvised.uvfits':f'{CASA_DATA}/MB_spw0_dirty.fits',
 f'{CASA_DATA}/mb_spw1_varvised.uvfits':f'{CASA_DATA}/MB_spw1_dirty.fits'
 # ...
}
#################################################

# Ideally this is the same pixel size as used in tclean -- this is used for GALARIO tasks
PIXEL_SIZE=0.016

#################################################

# If you have contaminated channels input their indicies here (0-indexed) 
EXCLUDED_CHANNEL_IDXs=[] # ex: [4,5,8]
#################################################

# Use SIMBAD (to get these distance measurements)
# NOTE: I converted to arcsec, SIMBAD gives you measurements in milli-arcsec
# https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=HD152989&submit=SIMBAD+search
SIMBAD_parallax_arcsec=17.4725/1000
SIMBAD_parallax_uncertainty_arcsec=0.0547/1000
DISTANCE_PC=1/SIMBAD_parallax_arcsec

#################################################

# ONLY MODIFY THIS LIST BY ADDING PARAMETERS TO THE END 
# IF YOU MODIFY any of param_order, init_params, params_uncertainty, param_priors, or name_map
# THEN YOU MUST MODIFY ALL 
param_order=[
            "logmass_stell", 
            "incl", 
            "Rc", 
            "pp", 
            "pa", 
            "logmass",
            "Rin", 
            "xoff",
            "yoff",
            "vsys", 
            "tatm",
            'qq',

            #spiral params
            'massdisk',
            'pitch_angle',
            'rotation',
            'spiral_amp',
            #"new_parameter_name1",
            #...,
]

# the intitial parameters for your MCMC run (p0) 
init_params={
            "logmass_stell": 0.189,  
            "incl": 84.096, 
            "Rc": 123.08,  
            "pp": -3.41, 
            "pa": 109.1,    
            "logmass": -4.261, 
            #Caroline's best-fit to 0, but I'm worried about initializing with that value
            "Rin": 19.02,   
            "xoff": 0.0294,   
            "yoff": -0.045,  
            "vsys": 2.356, 
            "tatm": 12.52, 
            'qq':-4,

            #spiral_parmas
            'massdisk': 0.3,
            'pitch_angle': 20,
            'rotation': 60,
            'spiral_amp': 0.5,
            #"new_parameter_name1":new_parameter_initial_value,
            #..., 
}

# the widths of the parameter distributions (AKA "sigma" in previous mcmc scripts) 
params_uncertainty={
            "logmass_stell": 0.05,
            "incl": 1,
            "Rc": 0.5,
            "pp": 0.1,
            "pa": 1,
            "logmass": 0.05,
            "Rin": 1.0,
            "xoff": 0.05,
            "yoff": 0.05, 
            "vsys": 0.1,
            "tatm": 3,
            'qq':0.1,

            #spiral params
            'massdisk': 0.1,
            'pitch_angle': 5,
            'rotation': 10,
            'spiral_amp': 0.1,
            #"new_parameter_name1":new_parameter_uncertainty,
            #...,
}

param_priors={
            "logmass_stell":[-1, 1],
            "incl":[60, 90], # you should only ever restrict this range; do not make it larger as it's physically unreasonable
            "Rc":[0, 200],
            "pp":[-5, 5],
            "pa":[0, 360],
            "logmass":[-10, 0],
            "Rin":[0,100],
            "xoff":[-1,1],
            "yoff":[-1,1],
            "vsys":[2,3],
            "tatm":[0,500],
            'qq':[-5,5],

            #spiral params
            'massdisk':[0,1],
            'pitch_angle':[5,40],
            'rotation':[0,180],
            'spiral_amp':[0,1]
            #"new_parameter_name1":[new_parameter_lower_bound,new_parameter_upper_bound],
            #...,
}

name_map={
         'logmass_stell':r'log(M$_{star}$) [log(M$_{\rm \odot}$)]', 
         'incl':r'$incl$ [$^{\circ}$]', 
         'Rc':r'$R_{c}$ [au]', 
         'pp':r'$pp$', 
         'pa':r'PA [$^{\circ}$]', 
         'logmass':r'log(M$_{disk}$) [log(M$_{\rm \odot}$)]',
         'Rin':r'$R_{in} [au]$',
         'xoff':r'x$_{off}$', 
         'yoff':r'y$_{off}$',
         'vsys':r'$v_{sys} [km s^{-1}]$',
         'tatm':r'$T_{atm}$ [K]',
         'qq':r'$qq$',
         #"new_parameter_name1":r'$newparametername_{1}',
         #...,

         #spiral params
         'massdisk':r'Disk Mass [M$_{\rm \odot}]$',
         'pitch_angle':r'Pitch Angle [$^{\circ}$]',
         'rotation':r'Spiral Rotation [$^{\circ}$]',
         'spiral_amp':r'Surface Density Perturbation Amplitude'
}

#################################################

# MCMC chain configurations
mcmc_hyper_params = {
            "nwalkers": 25, 
            "nsteps": 5000,
            "ndim": len(init_params),
            "restart": False
}
#################################################

# NEED this print statement to have dmr-script read in variables
print(f"{DATAFILE} {PIXEL_SIZE} {INPUT_MODEL_DMR} {INPUT_RESID_DMR} {DMR_OUTPUTS} {DMR_PREFIX} {mcmc_hyper_params['nwalkers']}") 

#################################################

# the code below will only be executed if this file is run directly (as opposed to calling it from another file as done in mcmc_trigger.py and dmr.py
# to run file directly enter python p0.py in the engine/. directory 
if __name__ == '__main__':
   # only add print statements here ... this is your playground 
   print(init_params)
   print(param_order)
