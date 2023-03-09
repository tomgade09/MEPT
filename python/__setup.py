import os, time, shutil

from __simulationvariables import *

#Setting up folders, changing directory
def setupFolders():
    os.chdir(PYROOTDIR)
    dtg = time.strftime("%y%m%d") + "_" + time.strftime("%H.%M.%S")
    savedir = os.path.abspath(ROOTDIR + '/_dataout' + '/' + dtg)
    if (not(os.path.isdir(savedir))):
        os.makedirs(savedir)
        os.makedirs(savedir + '/bins/particles_init')
        os.makedirs(savedir + '/bins/particles_final')
        os.makedirs(savedir + '/bins/satellites')
        os.makedirs(savedir + '/graphs/EBModels')
        os.makedirs(savedir + '/ADPIC')
    os.chdir(savedir)

    srcfile = PYROOTDIR + '/__simulationvariables.py' #change this
    shutil.copy(srcfile, './')

    return savedir, dtg