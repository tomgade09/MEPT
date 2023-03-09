import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os, sys
import csv
import struct

def plotXY(xdata, ydata, title, xlabel, ylabel, filename, showplot=False):
    plt.plot(xdata, ydata, '.')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

    if showplot:
        plt.show()

def plotAllParticles(v_e_para, v_e_perp, z_e, v_i_para, v_i_perp, z_i, B_z, E_z, B_E_z_dim, showplot=False):
    if (not(os.path.isdir('./graphs/allparticles'))):
                os.makedirs('./graphs/allparticles')
    os.chdir('./graphs/allparticles')
    
    plt.figure(1)
    plotXY(v_e_para, v_e_perp, 'Electrons', 'Vpara (Re / s)', 'Vperp (Re / s)', 'electrons.png')

    plt.figure(2)
    plotXY(v_i_para, v_i_perp, 'Ions', 'Vpara', 'Vperp', 'ions.png')

    plt.figure(3)
    plotXY(v_e_perp, z_e, 'Electrons', 'Vperp (Re / s)', 'Z (Re)', 'z_vprp_electrons.png')
    
    plt.figure(4)
    plotXY(v_i_perp, z_i, 'Ions', 'Vperp (Re / s)', 'Z (Re)', 'z_vprp_ions.png')

    plt.figure(5)
    plotXY(v_e_para, z_e, 'Electrons', 'Vpara (Re / s)', 'Z (Re)', 'z_vpra_electrons.png')
    
    plt.figure(6)
    plotXY(v_i_para, z_i, 'Ions', 'Vpara (Re / s)', 'Z (Re)', 'z_vpra_ions.png')

    os.chdir('./../EBModels')

    plt.figure(7)
    plotXY(B_E_z_dim, B_z, 'B Field', 'Z (Re)', 'B (T)', 'B(z).png')

    plt.figure(8)
    plotXY(B_E_z_dim, E_z, 'E Field', 'Z (Re)', 'E (V/m)', 'E(z).png')
    
    os.chdir('./../..')

    if showplot:
        plt.show()

def plotFields(BModel, EField, sGrid, norm=False):
    os.chdir('./graphs/EBModels')

    if norm:
        units = ' (Re)'
    else:
        units = ' (m)'

    plt.figure(1)
    plotXY(sGrid, BModel, 'B Field', 's' + units, 'B (T)', 'B(z).png')

    plt.figure(2)
    plotXY(sGrid, EField, 'E Field', 's' + units, 'E (V/m)', 'E(z).png')
    
    os.chdir('./../..')

def plotSatelliteData(dataArray4D, numberMsmts, numberSats, dt, satNamesTuple, showplot=False):
    #need to add satellite plotting here
    for iii in range(numberMsmts):
        for jjj in range(numberSats):
            if (not(os.path.isdir('./graphs/satellites/' + satNamesTuple[jjj] + '/' + str(iii * 1000 * dt) + 's'))):
                os.makedirs('./graphs/satellites/' + satNamesTuple[jjj] + '/' + str(iii * 1000 * dt) + 's')
            os.chdir('./graphs/satellites/' + satNamesTuple[jjj] + '/' + str(iii * 1000 * dt) + 's')
            
            #print(os.getcwd())
            
            fignum = 9 + iii * numberSats * 3 + jjj * 3
            plt.figure(fignum)
            plotXY(dataArray4D[iii][jjj][0], dataArray4D[iii][jjj][1], 'Particles', 'Vpara (Re / s)', 'Vperp (Re / s)', 'electrons.png')
            plt.figure(fignum + 1)
            plotXY(dataArray4D[iii][jjj][1], dataArray4D[iii][jjj][2], 'Particles', 'Vperp (Re / s)', 'Z (Re)', 'z_vprp_electrons.png')
            plt.figure(fignum + 2)
            plotXY(dataArray4D[iii][jjj][0], dataArray4D[iii][jjj][2], 'Particles', 'Vpara (Re / s)', 'Z (Re)', 'z_vpra_electrons.png')
            
            plt.close(fignum)
            plt.close(fignum + 1)
            plt.close(fignum + 2)

            os.chdir('./../../../../')
            #print(os.getcwd())

def save4DDataToCSV(dataArray, folder):
    if (not(os.path.isdir(folder))):
        os.makedirs(folder)

    #print(dataArray[0][0][0][0], dataArray[0][0][1][0], dataArray[0][0][2][0]) #sat 1: para, perp, z
    #print(dataArray[0][1][0][0], dataArray[0][1][1][0], dataArray[0][1][2][0]) #sat 2
    #print(dataArray[0][2][0][0], dataArray[0][2][1][0], dataArray[0][2][2][0]) #sat 3
    #print(dataArray[0][3][0][0], dataArray[0][3][1][0], dataArray[0][3][2][0]) #sat 4

    for iii in range(len(dataArray)): #measurements
        if (not(os.path.isdir(folder + "/" + "msmt" + str(iii)))):
                os.makedirs(folder + "/" + "msmt" + str(iii))
        for jjj in range(len(dataArray[0])): #satellites
                with open(folder + "/" + "msmt" + str(iii) + "/sat" + str(jjj) + ".csv", "w", newline='\n') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerows(dataArray[iii][jjj])


def saveEscapedParticlesAndTimeToCSV(origParticles, escapedData):
    with open("./elecoutput.csv", "w", newline='\n') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["v_para orig", "v_perp orig", "z orig", "", "time escaped top", "para top", "perp top", "z top", "", "time escaped bottom", "para bottom", "perp bottom", "z bottom"])
        for iii in range(100352):
            data = []
            data.append(origParticles[0][0][iii]) #para
            data.append(origParticles[0][1][iii]) #perp
            data.append(origParticles[0][2][iii]) #z
            data.append("")
            data.append(escapedData[0][2][3][iii]) #time escaped top
            data.append(escapedData[0][2][0][iii]) #para top
            data.append(escapedData[0][2][1][iii]) #perp top
            data.append(escapedData[0][2][2][iii]) #z top
            data.append("")
            data.append(escapedData[0][0][3][iii]) #time escaped bottom
            data.append(escapedData[0][0][0][iii]) #para bottom
            data.append(escapedData[0][0][1][iii]) #perp bottom
            data.append(escapedData[0][0][2][iii]) #z bottom
            csvwriter.writerow(data)

    with open("./ionsoutput.csv", "w", newline='\n') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["v_para orig", "v_perp orig", "z orig", "", "time escaped top", "para top", "perp top", "z top", "", "time escaped bottom", "para bottom", "perp bottom", "z bottom"])
        for iii in range(100352):
            data = []
            data.append(origParticles[1][0][iii]) #para
            data.append(origParticles[1][1][iii]) #perp
            data.append(origParticles[1][2][iii]) #z
            data.append("")
            data.append(escapedData[0][3][3][iii]) #time escaped top
            data.append(escapedData[0][3][0][iii]) #para top
            data.append(escapedData[0][3][1][iii]) #perp top
            data.append(escapedData[0][3][2][iii]) #z top
            data.append("")
            data.append(escapedData[0][1][3][iii]) #time escaped bottom
            data.append(escapedData[0][1][0][iii]) #para bottom
            data.append(escapedData[0][1][1][iii]) #perp bottom
            data.append(escapedData[0][1][2][iii]) #z bottom
            csvwriter.writerow(data)


#Tools
def readDoubleBinary(filename):
    ret = []
    f = open(filename, "rb")
    
    bytes = f.read(8)
    while bytes:
        tmp = struct.unpack('d', bytes)
        ret.append(tmp)
        bytes = f.read(8)

    f.close()

    return ret

def readBinsAndOutputGraphs(saveFolder, binFolder): #great to call from the command line - from __plotParticles import *; readBinsAndOutputGraphs(args)
    e_vpara = readDoubleBinary(binFolder + 'e_vpara.bin')
    e_vperp = readDoubleBinary(binFolder + 'e_vperp.bin')
    e_z = readDoubleBinary(binFolder + 'e_z.bin')
    i_vpara = readDoubleBinary(binFolder + 'i_vpara.bin')
    i_vperp = readDoubleBinary(binFolder + 'i_vperp.bin')
    i_z = readDoubleBinary(binFolder + 'i_z.bin')
    
    B_z = []
    E_z = []
    B_E_z_dim = []

    if (not(os.path.isdir(saveFolder))):
        os.makedirs(saveFolder)
    os.chdir(saveFolder)

    plotAllParticles(e_vpara, e_vperp, e_z, i_vpara, i_vperp, i_z, B_z, E_z, B_E_z_dim)
