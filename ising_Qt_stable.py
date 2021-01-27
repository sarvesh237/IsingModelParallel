__title__ = "ising model simulation"
__author__ = "Sarvesh Thakur, MS16146"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import *
import random
from datetime import datetime
import time
import multiprocessing as mp
from scipy.optimize import curve_fit
import pandas as pd 
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, QtCore, QtGui
from numba import jit
import gc
from numpy.random import rand

#creates a 2d nxn square lattice 
def createLattice(X,temperature):
    lattice = 1*np.ones((X,X))
    #lattice = np.random.choice([-1,1],size=(X,X))
    
    return lattice
#calculates the initialEnergy of the lattice
@jit(nopython=True)
def initialEnergy(lattice,n):
    energy = 0
    for i in range(0,n):
        for j in range(0,n):
            for l in [(i-1)%n,(i+1)%n]:
                energy = energy + (lattice[i][j]*lattice[l][j])
            for m in [(j-1)%n,(j+1)%n]:
                energy = energy + (lattice[i][j]*lattice[i][m])
    return -J*energy-B*np.sum(lattice)

#change in energy after a flip in spin, 
@jit(nopython=True)
def deltaE(i,j,lattice,n):
    return 2*J*lattice[i,j]*(lattice[i,(j+1)%n]+lattice[i,(j-1)%n]+lattice[(i+1)%n,j]+lattice[(i-1)%n,j]) + 2*B*lattice[i,j]

#decides whether the flip is accepted, X is the lattice
@jit(nopython=True)
def flipspinDecide(X,temperature):
    if (X <= 0) or (random.random() < np.exp(-X/(Kb*temperature))):
        return 1
    else: return 0

#takes a list of energy values as an input and outputs the specific heat
@jit(nopython=True)
def specificHeat(X,temperature,n):
    return np.var(X)/((n**2)*temperature**2)

#calculates the autocorrelation for a given temperature
@jit(nopython=True,parallel=True)
def autoCorrelation(t_max,t_now,m,temperature):
    s = slice(0,t_max-t_now)
    p = slice(t_now,t_max)
    var1 = np.sum(m[s]*m[p])    
    var3 = np.sum(m[s])*np.sum(m[p])/(t_max-t_now)    
    chi = (var1 - var3)/(t_max - t_now)
    return chi

#calculates the error using jacKnife method
@jit(nopython=True)
def jacKnife(store_energy,temperature,n,sp_heat):
    c_i = [] 
    length_of_energy = len(store_energy)
    for i in range(0,length_of_energy):
        if i % 10000 == 0:
            print(i,'/',length_of_energy,"jacknife status for t = ",temperature)
        temp_list = store_energy
        temp_list = np.delete(temp_list,i)
        c_i.append(specificHeat(temp_list,temperature,n))
    temp_c = 0
    for i in c_i:
        temp_c = temp_c + (i-sp_heat)*(i-sp_heat)
    jacknife = np.sqrt(temp_c)
    print(length_of_energy,'/',length_of_energy,"jacknife status for t = ",temperature)
    return jacknife

#func to fit autocorrelation data
def func(t,tau):
    return np.exp(-t/tau)

#metropolis loop that returns m and E, this function is called inside of mainISING()
@jit(nopython=True,parallel=True)
def fastMetropolis(iterations,lattice,n,temperature,saveLattice,showLattice,parallel,noofPoints):
    
    current_energy = initialEnergy(lattice,n) #variable that stores the current energy values
    current_mag = np.sum(lattice) #variable that stores the current magnetization value (note that this is not magnetization/spin)

    m = [] #list to store magnetization
    store_energy = [] #list to store energy values
    _time_list = []
    
    keep_count = 0 #just a counter that resets after every 'noofPoints' iterations
    for x_iter in range(0,iterations):
        
        #update sites randomly
        
        i = int(random.random()*(n))
        j = int(random.random()*(n))
        
        energyChange = deltaE(i,j,lattice,n)
        if  flipspinDecide(energyChange,temperature) == 1:
            lattice[i,j] = -1*lattice[i,j]
            current_energy = current_energy + energyChange
            current_mag = current_mag + 2*lattice[i,j]
        
        #stores the magnetization,energy,iterations for 'noofPoints' i.e if the value of noofPoints is 10k, we will get a list of 10k points
        
        if keep_count % (iterations//noofPoints) == 0:
            store_energy.append(current_energy)
            m.append(current_mag)
            _time_list.append(x_iter)
        keep_count += 1

        '''uncomment the following two lines to show the live preview'''
        #if (showLattice == True) and (parallel == False) and (x_iter % (iterations/10000) == 0):
            #main.oldPlot(lattice)
    return store_energy,m,_time_list

#here we calculated all quantities and errors
def mainISING(iterations,temperature,n,saveLattice,usePrevLattice,parallel,showLattice,initialGuess,noofPoints):
    if usePrevLattice == False:
        lattice = createLattice(n,temperature)
    print("MCMC loop for ","\t",temperature,"\t", " : STARTED")
    #pbar = tqdm(total=iterations)
    
    #calling the metropolis loop
    store_energy,m,_time_list = fastMetropolis(iterations,lattice,n,temperature,saveLattice,showLattice,parallel,noofPoints)
    
    print("MCMC loop for ","\t",temperature,"\t", " : FINISHED")
    
    #converting to numpy arrays
    store_energy = np.array(store_energy)
    m = np.array(m)
    m = m/(n*n) #we want magnetization per spin
    
    #saves the picture of the evolved lattice if the saveLattice variable is set to True
    if saveLattice == True:
        now = datetime.now()
        todaysDate = now.strftime("%d-%m-%Y_%H %M %S")
        fileName = f"GRAPH_{todaysDate}_{temperature}.png"
        plt.imshow(lattice,cmap='gray')
        plt.title(f"B = {B} J = {J} T = {temperature} {n}x{n} lattice")
        
        plt.savefig(fileName,dpi = 300)
        plt.clf()
        
        fileName = f"MAGvsTime_{todaysDate}_{temperature}.png"
        plt.plot(_time_list,m,color='royalblue', label='Magnetization',linewidth = 0.5)
        plt.title(f"B = {B} T = {temperature} lattice size = {n}x{n} iterations = {iterations}")
        plt.ylim(-0.1,1)
        plt.xlabel("Time")
        plt.ylabel("Magnetization")
        plt.legend()
        plt.savefig(fileName,dpi = 300)
        plt.clf()
        
        fileName = f"EnergyvsTime_{todaysDate}_{temperature}.png"
        plt.plot(_time_list,store_energy,color='royalblue', label='Energy',linewidth = 0.5)
        plt.title(f"B = {B} T = {temperature} lattice size = {n}x{n} iterations = {iterations}")
        plt.ylim(-40000,-24000)
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.savefig(fileName,dpi = 300)
        plt.clf()
        
    #AUTO-CORR
    
    print("AUTOCORR for ","\t",temperature,"\t", " : STARTED")

    #autocorrelation is only calculated for ~ 10,000 points to speed up the process, so we reduce the m2 and _time_list if the size is greater than 10,000
    m2 = m #a temporary variable to store m, incase we reduce the full m list
    if noofPoints > 10000:
        order_of_iterations = len(str(iterations)) - 1
        jump_by = int(iterations/(10**(order_of_iterations-4)))
        m2 = m2[0::jump_by]
        _time_list = _time_list[0::jump_by]
    
    autocorr = []
    time1 = []     #x-axis of auto correlation
    '''list of time where autocorrelation will be calculated, 
    taking 20% of the values for temperatures away from Tc 
    and 50 % of the values for temperatures near Tc '''  
    
    for t_now_pos in range(0,len(_time_list)):
        time1.append(_time_list[t_now_pos])
        autocorr.append(autoCorrelation(len(_time_list),t_now_pos,m2,temperature))
    autocorr = np.array(autocorr)
    
    if (temperature < 2) or (temperature > 3) :
        time1 = time1[slice(0,int(0.2*len(time1)))]
        autocorr = autocorr[slice(0,int(0.2*len(autocorr)))]
    else:
        time1 = time1[slice(0,int(0.5*len(time1)))]
        autocorr = autocorr[slice(0,int(0.5*len(autocorr)))]
    
    if autocorr[0] != 0:
        params = curve_fit(func, time1, autocorr/autocorr[0],p0=[initialGuess],bounds=(0,iterations),maxfev = 1000000)
        tau = int(params[0][0])
        if tau == 0:
            tau = 1
        print(tau,"\t", "this is tau for t = ","\t",temperature)
    else:
        '''this is to prevent ZeroDivisionError incase all autocorrelations are zero which happens 
        at low temperatures which have zero magnetization for the entire list'''
        tau = 2
        return 1,0,temperature,autocorr,time1,0,tau,0,0
    
    print("AUTOCORR for ","\t",temperature,"\t", " : FINISHED")
    print("Tau for ","\t",temperature,"\t", " = ",tau)
    print("M_AVG,M_STD for ","\t",temperature, " : STARTED")
    
    '''this function locates the variable which is after 1 tau iterations'''
    def slicer(p,tau):
        for i in range(0,len(p)):
            if i*iterations/noofPoints >= tau:
                return i
    one_tau = slicer(m,tau)
    tau_reduced = one_tau//4
    '''fallback to tau = 1 to avoid slice size cannot be zero error '''
    if tau_reduced == 0:
        tau_reduced = 1


    #calculation of magnetization per spin and its standard deviation
    m = m[slice(one_tau,len(m))]
    m = m[0::tau_reduced]
    m_avg = np.average(m)
    m_stdev = np.std(m)
    print("M_AVG,M_STD for ","\t",temperature,"\t", " : FINISHED")
    print("CHI,JACKNIFE for ","\t",temperature,"\t", " : STARTED")

    
    #calculation of susceptibility
    chi = n*n*m_stdev*m_stdev/(Kb*temperature)

    #calculation of specific heat and its error (jacknife)
    sp_heat = 0
    #one_tau = slicer(store_energy,tau)
    store_energy = store_energy[slice(one_tau,len(store_energy))]
    store_energy = store_energy[0::tau_reduced]
    sp_heat = specificHeat(store_energy,temperature,n)
    jacknife = jacKnife(store_energy,temperature,n,sp_heat)
    
    print("CHI,JACKNIFE for ","\t",temperature,"\t", " : FINISHED")
    print("MAIN LOOP finished for t = ",temperature)
    
    return m_avg,sp_heat,temperature,autocorr,time1,m_stdev,tau,jacknife,chi



J = 1
B = 0
Kb = 1

'''If set to True, the previous equilibrium lattice will be used as the starting lattice for the next temperature.'''
usePrevLattice = False

#single thread function
def singleThread():
    _mag = []
    _specific_heat = []
    _store_energy = []
    _auto = []
    _time = []
    _mstd = []
    _tau = []
    _jacknife = []
    _chi =[]
    if usePrevLattice == True:
        lattice = createLattice(n,temperature)
    for temperature in t:
        results = mainISING(iterations[0],temperature,n[0],saveLattice[0],usePrevLattice[0],parallel[0],showLattice[0],initialGuess[0],noofPoints[0])
        _mag.append(results[0])
        _specific_heat.append(results[1])
        _auto.append(results[3])
        _time.append(results[4])
        _mstd.append(results[5])
        _tau.append(results[6])
        _jacknife.append(results[7])
        _chi.append(results[8])
    return _mag,_specific_heat,_auto,_time,_mstd,_tau,_jacknife,_chi

#this function is called when using parallel threads
def parallelThreads():
    p = get_context("spawn").Pool() 
    results = p.starmap(mainISING, zip(iterations,t,n,saveLattice,usePrevLattice,parallel,showLattice,initialGuess,noofPoints))
    p.close()
    p.join()
    print("EXITING PARALLEL POOL")
    _mag = []
    _specific_heat = []
    _auto = []
    _t = []
    _mstd = []
    _tau = []
    _jacknife = []
    _chi=[]
    for r in results:
        _mag.append(r[0])
        _specific_heat.append(r[1])
        _auto.append(r[3])
        _t.append(r[4])
        _mstd.append(r[5])
        _tau.append(r[6])
        _jacknife.append(r[7])
        _chi.append(r[8])
    return _mag,_specific_heat,_auto,_t,_mstd,_tau,_jacknife,_chi


#main window written in pyqt5
class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        #white canvas for the graph
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        #starts the simulation
        self.button = QtWidgets.QPushButton('Start',self)
        self.button.clicked.connect(self.begin_simulation)
        
        #quits the window
        self.button2 = QtWidgets.QPushButton('Quit')
        self.button2.clicked.connect(self.quit)
        
        self.helloMsg = QLabel('<h2>Ising Model Simulation</h2>')
        self.msg = QLabel('Click below to start the simulation : ')
        self.setWindowTitle("Ising Model - Sarvesh MS16146")
        
        #lattice size
        self.n_input = QLineEdit("100")
        self.n_input.setValidator(QIntValidator())
        self.n_input.setAlignment(Qt.AlignCenter)
        
        #iterations
        self.iter_input = QLineEdit("1000000")
        self.iter_input.setAlignment(Qt.AlignCenter)
        
        #initial guess for tau
        self.initialguess1 = QLineEdit("100000")
        self.initialguess1.setValidator(QIntValidator())
        self.initialguess1.setAlignment(Qt.AlignCenter)
        
        #this is the maxfev setting which determines the number of iterations the curve fitting is allowed
        self.maxfev = QLineEdit("1000000")
        self.maxfev.setAlignment(Qt.AlignCenter)
        self.maxfev.setValidator(QIntValidator())
        
        #number of points to sample m,E at
        self.noofPoints = QLineEdit("10000")
        self.noofPoints.setAlignment(Qt.AlignCenter)
        self.noofPoints.setValidator(QIntValidator())
        
        #range of temp to be simulated
        self.tempRange1 = QLineEdit("2.069")
        self.tempRange1.setAlignment(Qt.AlignCenter)
        self.tempRange1.setValidator(QDoubleValidator())
        self.tempRange2 = QLineEdit("2.469")
        self.tempRange2.setAlignment(Qt.AlignCenter)
        self.tempRange2.setValidator(QDoubleValidator())
        
        #save data with this particular filename
        self.dataName = QLineEdit("data.csv")
        
        #no of points
        self.points = QSlider(Qt.Horizontal)
        self.points.setMaximum(50)
        self.points.setMinimum(5)
        self.points.setValue(15)
        self.points.setTickPosition(QSlider.TicksBothSides)
        self.points.setTickInterval(1)
        self.pointsDisplay = QLabel("15")
        self.pointsDisplay.setAlignment(Qt.AlignCenter)
        self.points.valueChanged.connect(self.valuechange)
        
        #parallel or single thread
        self.msg2 = QLabel("Use multiple threads ?")
        self.parallel = QRadioButton('Yes')
        self.single = QRadioButton('No')
        self.parallel.setChecked(True)
        self.single.setChecked(False)
        
        self.yes1 = QCheckBox('Save lattice images and raw data ?')
        self.yes1.setChecked(False)
        self.timetaken = QLabel("")
        
        self.yes2 = QCheckBox('Show the evolution of the lattice (only works with single thread)?')
        self.yes2.setChecked(False)
        
        self.button3 = QtWidgets.QPushButton('Finite Size Scaling')
        self.button3.clicked.connect(self.launchFinite)
        
        #adding all the above objects into a form
        formLayout = QtWidgets.QFormLayout()        
        formLayout.addRow('Lattice size (n x n) : ',self.n_input)
        formLayout.addRow('Iterations (can also input like 1e9)', self.iter_input)
        formLayout.addRow('Initial guess for Tau', self.initialguess1)
        formLayout.addRow('maxfev', self.maxfev)
        formLayout.addRow('No. of points to sample m and E at \n(lesser is faster)', self.noofPoints)
        hbox = QtWidgets.QHBoxLayout()
        formLayout.addRow(QLabel("Temperature range"),hbox)
        hbox.addWidget(self.tempRange1)
        hbox.addWidget(self.tempRange2)
        formLayout.addRow('No. of temperatures',self.points)
        formLayout.addRow(self.pointsDisplay)
        formLayout.addRow(self.msg2)
        formLayout.addRow(self.parallel)
        formLayout.addRow(self.single)
        formLayout.addRow(self.yes1)        
        formLayout.addRow(self.yes2)        
        formLayout.addRow('Save data as',self.dataName)
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.toolbar,0,1)
        layout.addWidget(self.button2,4,1)
        layout.addWidget(self.canvas,1,1)
        layout.addWidget(self.button,4,0)
        layout.addWidget(self.button3,6,0)
        layout.addWidget(self.timetaken,3,1)
        layout.addWidget(self.helloMsg,0,0)
        layout.addWidget(self.msg,3,0)
        layout.addLayout(formLayout,1,0)
        layout.setContentsMargins(30,20,30,20)
        self.setLayout(layout)
    #value change for no. of points slider
    def valuechange(self):
        pts = self.points.value()
        print(pts)
        self.pointsDisplay.setText(str(pts))
    
    #plots the evolution of the lattice, no that all numba decorators need to be commented out for this to work
    def oldPlot(self,lattice):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.imshow(lattice, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        self.canvas.draw()
        self.canvas.flush_events()
    
    #launches finite size scaling GUI
    def launchFinite(self):
        self.pop = FiniteSizeScaling()
        self.pop.show()
        
    #takes all the inputs and calls parallel() or singleThread() to start the simulation
    def begin_simulation(self):
        self.button.setEnabled(False)
        self.button2.setEnabled(False)
        main.repaint() #updates the gui
        self.timetaken.setStyleSheet('color: red')
        self.timetaken.setText("Please wait for the simulation to finish...")
        main.repaint()
        global parallel,n,pts,t,iterations,saveLattice,usePrevLattice,showLattice,initialGuess,noofPoints
        
        if self.yes1.isChecked() == True:
            saveLattice = True
        else:
            saveLattice = False

        if self.yes2.isChecked() == True:
            showLattice = True
        else:
            showLattice = False
        
        if self.parallel.isChecked() == True:
            print("using parallel threads")
            parallel = True
        else:
            parallel = False
            print("using single core")
        pts = int(self.points.value())
        n = int(self.n_input.text())
        initialguess1 = int(self.initialguess1.text())
        maxfev = int(self.maxfev.text())
        noofPoints = int(self.noofPoints.text())
        
        #creates a list of temperatures
        ti = float(self.tempRange1.text())
        if ti == 0:
            print("initial temperature should be greater than 0")
            print("the simulation cannot continue")
            self.timetaken.setText("<h2>error !</h2> initial temperature should be greater than 0")
            main.repaint()
            return 0
        tf = float(self.tempRange2.text())
        if ti>tf:
            print("initial temperature cannot be greater than final")
            self.timetaken.setText("<h2>error !</h2> initial temperature cannot be greater than final")
            main.repaint()
            return 0
        t = np.round(np.linspace(ti,tf,pts),3)
        
        pts = len(t)
        mag = []
        specific_heat = []
        store_energy = []
        auto = []
        chi =[]
        n = list([n]*pts)
        saveLattice = list([saveLattice]*pts)
        usePrevLattice = list([False]*pts)
        parallel = list([parallel]*pts)
        showLattice = list([showLattice]*pts)
        iterations = list([int(   float(self.iter_input.text())   )]*pts)
        initialGuess = list([initialguess1]*pts)
        noofPoints = list([noofPoints]*pts)
        #Calling the simulation
        t0 = time.time()
        if parallel[0] == True :
            mag,specific_heat,auto,_time,_mstd,tau,jacknife,chi = parallelThreads()
        else:
            mag,specific_heat,auto,_time,_mstd,tau,jacknife,chi = singleThread()
        
        #Saving plots
        t3 = time.time()
        print("Please wait, saving plots ,main program finished in ",t3-t0)
        self.timetaken.setText("Saving plots.., main program finished in "+ str(int(t3-t0)))
        main.repaint()
        #for saving the plots
        now = datetime.now()
        todaysDate = now.strftime("%d-%m-%Y_%H %M %S")
        
        #magnetization
        fileName = "MAG_" +  todaysDate + ".png"
        plt.errorbar(t,mag,yerr = _mstd, uplims=True, lolims=True,capsize=0.9,fmt='.',ls='-',elinewidth=0.8,linewidth=1.5,ecolor='royalblue', label='Magnetization',color = "crimson")
        #onsager's exact solution
        t_critical = 2/(np.log(1+np.sqrt(2)))
        if ti < t_critical:
            t_dense = np.linspace(ti,t_critical,100)
            exact_m = lambda t: np.piecewise(t, [t < t_critical], [lambda t: (1-(1/np.sinh(2*J/(Kb*t))**4))**(1/8)])
            plt.plot(t_dense,exact_m(t_dense),color='blue',linewidth = 0.8,label = "Onsager's exact soln",ls='--')
        plt.title(f"B = {B} J = {J} lattice size = {n[0]}x{n[0]} iterations = {iterations[0]}")
        plt.xlabel("Temperature")
        plt.ylabel("Magnetization")
        plt.legend()
        plt.savefig(fileName,dpi = 300)
        plt.clf()
        
        #susceptibility
        fileName = "CHI_" +  todaysDate + ".png"
        plt.plot(t,chi,label='susceptibility',color='royalblue',linewidth=1.5)
        plt.scatter(t,chi,color='crimson',s = 8)
        plt.title(f"B = {B} J = {J} lattice size = {n[0]}x{n[0]} iterations = {iterations[0]}")
        plt.xlabel("Temperature")
        plt.ylabel("Susceptibility")
        plt.legend()
        plt.savefig(fileName,dpi = 300)
        plt.clf()
        
        #specific heat
        fileName = "SPH_" +  todaysDate + ".png"
        #plt.scatter(t,specific_heat,label='specific heat',color = "royalblue")
        plt.errorbar(t,specific_heat,yerr = jacknife, uplims=True, lolims=True,capsize=0.9,fmt='.',ls='-',elinewidth=0.8,linewidth=1.5,ecolor='royalblue', label='specific heat with error (jacknife)',color = "crimson")
        plt.title(f"B = {B} J = {J} lattice size = {n[0]}x{n[0]} iterations = {iterations[0]}")
        plt.xlabel("Temperature")
        plt.ylabel("Specific Heat")
        plt.legend()
        plt.savefig(fileName,dpi = 300)
        plt.clf()

        #AUTOCORRELATION
        fig, ax = plt.subplots(13, 4)
        fig.set_size_inches(20, 50)
        for i in range(0,pts):
            exponential = []
            for k in _time[i]:
                exponential.append(func(k,tau[i]))
            
            if auto[i][0] != 0:
                plt.subplot(13, 4, i+1)
                plt.scatter(_time[i],auto[i]/auto[i][0],label='autoCorrelation')
                plt.plot(_time[i],exponential,label='autoCorrelation curvefit',color='orange')
                plt.title("t="+str(t[i])+'|tau='+str(int(tau[i])))
                plt.legend()
        fileName = "AUTO_" + todaysDate + ".png"
        plt.savefig(fileName)
        print("DONE !")
        plt.close(fig)
        t1 = time.time()
        
        #Printing the time taken
        minutes_taken = (t1-t0)
        self.timetaken.setStyleSheet('color: green')
        if minutes_taken//60 < 1:
            time_taken = "<h2>Finished in "+str(int(t1-t0))+ " secs !</h2>"
            print(time_taken)
            self.timetaken.setText(time_taken)
            
        else:
            time_taken = "<h2>Finished in " + str((minutes_taken)//60) + " min " + str(int((minutes_taken)- ((minutes_taken)//60)*60))+ " secs !</h2>"
            self.timetaken.setText(time_taken)
            print(time_taken)
        
        np.savetxt(self.dataName.text(), np.c_[t,mag, specific_heat,chi,n],delimiter = ',',header="t,mag,specific_heat,chi,n",comments='')
        
        del fig,mag,specific_heat,auto,_time,_mstd,tau,jacknife,chi
        gc.collect()
        self.button.setEnabled(True)
        self.button2.setEnabled(True)
        main.repaint()
        return None
    
    #the quit function
    def quit(self):
        app.quit()
        sys.exit()

#finite size scaling window
class FiniteSizeScaling(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(FiniteSizeScaling, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        #data file list
        self.file1 = QLineEdit("data1.csv")
        self.file1.setAlignment(Qt.AlignCenter)
        self.file2 = QLineEdit("data2.csv")
        self.file2.setAlignment(Qt.AlignCenter)
        self.file3 = QLineEdit("data3.csv")
        self.file3.setAlignment(Qt.AlignCenter)
        self.file4 = QLineEdit("data4.csv")
        self.file4.setAlignment(Qt.AlignCenter)
        self.file5 = QLineEdit("data5.csv")
        self.file5.setAlignment(Qt.AlignCenter)
        self.file6 = QLineEdit("data6.csv")
        self.file6.setAlignment(Qt.AlignCenter)
        self.btn1 = QPushButton("Select File 1")
        self.btn1.clicked.connect(lambda : self.selectFile(self.file1))
        self.btn2 = QPushButton("Select File 2")
        self.btn2.clicked.connect(lambda : self.selectFile(self.file2))
        self.btn3 = QPushButton("Select File 3")
        self.btn3.clicked.connect(lambda : self.selectFile(self.file3))
        self.btn4 = QPushButton("Select File 4")
        self.btn4.clicked.connect(lambda : self.selectFile(self.file4))
        self.btn5 = QPushButton("Select File 5")
        self.btn5.clicked.connect(lambda : self.selectFile(self.file5))        
        self.btn6 = QPushButton("Select File 6")
        self.btn6.clicked.connect(lambda : self.selectFile(self.file6))
        
        self.label1 = QLabel("<h4>xc</h4>")
        self.label2 = QLabel("<h4>gamma</h4>")
        self.label3 = QLabel("<h4>nu</h4>")

        #sliders for different parameters
        self.points = QSlider(Qt.Horizontal)
        self.points.setMaximum(2300)
        self.points.setMinimum(2238)
        self.points.setValue(2269)
        self.points.setTickPosition(QSlider.TicksBothSides)
        self.points.setTickInterval(1)
        self.pointsDisplay = QLabel("2.269")
        self.pointsDisplay.setAlignment(Qt.AlignCenter)
        self.points.valueChanged.connect(self.valuechange)

        self.points1 = QSlider(Qt.Horizontal)
        self.points1.setMaximum(5280)
        self.points1.setMinimum(-3520)
        self.points1.setValue(1760)
        self.points1.setTickPosition(QSlider.TicksBothSides)
        self.points1.setTickInterval(1)
        self.points1Display = QLabel("1.760")
        self.points1Display.setAlignment(Qt.AlignCenter)
        self.points1.valueChanged.connect(self.valuechange)       
        
        self.points2 = QSlider(Qt.Horizontal)
        self.points2.setMaximum(1200)
        self.points2.setMinimum(800)
        self.points2.setValue(1000)
        self.points2.setTickPosition(QSlider.TicksBothSides)
        self.points2.setTickInterval(1)
        self.points2Display = QLabel("1.0")
        self.points2Display.setAlignment(Qt.AlignCenter)
        self.points2.valueChanged.connect(self.valuechange)       
        
        #select the function, by default susceptibility is selected
        self.magnetization = QRadioButton('Magnetization')
        self.susceptibility = QRadioButton('Susceptibility')
        self.susceptibility.setChecked(True)
        self.spec_heat = QRadioButton('Specific Heat')
        
        #adding everything into a form layout
        formLayout = QtWidgets.QGridLayout()        
        formLayout.setVerticalSpacing(5)
        formLayout.addWidget(self.file1,0,0)
        formLayout.addWidget(self.btn1,0,1)
        formLayout.addWidget(self.file2,1,0)
        formLayout.addWidget(self.btn2,1,1)
        formLayout.addWidget(self.file3,2,0)
        formLayout.addWidget(self.btn3,2,1)
        
        formLayout.addWidget(self.file4,3,0)
        formLayout.addWidget(self.btn4,3,1)
        formLayout.addWidget(self.file5,4,0)
        formLayout.addWidget(self.btn5,4,1)
        formLayout.addWidget(self.file6,5,0)
        formLayout.addWidget(self.btn6,5,1)
        
        formLayout.addWidget(self.label1,6,0)
        formLayout.addWidget(self.points,7,0)
        formLayout.addWidget(self.label2,8,0)
        formLayout.addWidget(self.points1,9,0)
        formLayout.addWidget(self.label3,10,0)
        formLayout.addWidget(self.points2,11,0)
        formLayout.addWidget(self.magnetization,12,0)
        formLayout.addWidget(self.susceptibility,13,0)
        formLayout.addWidget(self.spec_heat,14,0)
        
        formLayout.addWidget(self.pointsDisplay,7,1)
        formLayout.addWidget(self.points1Display,9,1)
        formLayout.addWidget(self.points2Display,11,1)
        formLayout.addWidget(self.toolbar,15,0)
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.canvas,1,70,40,100)
        layout.addLayout(formLayout,1,0,10,60)
        layout.setContentsMargins(30,30,30,30)
        self.setLayout(layout)
        
    #dQ is updated first time a quantity is selected, it is used to set initial parameters of alpha,beta,gamma
    dQ = ""
    #selects a funtion based on which radio button is selected
    def decideQuantity(self):
        if self.magnetization.isChecked() == True:
            if FiniteSizeScaling.dQ != "a":
                self.points1.setMaximum(250)
                self.points1.setMinimum(0)
                self.points1.setValue(125)
                self.points1Display.setText("0.125")
                FiniteSizeScaling.dQ = "a"
            self.label2.setText("<h4>beta</h4>")
            quantity = "mag"
            return quantity
        elif self.susceptibility.isChecked() == True:
            if FiniteSizeScaling.dQ != "b":
                self.points1.setMaximum(1850)
                self.points1.setMinimum(1250)
                self.points1.setValue(1750)
                self.points1Display.setText("1.760")
                FiniteSizeScaling.dQ = "b"
            quantity = "chi"
            self.label2.setText("<h4>gamma</h4>")
            return quantity
        elif self.spec_heat.isChecked() == True:
            if FiniteSizeScaling.dQ != "c":
                self.points1.setMaximum(500)
                self.points1.setMinimum(-500)
                self.points1.setValue(0)
                self.points1Display.setText("0.000")
                FiniteSizeScaling.dQ = "c"
            self.label2.setText("<h4>aplha</h4>")
            quantity = "specific_heat"
            return quantity
    #value change of parameters        
    def valuechange(self):
        #pts is xc, pts1 is p,pts2 is q
        quantity = self.decideQuantity()
        pts = self.points.value()
        pts1 = self.points1.value()
        pts2 = self.points2.value()
        self.pointsDisplay.setText(str(  round(0.001*pts,3) ))
        self.points1Display.setText(str( round(0.001*pts1,3) ))
        self.points2Display.setText(str( round(0.001*pts2,3) ))
        data1 = self.file1.text()
        data2 = self.file2.text()
        data3 = self.file3.text()
        data4 = self.file4.text()
        data5 = self.file5.text()
        data6 = self.file6.text()
        #plotting the function with new set of parameters
        self.oldPlot(pts,pts1,pts2,data1,data2,data3,data4,data5,data6,quantity)
    #opens a file selector dialogue and outputs the file location    
    def selectFile(self,a):    
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        filenames = list()
        if dlg.exec_():
            a.setText(str(dlg.selectedFiles()[0]))
            return dlg.selectedFiles()[0]
            
    #plots the functions
    def oldPlot(self,pts,pts1,pts2,data1,data2,data3,data4,data5,data6,quantity):
        xc = round(0.001*pts,3)
        p = round(0.001*pts1,3)
        q = round(0.001*pts2,3)
        print(quantity)
        try:
            df = pd.read_csv(str(data1))
            t1 = np.array(df.t)
            L1 = int(np.array(df.n)[0])
        except:pass

        try:
            df2 = pd.read_csv(str(data2))
            t2 = np.array(df2.t)
            L2 = int(np.array(df2.n)[0])
        except:pass

        try:    
            df3 = pd.read_csv(str(data3))
            t3 = np.array(df3.t)
            L3 = int(np.array(df3.n)[0])
        except:pass
    
        try:
            df4 = pd.read_csv(str(data4))
            t4 = np.array(df4.t)
            L4 = int(np.array(df4.n)[0])
        except:pass

        try:
            df5 = pd.read_csv(str(data5))
            t5 = np.array(df5.t)
            L5 = int(np.array(df5.n)[0])
        except:pass

        try:    
            df6 = pd.read_csv(str(data6))
            t6 = np.array(df6.t)
            L6 = int(np.array(df6.n)[0])
        except:pass
    
        try:    
            qty1 = np.array(df[quantity])
        except:pass

        try:
            qty2 = np.array(df2[quantity])
        except:pass    

        try:
            qty3 = np.array(df3[quantity])
        except:pass
        
        try:    
            qty4 = np.array(df4[quantity])
        except:pass
    
        try:    
            qty5 = np.array(df5[quantity])
        except:pass
    
        try:    
            qty6 = np.array(df6[quantity])
        except:pass
        
        
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.set_xlim([-20,20])
        ax.set_xlabel(r"$(x-x_c)/((x_c)(L^\nu))$",fontsize = 15)
        
        
        
        
        if quantity == "chi":
            ax.set_title(f"Finite Size Scaling for Susceptibility \n xc = {xc}, p = {p}, q = {q}", fontsize = 20)
            A = lambda chi,L : chi*L**(-p/q)
            ax.set_ylabel(r'$L^{\frac{-\gamma}{\nu}} \chi_L(t) $',fontsize = 15)
        elif quantity == "specific_heat":
            ax.set_title(f"Finite Size Scaling for Specific Heat \n xc = {xc}, p = {p}, q = {q}", fontsize = 20)
            A = lambda c,L : c*L**(-p/q)
            ax.set_ylabel(r'$L^{\frac{-\alpha}{\nu}} c_L(t) $',fontsize = 15)
        elif quantity == "mag":
            ax.set_title(f"Finite Size Scaling for Magnetization \n xc = {xc}, p = {p}, q = {q}", fontsize = 20)
            A = lambda m,L : m*L**(p/q)
            ax.set_ylabel(r'$L^{\frac{\beta}{\nu}} m_L(t) $',fontsize = 15)

        B = lambda x,L : L**(1/q)*(x-xc)/xc
        
        try:
            i = ax.scatter(B(t1,L1),A(qty1,L1),marker='^')
            #i1 = ax.plot(B(t1,L1),A(qty1,L1),linewidth = '0.3')
            i.set_label(L1)
        except:pass
        try:
            j = ax.scatter(B(t2,L2),A(qty2,L2),marker = 's')
            #j1 = ax.plot(B(t2,L2),A(qty2,L2),linewidth = '0.3')

            j.set_label(L2)
        except:pass    
        try:    
            k = ax.scatter(B(t3,L3),A(qty3,L3),marker = 'o')
            #k1 = ax.plot(B(t3,L3),A(qty3,L3),linewidth = '0.3')

            k.set_label(L3)
        except:pass
    
        try:
            l = ax.scatter(B(t4,L4),A(qty4,L4),marker='p')
            #l1 = ax.plot(B(t4,L4),A(qty4,L4),linewidth = '0.3')

            l.set_label(L4)
        except:pass
        try:
            m = ax.scatter(B(t5,L5),A(qty5,L5),marker = '*')
            #m1 = ax.plot(B(t5,L5),A(qty5,L5),linewidth = '0.3')

            m.set_label(L5)
        except:pass    
        try:    
            n = ax.scatter(B(t6,L6),A(qty6,L6),marker = 'x')
            #n1 = ax.plot(B(t6,L6),A(qty6,L6),linewidth = '0.3')

            n.set_label(L6)
        except:pass
        
        ax.legend()
        self.canvas.draw()
        self.canvas.flush_events()
        
if __name__ == '__main__':
    from multiprocessing import get_context
    app = QtWidgets.QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
