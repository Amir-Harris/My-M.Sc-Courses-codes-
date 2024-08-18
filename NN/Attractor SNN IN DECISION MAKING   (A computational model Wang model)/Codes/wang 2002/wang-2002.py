'''

Integration circuit from:

Wang, X.-J. Probabilistic decision making by slow reverberation in cortical circuits. Neuron, 2002, 36, 955-968.

@author: Klaus Wimmer and Albert Compte

wimmer.klaus@googlemail.com
acompte@clinic.ub.es

'''

import brian2 as brian
from brian2 import mV, nS, ms, nF, kHz, Hz, network_operation, second
import numpy as np
import matplotlib.pyplot as plt
import time
ts = time.time()


'''
Creates the spiking network described in Wang 2002.

returns:
    groups, connections, update_nmda, subgroups

groups, connections, and update_nmda have to be added to the "Network" in order to run the simulation.
subgroups is used for establishing connections between the sensory and integration circuit; do not add subgroups to the "Network"

'''

# -----------------------------------------------------------------------------------------------
# Model parameters for the integration circuit
# -----------------------------------------------------------------------------------------------

brian.start_scope()
# Populations
f_E = 0.15  # Fraction of stimulus-selective excitatory neurons
N = 2000  # Total number of neurons
f_inh = 0.2  # Fraction of inhibitory neurons
NE = int(N * (1.0 - f_inh))  # Number of excitatory neurons (1600)
NI = int(N * f_inh)  # Number of inhibitory neurons (400)
N_D1 = int(f_E * NE)  # Size of excitatory population D1
N_D2 = N_D1  # Size of excitatory population D2
N_DN = int((1.0 - 2.0 * f_E) * NE)  # Size of excitatory population DN

# Connectivity - local recurrent connections
w_p = 1.6  # Relative recurrent synaptic strength within populations D1 and D2
w_m = 1.0 - f_E * (w_p - 1.0) / (
            1.0 - f_E)  # Relative recurrent synaptic strength of connections across populations D1 and D2 and from DN to D1 and D2
gEE_AMPA = 0.05 * nS  # Weight of AMPA synapses between excitatory neurons
gEE_NMDA = 0.165 * nS  # Weight of NMDA synapses between excitatory neurons
gEI_AMPA = 0.04 * nS  # Weight of excitatory to inhibitory synapses (AMPA)
gEI_NMDA = 0.13 * nS  # Weight of excitatory to inhibitory synapses (NMDA)
gIE_GABA = 1.3 * nS  # Weight of inhibitory to excitatory synapses
gII_GABA = 1.0 * nS  # Weight of inhibitory to inhibitory synapses
d = 0.5 * ms  # Transmission delay of recurrent excitatory and inhibitory connections

# Connectivity - external connections
gextE = 2.1 * nS  # Weight of external input to excitatory neurons
gextI = 1.62 * nS  # Weight of external input to inhibitory neurons


# Neuron model
CmE = 0.5 * nF  # Membrane capacitance of excitatory neurons
CmI = 0.2 * nF  # Membrane capacitance of inhibitory neurons
gLeakE = 25.0 * nS  # Leak conductance of excitatory neurons
gLeakI = 20.0 * nS  # Leak conductance of inhibitory neurons
Vl = -70.0 * mV  # Resting potential
Vt = -50.0 * mV  # Spiking threshold
Vr = -55.0 * mV  # Reset potential
tau_refE = 2.0 * ms  # Absolute refractory period of excitatory neurons
tau_refI = 1.0 * ms  # Absolute refractory period of inhibitory neurons

# Synapse model
VrevE = 0 * mV  # Reversal potential of excitatory synapses
VrevI = -70 * mV  # Reversal potential of inhibitory synapses
tau_AMPA = 2.0 * ms  # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms  # Decay constant of GABA-type conductances
tau_NMDA_decay = 100.0 * ms  # Decay constant of NMDA-type conductances
tau_NMDA_rise = 2.0 * ms  # Rise constant of NMDA-type conductances
alpha_NMDA = 0.5 * kHz  # Saturation constant of NMDA-type conductances


# Inputs
nu_ext_1 = 2392 * Hz  # Firing rate of external Poisson input to neurons in population D1
nu_ext_2 = 2392 * Hz  # Firing rate of external Poisson input to neurons in population D2
nu_ext = 2400 * Hz  # Firing rate of external Poisson input to neurons in population Dn and I

# -----------------------------------------------------------------------------------------------
# Set up the model
# -----------------------------------------------------------------------------------------------

# Neuron equations
eqsE = '''
dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau): volt
dgea/dt = -gea/(tau_AMPA) : 1
dgi/dt = -gi/(tau_GABA) : 1
dspre/dt = -spre/(tau_NMDA_decay)+alpha_NMDA*xpre*(1-spre) : 1
dxpre/dt= -xpre/(tau_NMDA_rise) : 1
gen : 1
tau : second
'''
eqsI = '''
dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau): volt
dgea/dt = -gea/(tau_AMPA) : 1
dgi/dt = -gi/(tau_GABA) : 1
gen : 1
tau : second
'''

# Set up the integration circuit
decisionE = brian.NeuronGroup(NE, model=eqsE, threshold='V > Vt',
                        reset='V = Vr', refractory=tau_refE)

decisionI = brian.NeuronGroup(NI, model=eqsI, threshold='V > Vt',
                        reset='V = Vr', refractory=tau_refI)
decisionE.tau = CmE / gLeakE
decisionI.tau = CmI / gLeakI
decisionE1 = brian.Subgroup(decisionE, 0, N_D1)
decisionE2 = brian.Subgroup(decisionE, N_D1, N_D1 + N_D2)
decisionE3 = brian.Subgroup(decisionE, N_D1 + N_D2, NE)

# Connections involving AMPA synapses
C_DE1_DE1_AMPA = brian.Synapses(decisionE1, decisionE1, 'w: 1', on_pre='gea += w', delay=d)
C_DE1_DE2_AMPA = brian.Synapses(decisionE1, decisionE2, 'w: 1', on_pre='gea += w', delay=d)
C_DE1_DE3_AMPA = brian.Synapses(decisionE1, decisionE3, 'w: 1', on_pre='gea += w', delay=d)
C_DE2_DE1_AMPA = brian.Synapses(decisionE2, decisionE1, 'w: 1', on_pre='gea += w', delay=d)
C_DE2_DE2_AMPA = brian.Synapses(decisionE2, decisionE2, 'w: 1', on_pre='gea += w', delay=d)
C_DE2_DE3_AMPA = brian.Synapses(decisionE2, decisionE3, 'w: 1', on_pre='gea += w', delay=d)
C_DE3_DE1_AMPA = brian.Synapses(decisionE3, decisionE1, 'w: 1', on_pre='gea += w', delay=d)
C_DE3_DE2_AMPA = brian.Synapses(decisionE3, decisionE2, 'w: 1', on_pre='gea += w', delay=d)
C_DE3_DE3_AMPA = brian.Synapses(decisionE3, decisionE3, 'w: 1', on_pre='gea += w', delay=d)
C_DE1_DE1_AMPA.connect()
C_DE1_DE2_AMPA.connect()
C_DE1_DE3_AMPA.connect()
C_DE2_DE1_AMPA.connect()
C_DE2_DE2_AMPA.connect()
C_DE2_DE3_AMPA.connect()
C_DE3_DE1_AMPA.connect()
C_DE3_DE2_AMPA.connect()
C_DE3_DE3_AMPA.connect()
C_DE1_DE1_AMPA.w = C_DE2_DE2_AMPA.w = w_p * gEE_AMPA / gLeakE
C_DE1_DE2_AMPA.w = C_DE2_DE1_AMPA.w = C_DE3_DE1_AMPA.w = C_DE3_DE2_AMPA.w = w_m * gEE_AMPA / gLeakE
C_DE1_DE3_AMPA.w = C_DE2_DE3_AMPA.w = C_DE3_DE3_AMPA.w = gEE_AMPA / gLeakE

C_DE_DI_AMPA = brian.Synapses(decisionE, decisionI, 'w: 1', on_pre='gea += w', delay=d)
C_DE_DI_AMPA.connect()
C_DE_DI_AMPA.w = gEI_AMPA / gLeakI

# Connections involving NMDA synapses
# Note that due to the all-to-all connectivity, the contribution of NMDA can be calculated efficiently
selfnmda = brian.Synapses(decisionE, decisionE, 'w:1', on_pre='xpre_post += w', delay=d)
selfnmda.connect(j='i')
selfnmda.w = 1

# Calculate NMDA contributions in each time step
@network_operation()
def update_nmda():

    E1_nmda = np.asarray(decisionE1.spre)
    E2_nmda = np.asarray(decisionE2.spre)
    E3_nmda = np.asarray(decisionE3.spre)

    sE1 = np.sum(E1_nmda)
    sE2 = np.sum(E2_nmda)
    sE3 = np.sum(E3_nmda)
    decisionE1.gen[:] = gEE_NMDA / gLeakE * (w_p * sE1 + w_m * sE2 + w_m * sE3)
    decisionE2.gen[:] = gEE_NMDA / gLeakE * (w_m * sE1 + w_p * sE2 + w_m * sE3)
    decisionE3.gen[:] = gEE_NMDA / gLeakE * (sE1 + sE2 + sE3)
    decisionI.gen[:] = gEI_NMDA / gLeakI * (sE1 + sE2 + sE3)

# Connections involving GABA synapses
C_DI_DE = brian.Synapses(decisionI, decisionE, 'w: 1', on_pre='gi += w', delay=d)
C_DI_DI = brian.Synapses(decisionI, decisionI, 'w: 1', on_pre='gi += w', delay=d)
C_DI_DE.connect()
C_DI_DI.connect()
C_DI_DE.w = gIE_GABA / gLeakE
C_DI_DI.w = gII_GABA / gLeakI

# External inputs
extinputE1 = brian.PoissonGroup(N_D1, rates=nu_ext_1)
extinputE2 = brian.PoissonGroup(N_D2, rates=nu_ext_2)
extinputE3 = brian.PoissonGroup(N_DN, rates=nu_ext)
extinputI = brian.PoissonGroup(NI, rates=nu_ext)

# Connect external inputs
extconnE1 = brian.Synapses(extinputE1, decisionE1, 'w : 1', on_pre='gea += w')
extconnE2 = brian.Synapses(extinputE2, decisionE2, 'w : 1', on_pre='gea += w')
extconnE3 = brian.Synapses(extinputE3, decisionE3, 'w : 1', on_pre='gea += w')
extconnI = brian.Synapses(extinputI, decisionI, 'w : 1', on_pre='gea += w')
extconnE1.connect(j='i')
extconnE2.connect(j='i')
extconnE3.connect(j='i')
extconnI.connect(j='i')
extconnE1.w = gextE / gLeakE
extconnE2.w = gextE / gLeakE
extconnE3.w = gextE / gLeakE
extconnI.w = gextI / gLeakI

# Return the integration circuit
groups = {'DE': decisionE, 'DI': decisionI, 'DX1': extinputE1, 'DX2': extinputE2, 'DX3': extinputE3,
           'DXI': extinputI}
subgroups = {'DE1': decisionE1, 'DE2': decisionE2, 'DE3': decisionE3}
connections = {'selfnmda': selfnmda,
               # 'extconnE1': extconnE1, 'extconnE2': extconnE2, 'extconnE3': extconnE3, 'extconnI': extconnI,
               'C_DE1_DE1_AMPA': C_DE1_DE1_AMPA, 'C_DE1_DE2_AMPA': C_DE1_DE2_AMPA, 'C_DE1_DE3_AMPA': C_DE1_DE3_AMPA,
               'C_DE2_DE1_AMPA': C_DE2_DE1_AMPA, 'C_DE2_DE2_AMPA': C_DE2_DE2_AMPA, 'C_DE2_DE3_AMPA': C_DE2_DE3_AMPA,
               'C_DE3_DE1_AMPA': C_DE3_DE1_AMPA, 'C_DE3_DE2_AMPA': C_DE3_DE2_AMPA, 'C_DE3_DE3_AMPA': C_DE3_DE3_AMPA,
               'C_DE_DI_AMPA': C_DE_DI_AMPA, 'C_DI_DE': C_DI_DE, 'C_DI_DI': C_DI_DI}


mu0 = 40 * Hz
C = 12.8
rhoA = rhoB = mu0 / 100
muA = mu0 + rhoA * C
muB = mu0 - rhoB * C
G_ext_AMPA = 2.1
G_Leak = 25
w_ext = G_ext_AMPA / G_Leak
d = 0.1 * ms

stim_on = 50.0 * ms  # stimulus onset
stim_off = 2000.0 * ms
runtime = 3000.0 * ms

poisson_input1 = brian.PoissonGroup(len(decisionE1), rates=muA)
poisson_input2 = brian.PoissonGroup(len(decisionE2), rates=muB)
# poisson_input1.

C_I1_E1_AMPA = brian.Synapses(poisson_input1, decisionE1, 'w : 1', on_pre='gea += w')
C_I2_E2_AMPA = brian.Synapses(poisson_input2, decisionE2, 'w : 1', on_pre='gea += w')
C_I1_E1_AMPA.connect(j='i')
C_I2_E2_AMPA.connect(j='i')
C_I1_E1_AMPA.w = w_ext
C_I2_E2_AMPA.w = w_ext

myclock = brian.Clock(dt=100 * ms)
@network_operation(myclock)
def update_input(t):
    if t >= stim_on and t < stim_off:
        C_I1_E1_AMPA.w = w_ext
        C_I2_E2_AMPA.w = w_ext
    else:
        C_I1_E1_AMPA.w = 0
        C_I2_E2_AMPA.w = 0


connections['C_I1_E1_AMPA'] = C_I1_E1_AMPA
connections['C_I2_E2_AMPA'] = C_I2_E2_AMPA
groups['poisson_input1'] = poisson_input1
groups['poisson_input2'] = poisson_input2

decisionE.V = -70 * mV
decisionI.V = -70 * mV

# ---- set initial conditions (random)
decisionE.gen = decisionE.gen * (1 + 0.2 * np.random.random(decisionE.__len__()))
decisionI.gen = decisionI.gen * (1 + 0.2 * np.random.random(decisionI.__len__()))
decisionE.V = decisionE.V + np.random.random(decisionE.__len__()) * 2 * mV
decisionI.V = decisionI.V + np.random.random(decisionI.__len__()) * 2 * mV

S_DE1 = brian.PopulationRateMonitor(decisionE1)
S_DE2 = brian.PopulationRateMonitor(decisionE2)

brian.store()

for i in range(16):
    plt.subplot(4, 4, i + 1)
    brian.restore()
    brian.run(runtime, report='stdout')

    plt.plot(S_DE1.t / ms, S_DE1.smooth_rate(window='flat', width=50*ms) / Hz, 'b-')
    plt.plot(S_DE2.t / ms, S_DE2.smooth_rate(window='flat', width=50*ms) / Hz, 'r-')

# thresh_pass1 = (S_DE1.rate > 15 * Hz) * 1
# thresh_pass2 = (S_DE2.rate > 15 * Hz) * 1
# print('reaction time :', (S_DE1.times[np.argmax(thresh_pass1)] * second) - stim_on)
# print('reaction time :', (S_DE2.times[np.argmax(thresh_pass2)] * second) - stim_on)

plt.show()


