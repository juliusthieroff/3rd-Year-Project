from __future__ import print_function
import os
import tensorflow as tf
from functools import partial
import pickle
import numpy as np
from machine.rbm.real import RBMReal
from hamiltonian import Ising
from graph import Hypercube
from sampler import Gibbs
from learner import Learner
from logger import Logger
from observable import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#un-hash the observable_array in /logger/logger.py to run script-ising over different values of h.

# System
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#np.random.seed(123)
#tf.set_random_seed(123)

h = 1.0
master_dataset_o = []

h_vals = []
while h < 16:
    observable_array = []
    for iteration in range(1): #specify how many iterations per parameter h you want to average over
        ### Parameters for the graph of the model
        # length of the lattice
        lattice_length = 32
        # dimension
        dimension = 1
        # periodic boundary condition (True) or open boundary condition (False)
        pbc = False

        ### Parameters for the Hamiltonian
        # Type of the Hamiltonian
        hamiltonian_type = "ISING"
        # Parameters of the Hamiltonian
        jz = 2.0

        ### Parameters for the Sampler
        # Number of samples
        num_samples = 10000
        # Number of steps in the sampling process
        num_steps = 1

        ### Parameters for the RBM
        # Density (ratio between the number of hidden and visible nodes)
        density = 2
        # Function to initialise the weight
        initializer = partial(np.random.normal, loc= 0.0, scale=0.01)

        ### Parameters for the Learner
        # Initialise tensorflow session
        sess = tf.Session()
        # Optimiser for the gradient descent
        trainer = tf.train.RMSPropOptimizer
        # Initial learning of the optimiser
        learning_rate = 0.001
        # The number of iterations/epochs for the training
        num_epochs = 10000
        window_period = 200
        # Size of the minibatch
        minibatch_size = 0
        # Threshold value for the stopping criterion
        stopping_threshold = 0.005
        # Initialise reference energy
        reference_energy = None
        # If you want to compare with DMRG value
        use_dmrg_reference = True

        ### Parameters for the Logger
        log = True
        # The path for the result folder
        result_path = './results/'
        # The name of the subpath for your experiment, by default if it is empty it will be named 'cold-start' for cold start
        subpath = ''
        # Indicate whether you want to visualise the weight or visible layer and how frequent
        visualize_weight = False
        visualize_visible = False
        visualize_freq = 10
        # The list of observables that you wish to compute and store in the observable_array which
        # is to be plotted against the parameter
        observables = [MagnetizationZSquareFerro]#MagnetizationZ#, MagnetizationZSquareAntiFerro, CorrelationZ
        # Indicate whether you want to see the weight different after and before training
        weight_diff = True

        #### Create instances from all of the parameters
        graph = Hypercube(lattice_length, dimension, pbc)

        hamiltonian = None
        if hamiltonian_type == "ISING":
            hamiltonian = Ising(graph, jz, h)

        ## Sampler
        sampler = Gibbs(num_samples, num_steps)
        machine = RBMReal(graph.num_points, density, initializer, num_expe=iteration, use_bias=False)
        machine.create_variable()

        if use_dmrg_reference:
            if hamiltonian_type == "ISING":
                refs = pickle.load(open('ising-energy-dmrg.p', 'r'))

            if lattice_length in refs:
                if jz in refs[lattice_length]:
                    reference_energy = float(refs[lattice_length][jz])
                    print('True energy:', reference_energy)

        learner = Learner(sess, graph, hamiltonian, machine, sampler, trainer, learning_rate, num_epochs, minibatch_size,
                          window_period, reference_energy, stopping_threshold, visualize_weight, visualize_visible, visualize_freq)

        logger = Logger(log, result_path, subpath, visualize_weight, visualize_visible, visualize_freq, observables, observable_array, weight_diff)

        learner.learn()
        logger.log(learner)

        # clear previous graph for multiple runs of learner
        tf.compat.v1.reset_default_graph()

        sess.close()

    master_dataset_o.append(np.average(observable_array, axis=None, weights=None, returned=False) / lattice_length**2) #quantizing the Observable - in this case Mz^2 Ferro
    h_vals.append(h)
    if h < 2:
        h = h + 0.1
    elif h < 3:
        h = h + 0.2
    elif h < 16:
        h = h + 4

print(master_dataset_o)
print (observable_array)
print (h_vals)
paramjh = []
for i in h_vals:
    if i != 0:
        paramjh.append(jz / i)

plt.figure()
px = paramjh
oy = master_dataset_o
plt.scatter(px,oy)
plt.plot(px,oy)
plt.title("Parameter vs. Observable")
plt.xlabel("Parameter")
plt.ylabel("Magnetisation Order Parameter Squared")
plt.axvline(x = 1, ymin=0, ymax=1, linewidth=1, color='k', label='QCP')
plt.legend(frameon=False)
plt.show()
plt.tight_layout()
plt.savefig('./results' + '/%s-4,3,1.5-Parameter-Mz2F-%05d.png')
plt.close()

