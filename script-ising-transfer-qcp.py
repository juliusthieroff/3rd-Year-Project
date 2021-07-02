from __future__ import print_function
import os
import pickle
from functools import partial

import numpy as np
import tensorflow as tf

from graph import Hypercube
from hamiltonian import Ising
from learner import Learner
from logger import Logger
from machine.rbm import RBMTransfer
from machine.rbm.real import RBMReal
from observable import *
from sampler import MetropolisLocal
import matplotlib.pyplot as plt

# System
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#np.random.seed(123)
#tf.set_random_seed(123)

h = 1.0
master_dataset_o = []

h_vals = []
#h_vals = []
#avg_obsv = []
while h < 16:
    observable_array = []
    for iteration in range(1):
        # Graph
        lattice_length = 16
        dimension = 1
        pbc = False
        if pbc:
            pbc_str = 'pbc'
        else:
            pbc_str = 'obc'

        # Hamiltonian
        hamiltonian_type = "ISING"
        #hamiltonian_type = "HEISENBERG"
      #  h = 1.0
        jx = 1.0
        jy = 1.0
        jz = 2.0

        # Sampler
        num_samples = 10000
        num_steps = 1

        # Machine config
        # Rbm
        density = 2
        initializer = partial(np.random.normal, loc= 0.0, scale=0.01)

        # Learner
        sess = tf.compat.v1.Session()
        trainer = tf.compat.v1.train.RMSPropOptimizer
        learning_rate = 0.001
        num_epochs = 10000
        window_period = 200
        minibatch_size = 0
        stopping_threshold = 0.005
        reference_energy = None
        use_dmrg_reference = True

        #### The parameter for the transfer learning method
        # transfer (k,p)-tiling, p is defined automatically
        #k_val = 1  # (1,2)-tiling
        #k_val = 2 # (2,2)-tiling
        k_val = lattice_length / 2# (L,p)-tiling

        # Logger
        log = True
        result_path = './results/'
        subpath = '%d,p_tiling' % k_val
        visualize_weight = False
        visualize_visible = False
        visualize_freq = 10
        observables = [MagnetizationZSquareFerro]#MagnetizationZ, MagnetizationZSquareFerro, MagnetizationZSquareAntiFerro, CorrelationZ]
        weight_diff = True

        # create instances
        graph = Hypercube(lattice_length, dimension, pbc)

        hamiltonian = None
        if hamiltonian_type == "ISING":
            hamiltonian = Ising(graph, jz, h)

        #sampler = Gibbs(num_samples, num_steps)
        sampler = MetropolisLocal(num_samples, num_steps)
        machine = RBMReal(graph.num_points, density, initializer, num_expe=iteration, use_bias=False)

        if hamiltonian_type == "ISING":
            if lattice_length == 8:
                transfer = RBMTransfer(machine, graph, '%sising_%dd_%d_%d_%.2f_1.50_%s/cold-start/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str), iteration)
            else:
                transfer = RBMTransfer(machine, graph, '%sising_%dd_%d_%d_%.2f_1.50_%s/%s/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str, subpath), iteration)
        elif hamiltonian_type == "HEISENBERG":
            if lattice_length == 8:
                transfer = RBMTransfer(machine, graph, '%sheisenberg_%dd_%d_%d_1.00_1.00_%.2f_%s/cold-start/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str), iteration)
            else:
                transfer = RBMTransfer(machine, graph, '%sheisenberg_%dd_%d_%d_1.00_1.00_%.2f_%s/%s/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str, subpath), iteration)

        transfer.tiling(k_val)

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

        logger.visualize_weights(transfer.W_base, logger.result_path, 0, 'before transfer', transfer.learner_base)
        logger.visualize_weights(transfer.W_transfer, logger.result_path, 1, 'after transfer', learner)

        # clear previous graph for multiple runs of learner
        tf.reset_default_graph()


        sess.close()
    master_dataset_o.append(np.average(observable_array, axis=None, weights=None, returned=False) / lattice_length**2) #quantizing the Observable - in this case Mz^2 Ferro
    h_vals.append(h)
    if h < 2:
        h = h + 0.1
    elif h < 3:
        h = h + 0.2
    elif h < 16:
        h = h + 4

    #avg_o = sum(observable_array) / 3.0
    #master_dataset_o.append(avg_o)
print(master_dataset_o)
print (observable_array)
print (h_vals)
order_param = []
for i in h_vals:
    if i != 0:
        order_param.append(jz / i)

plt.figure()
px = order_param
oy = master_dataset_o
plt.scatter(px,oy)
plt.plot(px,oy)
plt.title("Order Parameter vs. Observable")
plt.xlabel("Order Parameter")
plt.ylabel("Mz^2 Ferromagnetic")
plt.axvline(x = 1, ymin=0, ymax=1, linewidth=1, color='k', label='QCP')
plt.legend(frameon=False)
plt.show()
plt.tight_layout()
plt.savefig('./results' + '/%s-transfer8v3-1.5-Order Parameter-observable-%05d.png')
plt.close()

#finding the inflection point
#obs_h = []
#for i in range(len(h_vals)-1):
 #   obs_h.append([h_vals[i], master_dataset_o[i]])
# smooth
#smooth = gaussian_filter(obs_h, 100)

# compute second derivative
#smooth_d2 = np.gradient(np.gradient(smooth))

# find switching points
#infls = np.where(np.diff(np.sign(smooth_d2)))[0]

# plot results
#plt.plot(obs_h, label='Raw Data')
#plt.plot(smooth, label='Smoothed Data')
#plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')
#for i, infl in enumerate(infls, 1):
 #   plt.axvline(x=infl, color='k', label='Inflection Point {i}')
#plt.legend(bbox_to_anchor=(1.55, 1.0))
#plt.show()
#plt.tight_layout()
#plt.savefig('./results' + '/%s-Inflection-%05d.png')
