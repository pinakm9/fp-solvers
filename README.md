# fp-solvers

## Description
This repository contains implementations of various algorithms for solving high-dimensional Fokker-Planck equations as described in https://arxiv.org/pdf/2306.07068 and https://arxiv.org/pdf/2401.01292 .


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
    - [Navigation](#navigation)
    - [Modules](#modules)
    - [Data](#data)
        - [Pre-trained models](#pre-trained-models)
        - [Training logs](#training-logs)
        - [Miscellaneous](#miscellaneous)
- [Tutorials and Replication](#tutorials-and-replication)
- [License](#license)


## Installation
The code was tested on Python 3.11.13 (Google Colab) and uses tensorflow 2.14.0.
```sh
# Clone the repository
git clone https://github.com/pinakm9/fp-solvers.git
cd fp-solvers

# Install dependencies
pip install -r requirements.txt 
```


## Usage
### Navigation
This repository is structured as follows:
```plaintext 
fp-solvers/
|
|
├── ring-fp/                         # data for ring systems
│   ├── data/
│   │   ├── 10D/                     # solutions for the 10D ring system  
│   │   │   |- ...
│   │   │   ├── 356/                 # model after training 3.56M iterations
│   │   │   ├── 406/                 # model after training 4.06M iterations
│   │   │   ├── 436/                 # model after training 4.36M iterations
│   │   │   ├── 460/                 # model after training 4.6M iterations
|   |   |
│   │   ├── 2D/                      # solutions for the 2D ring system
|   |   ├── 2D-true-vs-learned/      # pre-trained model(s) for 2D ring system for several number of training iterations
│   │   ├── 4D/                      # pre-trained model(s) for 4D ring system 
│   │   ├── 6D/                      # pre-trained model(s) for 6D ring system 
│   │   └── 8D/                      # pre-trained model(s) for 8D ring system 
│   ...       
│  
|
├── non-grad3D/                      # data for non-gradient systems
│   ├── data/
│   │   ├── L63/                     # solutions for the L63 system
│   │   │   ├── 1M/                  # model after training 1M iterations
│   │   │   ├── 800k/                # model after training 800k iterations
│   │   │   ├── time-001/            # time-dependent solutions at t=1e-2
│   │   │   └── time-003/            # time-dependent solutions at t=3e-2
│   │   │   ...
│   │   │
│   │   ├── Thomas/                  # solutions for the Thomas system
│   |   |   └── 400k                 # model after training 400k iterations
|   |   |   ...                      # files containing solutions at various times
| 
│ 
|
├── modules/                         # Python classes for executing the experiments and some visualization
│   ├── arch.py                      # classes for implementing the neural network architectures
│   ├── collage.py                   # code for making a collage from a video by extracting frames
│   ├── compare.py                   # code for comparing the true and learned solutions for the 2D ring system
│   ├── integrator.py                # classes for various quadrature schemes
│   ├── lss_solver.py                # code for learning steady states of Fokker-Planck equation
│   ├── rate.py                      # code for customizing learning rates
│   ├── sde_evolve.py                # Euler-Maruyama solution of SDEs
│   ├── sim.py                       # code for computing Monte-Carlo solution of Fokker-Planck equations as well as solving time-dependent equations using Feynman-Kac
│   ├── utility.py                   # a collection of some helper functions for the experiments
|   ├── test/                        # some code for testing, does not affect the experiments
|
|
├── colab/                           # colab notebooks for reproducing the experimental results in this repository
│   ├── L63.ipynb                    # code for computing the steady-state of the L63 system with deep learning
│   ├── L63_comparison.ipynb         # code for comparing the Monte-Carlo steady state with the neural network solution for the L63 system
│   ├── L63_filter.ipynb             # code for computing the one-step filter with deep learning for the L63 system
│   ├── L63_pf.ipynb                 # code for computing the one-step filter with bootstrap particle filter for the L63 system
│   ├── L63_time.ipynb               # code for computing the time-dependent solution for the L63 system 
│   ├── Thomas.ipynb                 # code for computing the steady-state of the Thomas system with deep learning
│   ├── Thomas_comparison.ipynb      # code for comparing the Monte-Carlo steady state with the neural network solution for the Thomas system
│   ├── Thomas_time.ipynb            # code for computing the time-dependent solution for the Thomas system  
│   ├── ring10D.ipynb                # code for computing the steady-state of the 10D ring system with deep learning
│   ├── ring10D_time.ipynb           # code for computing the time-dependent solution for the 10D ring system
│   ├── ring2D.ipynb                 # code for computing the steady-state of the 2D ring system with deep learning
│   ├── ring2D_error.ipynb           # code for computing the errors in the steady state solutions (Monte-Carlo and deep learning) for the L63 system
│   ├── ring4D.ipynb                 # code for computing the steady-state of the 4D ring system with deep learning
│   ├── ring6D.ipynb                 # code for computing the steady-state of the 6D ring system with deep learning
│   ├── ring8D.ipynb                 # code for computing the steady-state of the 8D ring system with deep learning
│   ├── ring2D_normalize.ipynb       # code for computing the normalization constant for the steady-state of the 2D ring system as a function of training iterations
│   └── ring2D_sup.ipynb             # code for computing the sup error in the neural network steady state of the 2D ring system as a function of training iterations
|
|
├── helpers/                         # some helper code for organizing this repository, has no effect on the experimental results
├── plots/                           # plots describing the experimental results
├── plotters/                        # local jupyter notebooks for generating plots
├── README.md                        # this file
├── LICENSE                          # License for this repository
|...
```
<!--├── one-step-filter/                 # data for one-step filtering experiments for the L63 system
│   ├── L63-dl/                      # data for one-step filters with deep learning and Feynman-Kac, refer to colab/L63_filter.ipynb for generating the relevant data files here
│   ├── L63-pf/                      # data for one-step filters with bootstrap particle filter, refer to colab/L63_pf.ipynb for generating the relevant data files here, subfolders here contain results for various time steps eg 08 refers to making observations every 8 steps
│ --> 


### Modules
The main Python classes can be found in the modules folder. The [Navigation](#navigation) section here documents the kind of functionality implemented in each file within the modules folder. Here we point to the four main classes for conducting the experiments reported in the papers. 
#### LogSteadyStateSolver → modules/lss_solver.py
Learns the steady state of a Fokker-Planck equation by training a neural network. The training happens inside the "learn" method. For example usage, refer to colab/ring2D.ipynb, colab/ring4D.ipynb, colab/ring6D.ipynb, colab/ring8D.ipynb, colab/ring10D.ipynb, colab/L63.ipynb, colab/Thomas.ipynb etc. 
#### MCProb → modules/sim.py
Computes the Monte-Carlo estimate for solution to a Fokker-Planck equation. compute_p2 is the main method for computing 2D marginal pdfs depicted in the papers. For example usage, refer to colab/L63_comparison.ipynb, colab/Thomas_comparison.ipynb, colab/L63_time.ipynb, colab/Thomas_time.ipynb, colab/ring10D_time.ipynb, colab/ring2D_error.ipynb, colab/L63_filter.ipynb, colab/L63_pf.ipynb etc. 
#### FK32 → modules/sim.py
Computes the time-dependent solution of 3D Fokker-Planck equations combining a neural network steady state with the Feynman-Kac formula. calc_2D_prob is the main method for computing 2D marginal densities (the "32" in "FK32" refers to the 3D state and 2D marginals). For example usage, refer to colab/L63_time.ipynb, colab/Thomas_time.ipynb etc. Another useful method, in the context of filtering, is calc_2D_prob_weighted which uses the 2D marginals as priors and computes the posteriors using a weight/observation likelihood function. For example usage, refer to colab/L63_filter.ipynb.
#### LSTMForgetNet → modules/arch.py
This is the main network architecture used for conducting the experiments, and is necessary for reusing all the pre-trained models in this repository. For example usage, refer to colab/ring2D_error.ipynb, colab/L63_comparison.ipynb, colab/Thomas_comparison.ipynb, colab/L63_time.ipynb, colab/Thomas_time.ipynb, colab/ring10D_time.ipynb etc.


### Data
#### Pre-trained models
The trained models are stored in files that are named in the following manner: network.data-00000-of-00001 and network.index where "network" is a placeholder. E.g. the network representing the steady state of the 2D ring system saved after 500 training iterations might be stored in the files ring2D_500.data-00000-of-00001 and ring2D_500.index, see for example ring-fp/data/2D-true-vs-learned. When the training iteration is not a part of the filenames, assume that the files correspond to the final training iteration as suggested in the corresponding [training log](#training-logs). A pre-trained model can be loaded via the load_weights method of the corresponding architecture class. This code snippet shows an example of loading such a model. 
```sh
import arch
save_folder = "fp-solvers/ring-fp/data/2D"
network = arch.LSTMForgetNet(num_nodes=50, num_blocks=3, dtype=tf.float32, name="ring2D")
network.load_weights(f"{save_folder}/{network.name}").expect_partial()
```

#### Training logs
The files named train_log.csv document a brief history of the training. The first, second and third columns store training iteration, value of the loss function at said iteration and elapsed time in seconds, respectively.

#### Miscellaneous 
Monte-Carlo solutions and final computed 2D marginals over a chosen grid are stored in .npy files. Rather than discussing them separately, we refer the reader to the colab folder, where the notebooks document the process of creating such files, elucidating their content. Several auxiliary .csv and .npy files help with bookkeeping for various grid structures and box counting. Their function can be understood by looking at the class files in the modules folder. Since these files are not important for the utility of this repository, we do not discuss them in detail here.

## Tutorials and Replication
For tutorials on how to use the modules and perform the experiments reported in the linked papers, please refer to the colab folder. The [Navigation](#navigation) section here documents which experiments are conducted in each notebook within the colab folder. Also refer to the [Modules](#modules) section above to see which Python classes are used in each notebook.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.


