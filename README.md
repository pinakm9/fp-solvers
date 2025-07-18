# fp-solvers

## Description
This repository contains implementations of various algorithms for solving high-dimensional Fokker-Planck equations as described in https://arxiv.org/pdf/2306.07068 and https://arxiv.org/pdf/2401.01292 .


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
    - [Navigation](#navigation)
    - [Modules](#modules)
- [Tutorials and Replication](#tutorials)
- [License](#license)


## Installation
The code was tested on Python 3.11.12 (Google Colab) and uses tensorflow 2.14.0.
```sh
# Clone the repository
git clone https://github.com/pinakm9/fp-solvers.git
cd repository

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
├── circle-fp/                       # data for ring systems
│   ├── data/
│   │   ├── 10D/                     # solutions for the 10D ring system  
│   │   │   |- ...
│   │   │   ├── 356/                 # model after training 3.56M iterations
│   │   │   ├── 406/                 # model after training 4.06M iterations
│   │   │   ├── 436/                 # model after training 4.36M iterations
│   │   │   ├── 460/                 # model after training 4.6M iterations
|   |   |
│   │   ├── 2D/                      # solutions for the 2D ring system 
│   │   │  
│   │   ├── 4D/                      # pre-trained model(s) for 4D ring system 
│   │   │   
│   │   ├── 6D/                      # pre-trained model(s) for 6D ring system 
│   │   │   
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
│   ├── circle10D.ipynb              # code for computing the steady-state of the 10D ring system with deep learning
│   ├── circle10D_time.ipynb         # code for computing the time-dependent solution for the 10D ring system
│   ├── circle2D.ipynb               # code for computing the steady-state of the 2D ring system with deep learning
│   ├── circle2D_error.ipynb         # code for computing the errors in the steady state solutions (Monte-Carlo and deep learning) for the L63 system
│   ├── circle4D.ipynb               # code for computing the steady-state of the 4D ring system with deep learning
│   ├── circle6D.ipynb               # code for computing the steady-state of the 6D ring system with deep learning
│   ├── circle8D.ipynb               # code for computing the steady-state of the 8D ring system with deep learning
│   ├── circle2D_normalize.ipynb     # code for computing the normalization constant for the steady-state of the 2D ring system as a function of training iterations
│   └── circle2D_sup.ipynb           # code for computing the sup error in the neural network steady state of the 2D ring system as a function of training iterations
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
The main Python classes can be found in the modules folder. The [Navigation](#navigation) section here documents the kind of functionality implemented in each file within the modules folder. Here we point to the three main classes for conducting the experiments reported in the papers. 
#### LogSteadyStateSolver -> modules/lss_solver.py
Learns the steady state of a Fokker-Planck equation by training a neural network. The training happens inside the "learn" method. To see example usage refer to colab/circle2D.ipynb, colab/circle4D.ipynb, colab/circle6D.ipynb, colab/circle8D.ipynb, colab/circle10D.ipynb, colab/L63.ipynb, colab/Thomas.ipynb etc. 
#### MCProb -> modules/sim.py
Computes the Monte-Carlo estimate for solution to a Fokker-Planck equation. compute_p2 is the main method for computing 2D marginal pdfs depicted in the papers. To see example usage refer to colab/L63_comparison.ipynb, colab/Thomas_comparison.ipynb, colab/L63_time.ipynb, colab/Thomas_time.ipynb, colab/circle10D_time.ipynb, colab/circle2D_error.ipynb, colab/L63_filter.ipynb, colab/L63_pf.ipynb etc. 
#### FK32 -> modules/sim.py
Computes the time-dependent solution of 3D Fokker-Planck equations combing a neural network steady state with the Feynman-Kac formula. calc_2D_prob is the main method for computing 2D marginal densities (the "32" in "FK32" refers to the 3D state and 2D marginals). To see example usage refer to colab/L63_time.ipynb, colab/Thomas_time.ipynb etc. Another useful method, in the context of filtering, is calc_2D_prob_weighted which uses the 2D marginals as priors and computes the posteriors using a weight/observation likelihood function. To see example usage refer to see colab/L63_filter.ipynb


## Tutorials and Replication
For tutorials on how to use the modules and perform the experiments reported in the linked papers, please refer to the colab folder. The [Navigation](#navigation) section here documents which experiments are conducted in each notebook within the colab folder. Also refer to the [Modules](#modules) section above to see which Python classes are used in each notebook.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See LICENSE for details.


