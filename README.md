# pmf

pmf is a python library for performing inference using hierarchical Poisson
Matrix Factorization, see [Graph link prediction in computer networks using Poisson matrix factorisation](https://arxiv.org/abs/2001.09456).

## Dependencies
pmf requires python 3.7 or higher.

## Installation

To install pmf from the source code

```
git clone https://github.com/mjmt05/pmf.git
cd pmf
pip install .
```

## Documentation and usage

### Examples
Example python script for training a model is provided in the examples folder
```
./train.py -h
```

An example edge list file is also provided, to run the script with default arguments
```
./train.py -f train.txt
```

### Simulation Test
`simulation_test` has a script for simulating from the model, running this will simulate a training and test data set from the model. It performs inference on the training data and assesses predictive performance on the test data set using the area under the ROC curve. 

## Development
Use the python script in `regression_test` to validate any changes to the code. Add to the test when implementing new features.

## Citing

Please use the following bibtex for citing `pmf` in your research:
```
@article{Sanna:2021,
author = {Sanna, Passino F and Turcotte, MJM and Heard, NA},
journal = {Annals of Applied Statistics},
title = {Graph link prediction in computer networks using Poisson matrix  factorisation},
url = {http://arxiv.org/abs/2001.09456},
year = {2021}
}
```

## License
This code is released under the MIT license. 

