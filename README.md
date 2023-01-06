# Leap of lognormal: expensive $N$-body simulations from cheap lognormal fields

This repository contains some of the code to reproduce the results in Piras et al., [`"Fast and realistic large-scale structure from machine-learning-augmented random field simulations"`](https://ui.adsabs.harvard.edu/abs/2022arXiv220507898P/abstract). In short, we use a deep learning algorithm to map cheap lognormal random fields to more realistic $N$-body simulations. The dataset of highly-correlated pairs of cheap and expensive simulations is obtained from the [Quijote simulation suite](https://quijote-simulations.readthedocs.io/en/latest/). 


## Content description
In the single folder `scripts`, we provide a few scripts that can be used to reproduce most of the results in the paper. Note that to run them you will need the data byproduct that we generated for this paper, and/or the Quijote simulations themselves. The scripts are not meant to be run "as is". Get in touch if you need help with any of this.

We provide the following scripts:
- [`create_pairs`](https://github.com/dpiras/leap_of_lognormal/blob/main/scripts/create_pairs.py): create the pairs used to train the model. 
- [`fit`](https://github.com/dpiras/leap_of_lognormal/blob/main/scripts/fit.py): train the model. Also uses `utils.py`, `models.py` and `train_functions.py`.
- [`validate_best_model`](https://github.com/dpiras/leap_of_lognormal/blob/main/scripts/validate_best_model.py): select the best epoch model based on the perfomance on validation data. Also uses `models.py` and `test_single_epoch.py`.
- [`final_performance`](https://github.com/dpiras/leap_of_lognormal/blob/main/scripts/final_performance.py): apply the best model on test data and obtain the final dataset. Note this does not perform the summary statistics evaluation or figure generation, which is not present in this repository. Also uses `models.py` and `test_single_epoch.py`.

More information can be found inside each script. For anything that is unclear or missing, get in touch with Davide Piras or raise an issue. Similarly, since no trained model is currently provided, do get in touch if you are interested in using a particular pre-trained model &#151; we will be happy to help!


## Requirements

To run the scripts, beyond the usual `numpy`, `scipy`, `tensorflow` and `matplotlib`, you will also need:
- `CLASS`: you can refer to [these instructions](https://github.com/lesgourg/class_public/wiki/Python-wrapper); only the Python wrapper is used.
- `nbodykit`: you can refer to [these instructions](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).
- `PKLibrary`: this comes from [Pylians3](https://pylians3.readthedocs.io/en/master/), for which you can refer to [these instructions](https://pylians3.readthedocs.io/en/master/installation.html)
- `PiInTheSky` and `healpy`: this is only for the bispectra calculation. The code is not currently publicly available, so you will need to contact Davide Piras for this, or remove the parts of the code where bispectra are calculated (namely the validation part). `healpy` can be installed following [these instructions](https://healpy.readthedocs.io/en/latest/install.html).


## Contributing and contacts

Feel free to contact Davide Piras at dr.davide.piras@gmail.com, or [raise an issue here](https://github.com/dpiras/leap_of_lognormal/issues), in case you want to access training data or need help with the code. 


## Citation

If you work with this code or some data byproducts, please cite our paper ([and the Quijote one](https://quijote-simulations.readthedocs.io/en/latest/citation.html)):

    @article{Piras23,
     author = {Piras, D. and Joachimi, B. and Villaescusa-Navarro, F.},
     title = "{Fast and realistic large-scale structure 
               from machine-learning-augmented random field simulations}",
     journal = {Monthly Notices of the Royal Astronomical Society},
     volume = {TBC},
     number = {TBC},
     pages = {TBC},
     year = {2023},
     month = {1},
     issn = {TBC},
     doi = {10.1093/mnras/stad052},
     url = {TBC},
     eprint = {TBC},
    }


