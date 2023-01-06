# Leap of lognormal: expensive $N$-body simulations from cheap lognormal fields

This repository contains some of the code to reproduce the results in Piras et al., [`"Fast and realistic large-scale structure from machine-learning-augmented random field simulations"`](https://ui.adsabs.harvard.edu/abs/2022arXiv220507898P/abstract). In short, we use a deep learning algorithm to map cheap lognormal random fields to more realistic $N$-body simulations. The dataset of highly-correlated pairs of cheap and expensive simulations is obtained from the [Quijote simulation suite](https://quijote-simulations.readthedocs.io/en/latest/). 

## Content description
In the single folder `scripts`, we provide a few scripts that can be used to reproduce most of the results in the paper. Note that to run them you will need the data byproduct that we generated for this paper, and the Quijote simulations themselves. Get in touch if you need help with any of this.

We provide the following scripts@


- [`isotropic`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/isotropic.ipynb): contains the model training and inference for isotropic (ISO) sources. 
- [`dc`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/dc.ipynb): contains the model training and inference for double couple (DC) sources.
- [`clvd`](https://github.com/alessiospuriomancini/seismoML/blob/main/Piras_2022/clvd.ipynb): contains the model training and inference for compensated linear vector dipole (CLVD) sources.

## Requirements
To run the scripts, beyond the usual `numpy` and ``, you also need:

## Contributing and contacts

Feel free to contact Davide Piras at dr.davide.piras@gmail.com, or [raise an issue here](https://github.com/dpiras/leap_of_lognormal/issues), in case you want to access training data or need help with the code. 

## Citation

If you work with this code or some data byproducts, please cite our paper ([and the Quijote one](https://quijote-simulations.readthedocs.io/en/latest/citation.html)):

    @article{Piras23,
     author = {Piras, D and Joachimi, B and Villaescusa-Navarro, F},
     title = "{Fast and realistic large-scale structure 
               from machine-learning-augmented random field simulations}",
     journal = {Monthly Notices of the Royal Astronomical Society},
     volume = {TBC},
     number = {TBC},
     pages = {TBC},
     year = {2023},
     month = {1},
     issn = {TBC},
     doi = {stad052},
     url = {TBC},
     eprint = {TBC},
    }


