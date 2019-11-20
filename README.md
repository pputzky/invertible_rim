# Invertible RIM

Official Pytorch implementation of the code accompanying the 2019 NeurIPS paper 
[*Invert to Learn to Invert*](http://papers.nips.cc/paper/8336-invert-to-learn-to-invert)
by Patrick Putzky and Max Welling.

This package supplies methods for defining invertible models in Pytorch, 
a wrapper class that enables *invert to learn* on such invertible models, and
a reference implementation for the i-RIM. A number of examples demonstrate how to 
use the package.

This work was used for the [fastMRI challenge](https://fastmri.org). 
Training and evaluation code for the fastMRI problem will shortly be available 
in another repository. 

## Installation

This package requires pytorch>=1.3. In order to install invertible_rim,
run the following commands 

```bash
git clone https://github.com/pputzky/invertible_rim.git
cd invertible_rim
pip install --user -r requirements.txt
python setup.py install
```

### Package structure
1. **invertible_rim.examples**
    
   A number of examples for demonstrating the usage of this package. 
   The focus is on the usage of invertible_rim.irim.
    
2. **invertible_rim.irim**

   This is the core of the package. It implements all components necessary to 
   build an i-RIM and to train it using *invert to learn*. This module can easily 
   be extended with other invertible layers and modules.
   
3. **invertible_rim.rim**
   
   A reference implementation of the [Recurrent Inference Machines (RIM)](https://arxiv.org/pdf/1706.04008)

4. **invertible_rim.test**
   A number of tests for module invertible_rim.irim. You can run
   ```bash
   pytest
   ```
   to confirm that *invert to learn* is numerically stable on your system.


### Reference
If you use this code or derivatives thereof please cite the associated paper
```bibtex
@incollection{pputzky2019,
title = {Invert to Learn to Invert},
author = {Putzky, Patrick and Welling, Max},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {444--454},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8336-invert-to-learn-to-invert.pdf}
}
```