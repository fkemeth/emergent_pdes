# Learning Partial Differential Equations in Emergent Coordinates

This package builds upon the lpde python package (see [this GitHub repo](https://github.com/fkemeth/lpde)) to learn partial differential equations in emergent coordinates.

USAGE
---------


Via source

    git clone https://github.com/fkemeth/emergent_pdes
    cd emergent_pdes/
    pip install -r requirements.txt

This repository contains five different examples:

- `cgle_periodic`: Contains code to learn a partial differental equation describing the periodic dynamics in the complex Ginzburg-Landau equation.
- `cgle_chaotic`: Contains code to learn a partial differental equation describing the chaotic dynamics in the complex Ginzburg-Landau equation.
- `sle_periodic`: Contains code to learn a partial differental equation describing the periodic dynamics in an ensemble of Stuart-Landau oscillators.
- `sle_gamma`: Contains code to learn a partial differental equation describing the dynamics in an ensemble of Stuart-Landau oscillators for different parameter values.
- `preboetzinger`: Contains code to learn a partial differential equation describing the periodic dynamics of a biologically-motivated system of coupled neurons.

Each example can be run using

    python run.py

in the respective example folders.

- Configuration files are contained in the `config/` subdirectories.
- Some examples contain testing scripts in `tests.py`

NOTE
---------

This work is under constant development and might undergo significant changes in the future!

ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repo](https://github.com/fkemeth/lpde).


LICENCE
---------

This work is licenced under MIT License.
Please cite

"Learning emergent partial differential equations
in a learned emergent space"
F.P. Kemeth et al.
*Nature Communications* 13, Article number: 3318 (2022)
(https://www.nature.com/articles/s41467-022-30628-6)

if you use this package for publications.
