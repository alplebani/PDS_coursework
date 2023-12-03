# COURSEWORK Alberto Plebani (ap2387)

README containing instructions on how to run the code for the coursework.

The repository can be cloned with 
```shell
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/s1_assessment/ap2387.git
```

# Anaconda 

The conda environment can be created using the [conda_env.yml](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/conda_env.yml), which contains all the packages needed to run the code
```shell
conda env create -n mphil --file conda_env.yml
```

# Report

The final report is presented in [ap2387.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/ap2387.pdf?ref_type=heads). The file is generated using LaTeX, but all LaTeX-related files are not being committed as per the instructions on the [.gitignore](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/.gitignore?ref_type=heads) file

# Cose structure

The codes to run the exercises can be found in the ```src``` folder, whereas the file [Helpers/HelperFunctions.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/Helpers/HelperFunctions.py?ref_type=heads) contains the definition for the ```Model()``` classes, as well as addiitonal functions used in the code

# Exercise c

The code for this exercise can be found in [src/solve_part_c.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_c.py). The code can be run using ```parser``` options, which can be accessed with the command
```shell
python src/solve_part_c.py -h
```

The available options are ```-a, --alpha```, ```-b, --beta``` and ```-n, --nentries```, which allow to change the interval range $[\alpha,\beta]$ and the number of models to be tested. The default options are $\alpha=5$, $\beta=5.6$ and n_entries = 1000. Additionally, a flag ```--plots``` can be used if one wants the plots to be displayed and not only saved.

The code will then generate n_entries models, selecting n_entries random values for the parameters $\mathbf{\theta}$, uniformly distributed in a selected range. If the ```--plots``` flag is selected, the code will also produce a plot displaying the pdfs of all the models. This plot is then saved in the ```plots/``` folder, under the name ```PartC/a_alpha_beta_n_entries.pdf```. It is recommended to not use the flag if the number of entries is greater than 1000, because otherwise the plot is too messy and takes a lot of time to be opened (roughly 85 seconds for 10000).
Aftwerwards, the code will numerically evaluate the integral of the pdf in the $[\alpha,\beta]$ range, and then return the mean and the standard deviation of the n_entries integrals of the pdfs. These two values are then printed out, and then the code is exited.


# Exercise d

The code for this exercise can be found in [src/solve_part_d.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_d.py). This code simply plots the true distributions of signal, background and of the combined pdf on the same canva, fixing the values of $\mathbf{\theta}$. The plot generated can be found in ```plots/Part_d/true_pdf.pdf```. Similarly to before, the ```--plots``` flag can be selected if you want to see the plot while running the code.

# Exercise e

The code for this exercise can be found in [src/solve_part_e.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_e.py). For this part, an additional option can be passed using ```-p, --points```, which specifies how many points the user wants to generate, with this number set by default at 100000.

This code generated p points according to the pdf of the Model, and then performs a ML fit in order to estimate the parameters $\mathbf{\theta}$. The code uses the ```minuit``` package, and after the minimisation of the NLL, it prints out the best values of the parameters with their uncertainties, as well as the Hessian matrix for the parameters. The code also plots the generated data alongside the true model in ```plots/part_e.pdf``` and the best-fit model in ```plots/fit_e.pdf```.

# TO_DO

Fix background distribution