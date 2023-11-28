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

# Exercise c

The code for this exercise can be found in [src/solve_part_c.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_c.py). The code can be run using ```parser``` options, which can be accessed with the command
```shell
python src/solve_part_c.py -h
```

The available options are ```-a, --alpha```, ```-b, --beta``` and ```-n, --nentries```, which allow to change the interval range $[\alpha,\beta]$ and the number of models to be tested. The default options are $\alpha=5$, $\beta=5.6$ and n_entries = 1000.

The code will then generate n_entries models, selecting n_entries random values for the parameters $\vect{\theta}$, uniformly distributed in a selected range. If the number of models is smaller than 50, the code will also produce a plot displaying the pdfs of all the models. This plot is then saved in the ```plots\``` folder, under the name ```PartC_alpha_beta_n_entries.pdf```. The upper limit of 50 was selected because for greater values the plot is too messy and it takes too long to open.
Aftwerwards, the code will numerically evaluate the integral of the pdf in the $[\alpha,\beta]$ range, and then return the mean and the standard deviation of the n_entries integrals of the pdfs. These two values are then printed out, and then the code is exited.


# Exercise d

The code for this exercise can be found in [src/solve_part_c.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_c.py). This code simply plots the true distributions of signal, background and of the combined pdf on the same canva, fixing the values of $\vect{\theta}$. The plot generated can be found in ```plots\true_pdf.pdf```

# Exercise e

The code for this exercise can be found in [src/solve_part_c.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_c.py). For this part, an additional option can be passed using ```-p, --points```, which specifies how many points the user wants to generate, with this number set by default at 100000.

