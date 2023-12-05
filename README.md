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

The codes to run the exercises can be found in the ```src``` folder, whereas the file [Helpers/HelperFunctions.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/Helpers/HelperFunctions.py?ref_type=heads) contains the definition for the ```Model()``` classes, as well as additonal functions used in the code

# Exercise c

The code for this exercise can be found in [src/solve_part_c.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_c.py). The code can be run using ```parser``` options, which can be accessed with the following command
```shell
python src/solve_part_c.py -h
```

The available options are ```-a, --alpha```, ```-b, --beta``` and ```-n, --nentries```, which allow to change the interval range $[\alpha,\beta]$ and the number of models to be tested. The default options are $\alpha=5$, $\beta=5.6$ and n_entries = 1000. Additionally, a flag ```--plots``` can be used if one wants the plots to be displayed and not only saved.

The code will then generate N=n_entries models, selecting n_entries random values for the parameters $\mathbf{\theta}$, uniformly distributed in a selected range. If the ```--plots``` flag is selected, the code will also produce a plot displaying the pdfs of all the models. This plot is then saved in the ```plots/``` folder, under the name ```Part_c/a_alpha_beta_n_entries.pdf```. It is recommended to not use the flag if the number of entries is greater than 1000, because otherwise the plot is too messy and takes a lot of time to be opened (roughly 85 seconds for 10000).
Aftwerwards, the code will numerically evaluate the integral of the pdf in the $[\alpha,\beta]$ range, and then evaluate the mean and the standard deviation of the n_entries integrals of the pdfs. These two values are then printed out, and then the code is exited.


# Exercise d

The code for this exercise can be found in [src/solve_part_d.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_d.py). This code simply plots the true distributions of signal, background and of the combined pdf on the same canva, fixing the values of $\mathbf{\theta}$. The plot generated can be found in [plots/Part_d/true_pdf.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_d/true_pdf.pdf). Similarly to before, the ```--plots``` flag can be selected if you want to see the plot while running the code.

# Exercise e

The code for this exercise can be found in [src/solve_part_e.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_e.py). For this part, an additional option can be passed using ```-p, --points```, which specifies how many points the user wants to generate, with this number set by default at 100000.

This code generates p points according to the pdf of the Model, using the accept/reject method, and then performs a ML fit in order to estimate the parameters $\mathbf{\theta}$. The code uses the ```minuit``` package, and after the minimisation of the NLL, it prints out the best values of the parameters with their uncertainties, as well as the Hessian matrix for the parameters. As an output, this code plots the generated sample alongside the estimates of signal, background, and total probability all overlaid. This plot can be found in [plots/Part_e/fit.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_e/fit.pdf).  

# Exercise f

The code for this exercise can be found in [src/solve_part_f.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_f.py). This code tests the discovery rate of our model, evaluating how much data must be collected to discover at $5\sigma$ the signal at least 90% of the times. Multiple different values of dataset size were tested, ranging from 10 to 2000, and for each a fixed number of 1000 datasets is generated. For each dataset the "discovery" hypothesis is tested against the "null" hypothesis (background-only), and then the p-value is evaluated using the Neyman-Pearson lemma ($T=-2\Delta \ln \mathcal{L}$). In the final version of the code only 10 values are evaluated, chisen such that it's possible to see the value for which the 90% rate is met. The code prints out the number of failed fits for each number of models, and as expected this number decreases significantly as the dataset size increases, because with more points it's less likely that the fit fails. The code then prints out the discovery rate for each dataset size, and generates a plot of these values ([plots/Part_f/discovery_rates.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_f/discovery_rates.pdf)). Furthermore, the discovery rate values are stored in a numpy array in [data/discovery_rates.npy](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/data/discovery_rates.npy), so that the plot can be generated easily without having to re-run the entire code, which takes some time (roughly 4.5 minutes). For this purpose, a ```--fit``` flag was implemented, which forces the code to re-do the fit step for all the different dataset sizes. This flag is turned off by default, so that the plot can be generated quicker.

# Exercise g

The code for this exercise can be found in [src/solve_part_g.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/src/solve_part_g.py). The structure of this code is basically the same structure of parts e and f, except this time a different model is used. Therefore, there are three options that can be passed via command line, ```-p, --points```, which specifies the number of points to be generated for a single fit to the generated data (plot visible in [plots/Part_g/true_pdf.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_g/true_pdf.pdf)), as well as the usual ```--plots``` and ```--fit```, respectively to show the plots and to do the fit step to evaluate the discovery rate for each dataset size. Similarly to part e), the plot displaying the generated sample alongside the estimates of signal, background, and total probability all overlaid is shown in [plots/Part_g/fit.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_g/fit.pdf), whereas the true pdf distribution, like the one generated in part d, can be found in [plots/Part_g/true_pdf.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_g/true_pdf.pdf). Finally, the discovery rate for different dataset sizes is plot in [plots/Part_g/discovery_rates.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/plots/Part_g/discovery_rate.pdf), and similarly to part f) the numpy array containing the discovery rate values is saved in [data/discovery_rates_g.npy](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/ap2387/-/blob/main/data/discovery_rates_g.npy), due to the code taking roughly 26 minutes to run.