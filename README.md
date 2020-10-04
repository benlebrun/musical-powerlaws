# musical-powerlaws

- `data` contains the csv files with relative rhythmic and melodic bigram frequencies 
- `analysis` contains the code used to test for power-law and heavy-tailed behaviour
  - `bootstrap` contains all bootstrap data for rhythmic and melodic fits
  - `bootstrap.py` contains the code for the semi-parametric bootstrap
  - `distributions.py` contains the code used to fit each distribution - this is essentially a wrapper to the `powerlaw` library
  - `melody.ipynb` contains the parameter estimates, goodness-of-fit tests and likelihood ratio tests for melodic distributions
  - `rhythm.ipynb` contains the parameter estimates, goodness-of-fit tests and likelihood ratio tests for rhythmic distributions
