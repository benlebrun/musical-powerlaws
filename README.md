# musical-powerlaws

- `data` contains the csv files with relative rhythmic and melodic bigram frequencies 
- `analysis` contains the code used to test for power-law and heavy-tailed behaviour
  - `distributions.py` contains the code used to fit the data to a given distribution
  - `boostrap.py` contains the code used to generate goodness-of-fit statistics
  - `global_rhythm.ipynb` and `global_melody.ipynb` fit the global distributions to each heavy-tail model and assess goodness-of-fit
