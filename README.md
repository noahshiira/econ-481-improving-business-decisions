# Student Project: Replication of von Zahn et al. (2025)

<a href="https://nbviewer.jupyter.org/github/noahshiira/econ-481-improving-business-decisions/blob/main/replication-notebook.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20"> 
</a> 
<a href="https://mybinder.org/v2/gh/noahshiira/econ-481-improving-business-decisions/main?filepath=replication-notebook.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>

---

This repository contains my replication of the results from [von Zahn, M., Bauer, K., Mihale Wilson, C., Jagow, J., Speicher, M., & Hinz, O. (2025)](https://pubsonline.informs.org/doi/10.1287/mksc.2022.0393).  
**Smart Green Nudging: Reducing Product Returns Through Digital Footprints and Causal Machine Learning.** _Marketing Science_, 44(4), 954–969.

---

## Replication of von Zahn et al. (2025)

The paper investigates how digital “green nudges” can reduce product returns by using causal machine learning and digital footprint data. The authors analyze how customers respond to nudges that highlight environmental impact and provide personalized recommendations, using causal inference methods to estimate treatment effects and identify heterogeneous responses across different customer segments.

In this project, I replicate the key findings of von Zahn et al. (2025) by implementing the same empirical design and causal machine learning methods using Python. Additionally, I extend the analysis by conducting robustness checks and sensitivity analyses to verify the stability of the original results.

---

## This Repository

My replication is conducted using Python and presented in the Jupyter notebook **_replication-notebook.ipynb_**.  
The best way to view the notebook is by downloading this repository. Alternatively, the notebook can be viewed via mybinder or nbviewer by clicking the badges above. Some display features may not render perfectly in these viewers; any missing images or plots can be found in the [_plots_](https://github.com/noahshiira/econ-481-improving-business-decisions/tree/main/plots) folder.

The original paper and the data/code provided by the authors are available [here](https://pubsonline.informs.org/doi/10.1287/mksc.2022.0393).

---

## How to Run

1. Clone the repository  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
