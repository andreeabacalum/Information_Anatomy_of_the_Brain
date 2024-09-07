# Information Anatomy of the Human Brain

## Table of Contents
- [Project Description](#project-description)
- [Data](#data)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)


## Project Description

This project applies the Integrated Information Decomposition framework to movie-watching fMRI data, to understand how the brain processes complex real-world stimuli, offering insights into the dynamic flow of information within the brain during movie-watching. We are breaking down the intrinsic information flow in the brain using BOLD signals of 86 subjects who watched full-length films inside the fMRI, to examine brain activity patterns that are directly relevant to how the brain functions in everyday life. We treat the brain as a dynamic system where its future states are shaped by the current states of each brain region and the interactions between them. We explore the differences between how the brain structures its information when it is at rest, versus when it watches a film. We seek to understand the informational states the brain goes in most frequently during movie-watching, and analyse these patterns to further understand brain dynamics. Lastly, we explore the relationship between linguistic context comprehension and informational architecture of the brain during films.

## Data 

The data comes from 86 people who completed a series of behavioral assessments and watched a full-length movie while while functional magnetic resonance imaging (fMRI) was recorded

The data is publicly available at: [https://www.naturalistic-neuroimaging-database.org/](url)

## Tech Stack

The entire project was built using Python

## Installation

Requirements:
- Python 3
- Pip 3

1. Clone the repository:
```
 git clone https://github.com/andreeabacalum/Information_Anatomy_of_the_Brain.git
```

2. Installing dependencies: 

- Install the phyid library used for the computation of Integrated Information Decomposition (ΦID):
```
 pip install git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git
```
- Install all dependencies mentioned in requirements.txt file by running:
```
 pip3 install -r requirements.txt
```

## Usage

Below is an overview of the main code components in this project:

- **synergy_redundancy.py** : uses Integrated Information Decomposition to decompose BOLD signals of subjects watching films into synergy and redundancy atoms and conduct analyses on them
- **resting_state_synergy_redundancy.py** : uses Integrated Information Decomposition to decompose BOLD signals of subjects at rest into synergy and redundancy atoms and conduct analyses on them
- **data_analysis.ipynb** : conducts exploratory data analysis on the movie data
- **dynamic_information.py** : uses clustering methods to identify most frequent states the brain goes in during movie-watching
-**dynamic_states.ipynb** : conducts further analyses on the brain's most frequent informational states
- **brain_plot.py** : Plots cortical surface and subcortical MNI volume
- **brain_topology.ipynb** : Conducts network topology analyses (specifically modularity and global efficiency) of the whole-brain synergy, redundancy and unique networks
- **context_predict.py** : conducts supervised learning to predict context/surprisal based on BOLD signals and information atoms from BOLD signals
- **supervised_learning.ipynb** : constructs the target and predictor for supervise learning to predict context from BOLD signals and information atoms from BOLD signals
- **o_info.py** : computes o-information and conducts analyses on it to observe if informational architecture in the brain correlates to the level of predictability in the film's subtitle sequence


Below is an overview of the main result folders in this project:

- **Synergy_Redundancy_data** : results from the Integrated Information Decomposition analyses for both resting-state and movie-watching data. Contains synergy, redundancy and unique matrices for al subjects, average synergy-minus-redundancy rank gradient matrices for both movie-watching and resting states
- **Plots** : contains all plots from the project report
- **New_GMM** : contains all results from the Gaussian Mixture Model clustering to find brain microstates; Contains list of labels assigned for each timepoint by the GMM; Contains label analyses measures and transition probability matrix system analyses for both resting and movie-watching states
- **O-info_analysis** : Contains average sentence similarity scores for the words surrounding each positive (poz_similarity files) and negative (neg_similarity files) Local O-Information peak 

