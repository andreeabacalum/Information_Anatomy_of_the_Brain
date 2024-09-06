import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
# import phyid
# from phyid import calculate, measures
import nilearn
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image
import os



def synergy_redundancy(i):
    '''
    Decomposes BOLD signals into synergy, redundancy and unique information

    Parameters
    ----------
    i : int
        The subject id

    Returns
    -------
    synergy_matrix : ndarray
       The synergy information matrix
    redundancy_matrix : ndarray
       The redundancy information matrix
    unique_matrix : ndarray
       The unique information matrix
    '''
    #Loading BOLD signals for i'th subject
    data=loadmat(f'/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200/sub-{i}/aligned_ts.mat')
    data = data['data']

    #Initialisation
    redundancy_matrix = np.zeros(shape=(232, 232))
    synergy_matrix = np.zeros(shape=(232, 232))
    unique_matrix = np.zeros(shape=(232, 232))

    for i in range(len(data)):
        for j in range(len(data)):
            #Redundant information between brain region i and brain region j
            redundancy_matrix[i][j]=np.mean(phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['rtr'])
          
            #Synergistic information between brain region i and brain region j
            synergy_matrix[i][j]=np.mean(phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['sts'])
            
            #Unique information between brain region i and brain region j
            unique_matrix[i][j]=np.mean(phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['xtx'] + phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['yty'])/2 

    return synergy_matrix, redundancy_matrix, unique_matrix

def all_subjects_synergy_redundancy():
    '''
    Computes synergy and redundancy profiles (matrices) in the brain during movie watching 

    Returns
    -------
    final_redundancy_matrix, final_synergy_matrix (np.ndarray) : synergy and redundancy matrices
    '''
    final_synergy_matrix = np.zeros(shape=(232,232))
    final_redundancy_matrix = np.zeros(shape=(232,232))
    for i in range(1, 87):
        redundant_mat = np.genfromtxt(f'/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/redundancy_corr_{i}.csv', delimiter=',')
        synergy_mat = np.genfromtxt(f'/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/synergy_corr_{i}.csv', delimiter=',')
        final_redundancy_matrix += redundant_mat
        final_synergy_matrix += synergy_mat

    #we take the average synergy and redundancy matrices across all subjects
    final_redundancy_matrix = final_redundancy_matrix / 86
    final_synergy_matrix = final_synergy_matrix / 86

    #plotting the correlation matrix
    final_redundancy_corr_img = nilearn.plotting.plot_matrix(final_redundancy_matrix, title='Redundancy')
    final_synergy_corr_img = nilearn.plotting.plot_matrix(final_synergy_matrix, title='Synergy')

    #Saving the correlation matrix images
    final_redundancy_corr_img.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Plots/redundancy_corr.png', dpi=500)
    final_synergy_corr_img.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Plots/synergy_corr.png', dpi=500)

    return final_redundancy_matrix, final_synergy_matrix


def redundancy_synergy_per_region(redundancy_data, synergy_data):
    '''
    Computes the average redundancy and synergy for each region across subjects

    Parameters
    ----------
    redundancy_data : ndarray
        redundancy matrix for a subject
    synergy_data : ndarray
        synergy matrix for a subject

    Returns
    -------
    average_redundancy_matrix : ndarray
        average redundancy for each region
    average_synergy_matrix : ndarray
        average synergy for each region
    '''
    #averaging across regions
    average_redundancy_matrix = np.mean(redundancy_data, axis=1)
    average_synergy_matrix = np.mean(synergy_data, axis=1)
    return average_redundancy_matrix, average_synergy_matrix

def synergy_minus_redundancy_rank(average_redundancy_matrix, average_synergy_matrix):
    '''
    Computes the difference in rank between synergy and redundancy for each brain region

    Parameters
    ----------
    average_redundancy_matrix : ndarray
        average redundancy for each region
    average_synergy_matrix : ndarray
        average synergy for each region

    Returns
    -------
    synergy_minus_redundancy_rank : ndarray
        difference in rank between synergy and redundancy for each region
    '''
    #Sorting each brain region by synergy and redundancy values
    sorted_synergy = sorted(enumerate(average_synergy_matrix), key=lambda x: x[1])
    sorted_redundancy = sorted(enumerate(average_redundancy_matrix), key=lambda x: x[1])

    # we extract the sorted indices
    sorted_synergy_indices = [x[0] for x in sorted_synergy]
    sorted_redundancy_indices = [x[0] for x in sorted_redundancy]

    # we create dictionaries to map the brain region to its rank after sorting
    synergy_ranks = {region: i for i, region in enumerate(sorted_synergy_indices)}
    redundancy_ranks = {region: i for i, region in enumerate(sorted_redundancy_indices)}

    # Difference in rank
    synergy_minus_redundancy_rank = np.array([synergy_ranks[i] - redundancy_ranks[i] for i in range(232)])
    
    return synergy_minus_redundancy_rank


def network_analysis(rank_path, data_filepath):
    '''
    Analyses the synergy minus redundancy ranks for canonical brain networks

    Parameters
    ----------
    rank_path : str
        local path file file containing the synergy minus redundancy ranks
    data_filepath : str
       local path file containing brain region information

    Returns
    -------
    network_ranks : dict
        A dictionary where the keys are network names and the values are lists of ranks
    '''
    ranks = np.genfromtxt(rank_path)
    df = pd.read_csv(data_filepath)

    #dictionary of networks
    network_categories = {
     'SOM': ['SomMotA', 'SomMotB'],
     'SAL': ['SalVentAttnA', 'SalVentAttnB'],
     'VIS': ['VisCent', 'VisPeri'],
     'DAN': ['DorsAttnA', 'DorsAttnB'],
     'SUB': ['Subcort'],
     'LIM': ['LimbicA', 'LimbicB'],
     'DMN': ['DefaultA', 'DefaultB', 'DefaultC', 'TempPar'],
     'FPN': ['ContA', 'ContB', 'ContC']
    }
    network_list = df.network.tolist()
    network_ranks = {}

    #Splitting each brain region in networks
    for key in network_categories.keys():
        listed = []

        for value in network_categories[key]:
                indices = [index for index, element in enumerate(network_list) if element == value]
                for index in indices:
                    listed.append(ranks[index])
        network_ranks[key]=listed
    return network_ranks

def networks_plot(network_ranks):
    '''
    Plots violin plot for synergy minus redundancy ranks for different brain networks 

    Parameters
    ----------
    network_ranks : dict
        A dictionary where the keys are network names and the values are lists of ranks

    '''
    data = pd.DataFrame([(key, var) for key, values in network_ranks.items() for var in values], columns=['Network', 'Synergy minus Redundancy rank'])

    # Preparing data for each network
    categories = data['Network'].unique()
    data = [data[data['Network'] == category]['Synergy minus Redundancy rank'].values for category in categories]

    # Colourmap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    plt.figure(figsize=(12, 6))

    # Plotting each violin separately
    for idx, (d, color) in enumerate(zip(data, colors)):
        violin_parts = plt.violinplot([d], positions=[idx + 1], showmeans=True, showmedians=True)
        for partname in ('cbars', 'cmins', 'cmaxes'):
            violin_parts[partname].set_edgecolor(color)
        for part in violin_parts['bodies']:
            part.set_facecolor(color)
            part.set_edgecolor(color)

    plt.xticks(range(1, len(categories) + 1), categories, fontsize=14)
    plt.xlabel('Network Category', fontsize=16)
    plt.ylabel('Synergy minus Redundancy rank', fontsize=16)
    plt.show()




def entropy_rate(timeseries):
    '''
    Compute LZ76 entropy rate of a timeseries

    https://ieeexplore.ieee.org/document/1055501

    Parameters
    ----------
    timeseries : array
        1D array of a timeseries

    Returns
    -------
    ent_r : float
    '''

    # if its not already binary, binarise it around the median
    if len(np.unique(timeseries)) > 2:
        median = np.median(timeseries)
        ss = np.where(timeseries > median, 1, 0)

    ss = ss.flatten().tolist()
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(ss)
    while True:
        if ss[i + k - 1] == ss[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
               k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1

    ent_r = (c*np.log2(n))/(n)

    return ent_r




