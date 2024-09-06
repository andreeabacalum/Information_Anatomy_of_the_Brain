# Redoing the synergy-redundancy analysis for resting state data
import h5py
import numpy as np
import phyid
from phyid import calculate, measures
import nilearn
from nilearn import plotting, image
import pandas as pd

def open_resting_state_file(resting_state_file_path = '/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/BOLD_timeseries_HCP.mat'):
    '''
    Opens and processes the resting-state fMRI data

    Parameters
    ----------
    resting_state_file_path : str
         path to the HDF5 file containing the resting-state fMRI data

    Returns
    -------
    combined_resting_state_data : ndarray
        the resting state BOLD signals matrix
    '''
    # Open the HDF5 file
    with h5py.File(resting_state_file_path, 'r') as file:
        # acessing the dataset containing object references
        resting_state_dataset = file['BOLD_timeseries_HCP']

        # reading the dataset containing references into an array
        resting_data = resting_state_dataset[()]

        # Initialising a list to store the data arrays
        data_arrays = []

        for ref_array in resting_data:
            for i in range(len(ref_array)):
                ref = ref_array[i]

                # Dereference the HDF5 object reference
                dereferenced_data = file[ref]

                # converting the dataset to a numpy array and appending it to the list
                data_arrays.append(dereferenced_data[:])

        combined_resting_state_data = np.stack(data_arrays, axis=0)
        combined_resting_state_data = combined_resting_state_data.transpose(0, 2, 1)
        
    return combined_resting_state_data



def resting_state_synergy_redundancy(i, combined_resting_state_data):
    '''
    Computes the synergy, redundancy and unique information matrix for one subject for the resting state data

    Parameters
    ----------
    i : int
        subject id
    combined_resting_state_data : ndarray
        the resting-state fMRI data for all subjects

    Returns
    -------
    resting_state_unique_matrix : ndarray
        synergy, redundancy and unique matrices for resting state BOLD signals 
    '''

    #Initializing the resting state redundancy and synergy matrices
    resting_state_redundancy_matrix = np.zeros(shape=(232, 232))
    resting_state_unique_matrix = np.zeros(shape=(232, 232))
    resting_state_synergy_matrix = np.zeros(shape=(232, 232))

    #Taking the fMRi data for the i'th subject
    data = combined_resting_state_data[i]
    for i in range(len(data)):
        for j in range(len(data)):
            #redundnacy between brain region i and j
            resting_state_redundancy_matrix[i][j] = np.mean(phyid.calculate.calc_PhiID(src=data[i], trg=data[j], tau=1, kind='discrete')[0]['rtr'])
            
            #synergy between brain region i and j
            resting_state_synergy_matrix[i][j] = np.mean(phyid.calculate.calc_PhiID(src=data[i], trg=data[j], tau=1, kind='discrete')[0]['sts'])
            
            #unique between brain region i and j
            resting_state_unique_matrix[i][j]=np.mean(phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['xtx'] + phyid.calculate.calc_PhiID(src=data[i], trg = data[j], tau=1, kind='discrete')[0]['yty'])/2 

    return resting_state_synergy_matrix, resting_state_redundancy_matrix, resting_state_unique_matrix


def dynamic_resting_info(i, combined_resting_state_data):
    '''
    Computes dynamic information measures (redundancy, synergy, and unique information) for one subject

    Parameters
    ----------
    i : int
        subject id
    combined_resting_state_data : ndarray
         the resting-state fMRI data for all subjects

    Returns
    -------
    dynamic_synergy_matrix : ndarray
        the synergy matrix over time for a brain region
    dynamic_redundancy_matrix : ndarray
        the redundancy matrix over time for a brain region
    dynamic_unique_matrix : ndarray
        the unique information matrix over time for a brain region
    '''
    data = combined_resting_state_data[i]

    #initialising the matrices
    dynamic_redundancy_matrix = np.zeros(shape=(232, int(len(data[1]))-1))
    dynamic_synergy_matrix = np.zeros(shape=(232, int(len(data[1]))-1))
    dynamic_unique_matrix = np.zeros(shape=(232, int(len(data[1]))-1))

    #For each participant, for each region, for each timestep get an average measure of synergy, unique and redundancy
    # (average as in averages over all their interactions with the rest of the regions).
    for brain_region in range(len(data)):
        dynamic_redundancy_matrix[brain_region]=phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['rtr']
        dynamic_synergy_matrix[brain_region]=phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['sts']
        dynamic_unique_matrix[brain_region]=(phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['xtx'] + phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['yty'])/2 
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_synergy_resting.csv", dynamic_synergy_matrix, delimiter =',')
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_redundancy_resting.csv", dynamic_redundancy_matrix, delimiter =',')
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_unique_resting.csv", dynamic_unique_matrix, delimiter =',')
    return dynamic_synergy_matrix, dynamic_redundancy_matrix, dynamic_unique_matrix

    
            
def resting_state_all_subjects_synergy_redundancy():
        '''
        Averages the synergy and redundancy matrices across all subjects
        '''
        final_resting_state_synergy_matrix = np.zeros(shape=(232,232))
        final_resting_state_redundancy_matrix = np.zeros(shape=(232,232))
        for i in range(100):
            resting_state_redundant_mat = np.genfromtxt(f'/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/resting_redundancy_corr_{i}.csv', delimiter=',')
            resting_state_synergy_mat = np.genfromtxt(f'/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/resting_synergy_corr_{i}.csv', delimiter=',')
            final_resting_state_redundancy_matrix += resting_state_redundant_mat
            final_resting_state_synergy_matrix += resting_state_synergy_mat

        #we take the average synergy and redundancy matrices across all subjects
        final_resting_state_redundancy_matrix = final_resting_state_redundancy_matrix / 100
        final_resting_state_synergy_matrix = final_resting_state_synergy_matrix / 100

        #plotting the correlation matrix
        resting_state_redundancy_corr_img = nilearn.plotting.plot_matrix(final_resting_state_redundancy_matrix, title='Redundancy - Resting State')
        resting_state_synergy_corr_img = nilearn.plotting.plot_matrix(final_resting_state_synergy_matrix, title='Synergy - Resting State')
        #Saving the correlation matrix images
        
        resting_state_redundancy_corr_img.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Plots/resting_state_redundancy_corr.png', dpi=500)
        resting_state_synergy_corr_img.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Plots/resting_state_synergy_corr.png', dpi=500)

        np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/final_resting_state_redundancy_matrix.csv", final_resting_state_redundancy_matrix, delimiter=',')
        np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/final_resting_state_synergy_matrix.csv", final_resting_state_synergy_matrix, delimiter=',') 


def resting_state_redundancy_synergy_per_region(redundancy_data, synergy_data):
    '''
    Computes the average redundancy and synergy for each region across all subjects in the resting-state data

    Parameters
    ----------
    redundancy_data : ndarray
        redundancy matrix for a subject
    synergy_data : ndarray
        synergy matrix for a subject

    Returns
    -------
    resting_average_redundancy_matrix : ndarray
        average redundancy for each region
    resting_average_synergy_matrix : ndarray
        average synergy for each region
    '''
    #averages synergy and redundancy across regions
    resting_average_redundancy_matrix = np.mean(redundancy_data, axis=1)
    resting_average_synergy_matrix = np.mean(synergy_data, axis=1)
    return resting_average_redundancy_matrix, resting_average_synergy_matrix

def synergy_minus_redundancy_rank(resting_average_redundancy_matrix, resting_average_synergy_matrix):
    '''
    Computes the difference in rank between synergy and redundancy for each brain region

    Parameters
    ----------
    resting_average_redundancy_matrix : ndarray
        average redundancy for each region
    resting_average_synergy_matrix : ndarray
        average synergy for each region

    Returns
    -------
    synergy_minus_redundancy_rank : ndarray
        matrix of the difference in rank between synergy and redundancy for each region
    '''
    #sorts the values for synergy and redundancy
    resting_sorted_synergy = sorted(enumerate(resting_average_synergy_matrix), key=lambda x: x[1])
    resting_sorted_redundancy = sorted(enumerate(resting_average_redundancy_matrix), key=lambda x: x[1])

    # we extract the sorted indices
    sorted_synergy_indices = [x[0] for x in resting_sorted_synergy]
    sorted_redundancy_indices = [x[0] for x in resting_sorted_redundancy]

    # we create dictionaries to map the brain region to its rank after sorting
    synergy_ranks = {region: i for i, region in enumerate(sorted_synergy_indices)}
    redundancy_ranks = {region: i for i, region in enumerate(sorted_redundancy_indices)}

    # Difference in rank
    synergy_minus_redundancy_rank = np.array([synergy_ranks[i] - redundancy_ranks[i] for i in range(232)])
    
    return synergy_minus_redundancy_rank

