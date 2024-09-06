#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import entropy
import phyid
from phyid import calculate, measures
import nilearn
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, image
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle
from synergy_redundancy import entropy_rate

def dynamic_information(i):
    '''
    Computes local information atoms(redundancy, synergy, unique) from BOLD signals for a subject

    Parameters
    ----------
    i (int): subject id

    Returns
    -------
    dynamic_redundancy_matrix, dynamic_synergy_matrix, dynamic_unique_matrix (numpy.ndarray) : redundancy, synergy, and unique local values 
    '''
    #Loading the BOLD fMRI movie data for each subject
    data=loadmat(f'/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200/sub-{i}/full_ts.mat')
    data = data['data']

    #initialising the matrices
    dynamic_redundancy_matrix = np.zeros(shape=(232, int(len(data[1]))-1))
    dynamic_synergy_matrix = np.zeros(shape=(232, int(len(data[1]))-1))
    dynamic_unique_matrix = np.zeros(shape=(232, int(len(data[1]))-1))

    #For each participant, for each region, for each timestep get an average measure of synergy, unique and redundancy
    for brain_region in range(len(data)):
        #Redundancy values of the interactions of brain_region with all other brain regions
        dynamic_redundancy_matrix[brain_region]=phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['rtr']

        #Synergy values of the interactions of brain_region with all other brain regions
        dynamic_synergy_matrix[brain_region]=phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['sts']

        #Unique values of the interactions of brain_region with all other brain regions
        dynamic_unique_matrix[brain_region]=(phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['xtx'] + phyid.calculate.calc_PhiID(src=data[i], trg = data[brain_region], tau=1, kind='discrete')[0]['yty'])/2 

    #Saving the matrices locally
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_synergy.csv", dynamic_synergy_matrix, delimiter =',')
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_redundancy.csv", dynamic_redundancy_matrix, delimiter =',')
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_unique.csv", dynamic_unique_matrix, delimiter =',')

    return dynamic_redundancy_matrix, dynamic_synergy_matrix, dynamic_unique_matrix


def concatenate_information(dynamic_synergy_path, dynamic_redundancy_path,dynamic_unique_path):
    '''
    Concatenates each local information values for each atom over all 86 subjects

    Parameters
    ----------
    dynamic_synergy_path, dynamic_redundancy_path, dynamic_unique_path : local paths for the dynamic information for each atom for one subject

    Returns
    -------
    synergy_array, redundancy_array, unique_array (numpy.ndarray) : concatenated information atoms over all subjects
    '''
    #We initialise an empty list for each information atom, to store the local values for each subject
    dynamic_synergy_list = []
    dynamic_redundancy_list = []
    dynamic_unique_list = []

    for i in range(1, 87):
        df_synergy = pd.read_csv(dynamic_synergy_path, header=None)
        df_redundancy = pd.read_csv(dynamic_redundancy_path, header=None)
        df_unique = pd.read_csv(dynamic_unique_path, header=None)
        dynamic_synergy_list.append(df_synergy.values)
        dynamic_redundancy_list.append(df_redundancy.values)
        dynamic_unique_list.append(df_unique.values)

    #Concatenating synergy values over all subjects
    synergy_array = dynamic_synergy_list[0]
    for array in dynamic_synergy_list[1:]:
        synergy_array = np.concatenate((synergy_array, array), axis=1)

    #Concatenating redundancy values over all subjects
    redundancy_array = dynamic_redundancy_list[0]
    for array in dynamic_redundancy_list[1:]:
        redundancy_array = np.concatenate((redundancy_array, array), axis=1)
    
    #Concatenating unique values over all subjects
    unique_array = dynamic_unique_list[0]
    for array in dynamic_unique_list[1:]:
        unique_array = np.concatenate((unique_array, array), axis=1)

    #Saving the concatenated three 2D arrays of shape brain_regions x time, where time is of length of movie * number of participants
    np.savetxt("/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/concat_synergy_resting.csv", synergy_array, delimiter =',')
    np.savetxt("/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/concat_redundancy_resting.csv", redundancy_array, delimiter =',')
    np.savetxt("/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/concat_unique_resting.csv", unique_array, delimiter =',')

    return synergy_array, redundancy_array, unique_array



def brain_information_clustering(dynamic_information_path):

    '''
    Peforms k-means clustering for the concatenated dynamic states

    Parameters
    ----------
    dynamic_information_path : local path of concatenated information values over all subjects

    Returns
    -------
    cluster_labels (list) : a list of labels per timepoint, assigned by the k-means algorithm
    '''
    #reading in the file
    dynamic_information_data = pd.read_csv(dynamic_information_path)
    dynamic_information_data_values = dynamic_information_data.values

    # Normalise the data
    scaler = StandardScaler()
    dynamic_information_data_values = scaler.fit_transform(dynamic_information_data_values)

    # Performing k-means clustering
    kmeans = KMeans(n_clusters=8, random_state=0, n_init='auto', max_iter=500)

    #Getting each datapoint assigned to one of the 8 labels
    cluster_labels = kmeans.fit_predict(dynamic_information_data_values.T)
    
    #Saving the label list locally
    np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/all_info_cluster_labels.csv", cluster_labels, delimiter =',')
    return cluster_labels




def elbow_method_k_means(dynamic_information_path, i):
    '''
    Peforms k-means clustering for the concatenated dynamic states

    Parameters
    ----------
    dynamic_information_path : local path of values for a dynamic information of either synergy, redundancy or unique
    i (int) : number of clusters

    Returns
    -------
    k (float) : inertia of the model for the chosen number of clusters
    '''

    dynamic_information_data = pd.read_csv(dynamic_information_path)
    dynamic_information_data_values = dynamic_information_data.values

    # Normalise the data
    scaler = StandardScaler()
    dynamic_information_data_values = scaler.fit_transform(dynamic_information_data_values)
    dynamic_information_data_values = dynamic_information_data_values.T

    #Dimensionality reduction using PCA
    pca = PCA(n_components=0.9)
    dynamic_information_data_values = pca.fit_transform(dynamic_information_data_values)

    # train the model for current value of k on training data
    model = KMeans(n_clusters = i, random_state = 0, n_init='auto').fit(dynamic_information_data_values)
    k = model.inertia_

    return k


def labels_per_subj(cluster_labels_path, film_length):
    '''
    Chops the arrays of state values back up into individual participants

    Parameters
    ----------
    cluster_labels_path : local path of a list of labels from clustering
    film_length (int) : length of the film in seconds

    Returns
    -------
    labels_per_subj (list) : a list of labels per participant
    '''
    #Reading in the labels 
    cluster_labels_per_subject = pd.read_csv(cluster_labels_path)
    cluster_labels_per_subject = cluster_labels_per_subject.values

    #Splitting the label list for each subject
    labels_per_subj = [cluster_labels_per_subject[i*film_length:(i+1)*film_length] for i in range(86)]

    return labels_per_subj


def fractional_occupancy(information_measure_list):

    '''
    Calculates the fractional occupancy of each state (how much of the total time is spent in it)

    Parameters
    ----------
    cluster_labels_path : local path of a list of labels from clustering

    Returns
    -------
    fractional_occupancy (dictionary) : a dictionary of fractional occupancy for each label
    '''
    total_time = len(information_measure_list)
    occupancy = {}

    for state in information_measure_list:
        if state not in occupancy:
            occupancy[state] = 0
        occupancy[state] += 1

    #Computing fractional occupancy of each state
    fractional_occupancy = {state: count / total_time for state, count in occupancy.items()}

    return fractional_occupancy

def calculate_dwell_time(states):
    '''
    Calculates the dwell time in each state

    Parameters
    ----------
    states (list): list of labels for each timepoint

    Returns
    -------
    average_dwell_times (dictionary) : a dictionary of average dwell time for each label
    '''
    dwell_times = {} 
    current_state = None
    start_time = 0

    for i, state in enumerate(states):
        # Transition to a new state
        if state != current_state:  
            if current_state is not None:
                # Calculate the dwell time for the previous state
                dwell_time = i - start_time
                dwell_times[current_state] = dwell_times.get(current_state, []) + [dwell_time]
            current_state = state
            start_time = i

    # Calculate the dwell time for the last state
    dwell_time = len(states) - start_time
    dwell_times[current_state] = dwell_times.get(current_state, []) + [dwell_time]

    # Calculate the average dwell time for each state
    average_dwell_times = {}
    for state, times in dwell_times.items():
        average_dwell_times[state] = sum(times) / len(times)

    return average_dwell_times

def calculate_appearance_rate(states):
    '''
    Calculates the appearance rate of each state

    Parameters
    ----------
    states (list): list of labels for each timepoint

    Returns
    -------
    appearance_rates (dictionary) : a dictionary of appearance rate for each label
    '''
    # Count the occurrences of each state
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1

    # Calculate the total duration in minutes
    total_seconds = len(states)
    total_minutes = total_seconds / 60

    # Calculate the appearance rate per minute for each state
    appearance_rates = {}
    for state, count in state_counts.items():
        rate_per_minute = (count / total_minutes)
        appearance_rates[state] = rate_per_minute

    return appearance_rates


def transition_probability_matrix(information_measure_list):
    '''
    Calculates the transition probability matrix for a list of labels

    Parameters
    ----------
    information_measure_list (list): list of labels for each timepoint

    Returns
    -------
    transition_matrix (numpy.ndarray) : the transition probability matrix
    '''
    # converting the labels from float to integers
    information_measure_list = [int(x) for x in information_measure_list]
    
    # finding the maximum value to determine the size of the transition matrix
    max_state = max(information_measure_list)
    transition_matrix = np.zeros((max_state + 1, max_state + 1))
    
    for i in range(len(information_measure_list) - 1):
        current_label = information_measure_list[i]
        next_label = information_measure_list[i + 1]
        transition_matrix[current_label][next_label] += 1
    
    # We normalise the rows to get probabilities
    sums = transition_matrix.sum(axis=1)
    zero_rows = np.where(sums == 0)[0]  
    sums[zero_rows] = 1 
    transition_matrix = transition_matrix / sums[:, np.newaxis]
    
    return transition_matrix

def entropy_production(tpm):
    """
    Compute entropy production from a system TPM.
    Using Chris Lynn's method from https://doi.org/10.1073/pnas.210988911

    KL divergence (or relative entropy) between the forward and reverse transition probabilities

    Sum over all bidirectional transitions, 
    of the transition prob * log2(transition prob / transition prob in reverse direction)

    $D_{K L}(\overrightarrow{X} \| \overleftarrow{X}):=\sum_{x_i, x_j \in \mathcal{X}} P\left(x_i \rightarrow x_j\right) \log \left(\frac{P\left(x_i \rightarrow x_j\right)}{P\left(x_j \rightarrow x_i\right)}\right)$
    Parameters
    ----------
    tpm : transition probability matrix
        rows are current state, columns are next state

    Returns
    -------
    entropy : entropy production of the system
    
    """
    entropy = 0 
    
    for i in range(tpm.shape[0]): # for each row
        for j in range(tpm.shape[0]): # for each column
            
            if tpm[i][j] != 0 and tpm[j][i] != 0: # if transition is possible in both directions
                
                entropy += tpm[i][j] * np.log2(tpm[i][j] / tpm[j][i])
    
    return entropy 

def determinism(tpm, norm=False):
    """
    Compute determinism from a system TPM.
    Using Erik Hoel's method from https://doi.org/10.1073/pnas.131492211
  
    $\operatorname{Det}(X)=\frac{\log _2(N)-\left\langle H\left(W_i^{\text {out }}\right)\right\rangle}{\log _2(N)}$
    
    Parameters
    ----------
    tpm : transition probability matrix

    Returns
    -------
    determinism : determinism of the system
    
    """
    
    N = tpm.shape[0] # number of states
    det = 0
    for i in range(tpm.shape[0]): # for each row
        if np.sum(tpm[i]) != 0: # if there are *any* transitions from state i
            det += (entropy(tpm[i], base=2)) # add the entropy of the row to the det score
    
    if norm == False: 
        determinism = np.log2(N) - (det/N)
    elif norm == True:
        determinism = (np.log2(N) - (det/N)) / np.log2(N) 

    return determinism

def degeneracy(tpm, norm=False):
    """
    Compute degeneracy from a system TPM.
    Using Erik Hoel's method from https://doi.org/10.1073/pnas.131492211

    $\operatorname{Deg}(X)=\frac{\log _2(N)-H\left(\left\langle W_i^{\text {out }}\right\rangle\right)}{\log _2(N)}$

    Parameters
    ----------
    tpm : transition probability matrix

    Returns
    -------
    degeneracy : degeneracy of the system

    """

    N = tpm.shape[0] # number of states
    avg = np.mean(tpm, axis=0) # vector of average transition probability across each rows
    deg = entropy(avg, base=2) # entropy of the average transition probability vector
    
    if norm == False:
        degeneracy = np.log2(N) - deg 
    elif norm == True:
        degeneracy =  (np.log2(N) - deg) / np.log2(N)

    return degeneracy

def effectiveness(determinism, degeneracy):
    '''
    Compute effectiveness of a system.
    Using Erik Hoel's method from https://doi.org/10.1073/pnas.131492211

    $\operatorname{Eff}(X)=\frac{\left\langle H\left(W_i^{\text {out }}\right)\right\rangle-H\left(\left\langle W_i^{\text {out }}\right\rangle\right)}{\left\langle H\left(W_i^{\text {out }}\right)\right\rangle}$

    Parameters
    ----------
    tpm : transition probability matrix

    Returns
    -------
    effectiveness : effectiveness of the system

    '''

    return determinism - degeneracy



def save_effectiveness(labels_list):
    '''
    Provides a full analysis of the transition probability matrix system

    Parameters
    ----------
    labels_list (list) : list of labels from clustering

    '''

    tpr = pd.read_csv(labels_list)

    #entropy production of the system
    entropy = entropy_production(tpr)

    #determinism of the system
    determinism = determinism(tpr)

    #degeneracy of a system
    degeneracy= degeneracy(tpr)

    #effectiveness of a system
    effectiveness = effectiveness(determinism =determinism, degeneracy=degeneracy)

    syn_tpr_analysis = {"Entropy Production":entropy, "Determinism":determinism, "Degeneracy":degeneracy, "Effectiveness":effectiveness}

    #Saving them locally
    with open(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/New_GMM/allinfo_tpr_analysis.json", "w") as file:
        json.dump(syn_tpr_analysis, file)
       

def concat_dynamic_info(syn_concat_path, red_concat_path, uni_concat_path):

    '''
    Concatenates all synergy, redundancy and unique local values for each subject accross columns for the purpose of doing 3D clustering

    Parameters
    ----------
    syn_concat_path, red_concat_path, uni_concat_path : local path to information atoms for all subjects

    '''
    df_syn = pd.read_csv(syn_concat_path, header=None)
    df_red = pd.read_csv(red_concat_path, header=None)
    df_uni = pd.read_csv(uni_concat_path, header=None)

    result = pd.concat([df_syn, df_red, df_uni], axis=0)
    result.to_csv('/rds/general/user/ab5621/home/Masters-Dissertation/Results/New_GMM/resting_new_concat_all_info.csv', index=False)


def dynamic_information_all_regions(i):
    '''
    Computes a matrix of dynamic information flow for one subject

    Parameters
    ----------
    i (int) : subject id

    Returns
    -------
    dynamic_redundancy_matrix, dynamic_synergy_matrix, dynamic_unique_matrix : dynamic information atoms for one subject

    '''
    #Loading the BOLD fMRI movie data for each subject
    data=loadmat(f'/rds/general/user/ab5621/home/Masters-Dissertation/extended_schaefer_200/sub-{i}/full_ts.mat')
    data = data['data']

    #initialising the matrices
    dynamic_redundancy_matrix = np.zeros(shape=(232, 232, int(len(data[1]))-1))
    dynamic_synergy_matrix = np.zeros(shape=(232, 232, int(len(data[1]))-1))
    dynamic_unique_matrix = np.zeros(shape=(232, 232,  int(len(data[1]))-1))

    #For each participant, for each region, for each timestep get an average measure of synergy, unique and redundancy
    for brain_region_one in range(len(data)):
        for brain_region_two in range(len(data)):
            dynamic_redundancy_matrix[brain_region_one][brain_region_two]=phyid.calculate.calc_PhiID(src=data[brain_region_one], trg = data[brain_region_two], tau=1, kind='discrete')[0]['rtr']
            dynamic_synergy_matrix[brain_region_one][brain_region_two]=phyid.calculate.calc_PhiID(src=data[brain_region_one], trg = data[brain_region_two], tau=1, kind='discrete')[0]['sts']
            dynamic_unique_matrix[brain_region_one][brain_region_two]=(phyid.calculate.calc_PhiID(src=data[brain_region_one], trg = data[brain_region_two], tau=1, kind='discrete')[0]['xtx'] + phyid.calculate.calc_PhiID(src=data[brain_region_one], trg = data[brain_region_two], tau=1, kind='discrete')[0]['yty'])/2 
    
    return dynamic_redundancy_matrix, dynamic_synergy_matrix, dynamic_unique_matrix



def gmm_fit(data_path, K):
    '''
    Employes Gaussian Mixture Model

    Parameters
    ----------
    data_path : local path of the data we want to cluster
    K (int) : number of clusters for our model

    Returns
    -------
    bic_score (float) : Bayesian Information Criterion (BIC)
    labels (list) : list of labels that GMM assigned for each timepoint

    '''
    #Loading the data
    data= pd.read_csv(data_path)
    data.replace(0, 1e-10, inplace=True)
    data = np.array(data)
    
    #transposing the data so we get a label for each time point after fitting the GMM
    data_transposed = data.T

    #Standardising the data
    scaler = StandardScaler()
    data_transposed = scaler.fit_transform(data_transposed)

    #Dimensionality reduction using PCA
    pca = PCA(n_components=0.9)
    data_transposed = pca.fit_transform(data_transposed)

    #fitting GMM to the data with various number of clusters
    gmm = GaussianMixture(n_components=K, random_state=42, max_iter = 300)
    gmm.fit(data_transposed)

    #Getting a label per timepoint
    labels  = gmm.fit_predict(data_transposed)

    #Computing Bayesian Information criterion (BIC)
    bic_score = gmm.bic(data_transposed)

    return bic_score, labels 




def nbs_compute(i, context_vector_path):
    '''
    Computes the network-based statistic

    Parameters
    ----------
    i (int) : subject id
    context_vector_path : local path to the context vector

    '''
    #Loading the pickle files of 3D dynamic redundancy, synergy and unique of the shape brain_regions x brain_regions x time
    pkl_red = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_redundancy.pkl', 'rb')
    pkl_syn = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_synergy.pkl', 'rb')
    pkl_uni = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_unique.pkl', 'rb')

    data_red = pickle.load(pkl_red)
    data_syn = pickle.load(pkl_syn)
    data_uni = pickle.load(pkl_uni)

    #Converting to numpy arrays
    data_red = np.array(data_red)
    data_syn = np.array(data_syn)
    data_uni = np.array(data_uni)

    #Loading the context vector for the film
    with open(context_vector_path, 'r') as file:
        context_data = json.load(file)
        context_data_values = list(context_data.values())
        context_data_timings = list(context_data.keys())
    
    #We keep only the timings when words were spoken
    indices = [int(float(element) + 1) for element in context_data_timings]
    attention_values = [value[1] for value in context_data_values]
    
    indices = np.array(indices)
    attention_values = np.array(attention_values)

    data_red = data_red[:, :, indices]
    data_syn = data_syn[:, :, indices]
    data_uni = data_uni[:, :, indices]

    #Getting neural based statistic for synergy, redundancy and unique matirces for each subject
    pval_red, adj_red, null_red = NBS_vectorized_correlation.nbs_bct_corr_z(corr_arr =data_red , y_vec = attention_values, thresh = 0.01, k=1000, extent=True, verbose=False)
    pval_syn, adj_syn, null_syn = NBS_vectorized_correlation.nbs_bct_corr_z(corr_arr =data_syn , y_vec = attention_values, thresh = 0.01, k=1000, extent=True, verbose=False)
    pval_uni, adj_uni, null_uni = NBS_vectorized_correlation.nbs_bct_corr_z(corr_arr =data_uni , y_vec = attention_values, thresh = 0.01, k=1000, extent=True, verbose=False)

    


def tpr_analysis(label_path):
    '''
    Provides a full analysis of the transition probability matrix system

    Parameters
    ----------
    label_path : local path of the list of labels

    '''    
    allinfo_tpr = pd.read_csv(label_path, header=None)
    allinfo_tpr = np.array(allinfo_tpr)
    #entropy production of the system
    entropy_allinfo = entropy_production(allinfo_tpr)

    #determinism of the system
    determinism_allinfo = determinism(allinfo_tpr)

    #degeneracy of a system
    degeneracy_allinfo = degeneracy(allinfo_tpr)

    #effectiveness of a system
    effectiveness_allinfo = effectiveness(determinism =determinism_allinfo , degeneracy=degeneracy_allinfo)

    allinfo_tpr_analysis = {"Entropy Production":entropy_allinfo, "Determinism":determinism_allinfo, "Degeneracy":degeneracy_allinfo, "Effectiveness":effectiveness_allinfo}

    #Saving them locally
    with open("/rds/general/user/ab5621/home/Masters-Dissertation/Results/New_GMM/6_clusters_allinfo_tpr_analysis.json", "w") as file:
        json.dump(allinfo_tpr_analysis, file)


def gmm_label_analysis(label_path):
    '''
    Provides a full analysis of the label sequence

    Parameters
    ----------
    label_path : local path of the list of labels

    Returns
    -------
    labels_frac_oc (dict) : dictionary of fractional occupancy of each label
    labels_dwell (dict) : dictionary of dwell time of each label
    labels_app_rate (dict) : dictionary of appearance rate of each label
    labels_tpr (np.ndarray) : transition probability matrix
    '''  
    #Reading in the labels from the 3D GMM clustering
    labels = pd.read_csv(label_path, header=None)
    labels = np.array(labels)


    #Turning the labels from float to int
    labels = [int(element) for element in labels]
    labels = list(labels)

    #Fractional occupancy
    labels_frac_oc = fractional_occupancy(labels)

    #Dwell time
    labels_dwell = calculate_dwell_time(labels)

    #Appearance rate
    labels_app_rate = calculate_appearance_rate(labels)

    #transition probability matrix
    labels_tpr = transition_probability_matrix(labels)

    return labels_frac_oc, labels_dwell, labels_app_rate, labels_tpr




def values_per_label(allinfo_matrix, labels ):
    '''
    Splits up the concatenated data into synergy, redundancy and unique local values

    Parameters
    ----------
    allinfo_matrix (np.ndarray) : matrix of concatenated data
    labels (list) :list of labels

    Returns
    -------
    syn_list (dict) : a dictionary of average synergy values for all points assigned to each cluster
    red_list (dict) : a dictionary of average redundancy values for all points assigned to each cluster
    uni_list (dict) : a dictionary of average unique values for all points assigned to each cluster
    ''' 
    
    #turning the list of labels from floats to ints
    labels = [int(element) for element in labels]
    labels = list(labels) #now we have a list of integers which are labels 0 to 18 - we have 19 clusters

    unique_labels = set(labels) #set of our unique labels, 0 to 18

    syn_list =dict()
    red_list = dict()
    uni_list =dict()

    for label in unique_labels:

        #getting the indices of each label in the labels array
        indices = [index for index, element in enumerate(labels) if element == label]

        allinfo_onelabel = allinfo_matrix[:,indices]

        #getting only the first 232 rows which are synergy data
        syn_matrix = allinfo_onelabel[:232]
        syn_list[label] = syn_matrix

        #getting the next 232 rows which are redundancy data
        red_matrix = allinfo_onelabel[232:464]
        red_list[label] = red_matrix

        #getting the last 232 rows which are synergy data
        uni_matrix = allinfo_onelabel[464:696]
        uni_list[label] = uni_matrix

    return syn_list, red_list, uni_list