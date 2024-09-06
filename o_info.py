import jpype as jp
import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
from scipy.stats import pearsonr
import json
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
from sentence_transformers import SentenceTransformer, util
import torch
from scipy.signal import find_peaks
import sys
import jpype
from scipy import stats

def get_oinfo(data):
    """
    Compute the O-information or 'enigmatic information' for a set of time series data.

    Parameters
    ----------
    data : np.ndarray
        A numpy array in time x elements format.
    
    Returns
    -------
    oinfo : float
        O-information
    oinfo_local : np.ndarray
        Local O-information for each time point.
    """

    jarLocation = '/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/infodynamics-dist-1.6.1/infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    T, n_dim = data.shape

    if n_dim > T:
        raise ValueError('Data has more dimensions than timepoints, transpose the data so that it is in the correct format')

    oCalcClass = jpype.JClass('infodynamics.measures.continuous.gaussian.OInfoCalculatorGaussian')
    oCalc = oCalcClass()
    oCalc.initialise(n_dim)
    oCalc.setObservations(data)
    #computes average o-information
    oinfo = oCalc.computeAverageLocalOfObservations()
    #computes local o-information
    oinfo_local = np.array(oCalc.computeLocalOfPreviousObservations())

    return oinfo, oinfo_local


def random_sampling(subject_id, k):
    """
    Perform random sampling of brain regions from the subset of language regions for one subject

    Parameters
    ----------
    subject_id : int
        subject id 
    k : int
        The number of random samples (brain regions) to extract.

    Returns
    -------
    sampled_rows : np.ndarray
        matrix of the randomly sampled brain data rows
    """
    #Getting indexes of where the language brain regions are
    extended_schaefer_200_data = pd.read_csv('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200 (1).csv')

    #Subset of regions involved in langauge
    language_related_regions = ["Aud 1", "Aud 2", "Aud 3", "FrOper 1", "FrOper 2", "TempPole 1", "TempPole 2", "TempPole 3", "TempPole 4", "Temp 1", "Temp 2", "Temp 3", "Temp 4", "IPL 1", "IPL 2", "TempPar 1", "TempPar 2", "TempPar 3", "TempPar 4", "IPS 1", "IPS 2", "ParOper 1", "ParMed 1", "ParMed 2", "PrC 1", "Cent 1", "Cent 2"]

    #getting only the rows corresponding to language brain regions
    filtered_df = extended_schaefer_200_data[extended_schaefer_200_data['region'].isin(language_related_regions)]
    labels_list = filtered_df['label'].tolist()
    language_indexes = [x - 1 for x in labels_list]

    #Loading the brain data
    data = loadmat(f'/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200/sub-{subject_id}/full_ts.mat')['data']
    data = data[language_indexes, :]

    #Getting k random indices from 0 to 231
    random_indices = random.sample(range(54), k)

    # Use the indices to select rows from the matrix
    sampled_rows = data[random_indices, :]
    sampled_rows = sampled_rows.T
    
    return sampled_rows



def pearson_coef(brain_data, attention):
    """
    Compute the Pearson correlation coefficient between BOLD signals and attention values

    Parameters
    ----------
    brain_data : np.ndarray
        The brain activity data
    attention : np.ndarray
        The attention data for each time point

    Returns
    -------
    correlation : float
        The Pearson correlation coefficient
    """
    correlation, p_value = pearsonr(brain_data, attention)
    return correlation



def split_text(movie_path):
    """
    Split the subtitles into chunks of equal size of words

    Parameters
    ----------
    movie_path : str
        JSON file containing movie subtitles

    Returns
    -------
    movie_info : list of tuples
        A list of tuples containing subtitle text and corresponding time
    """
    #Opening the subtitles
    with open(movie_path, 'r') as movie_file:
        movie_json = json.load(movie_file)
        movie_subtitles = ' '.join(word_dict['subtitle'] for word_dict in movie_json)
        movie_subtitles = movie_subtitles.split()

        #getting the start time of the word in the film
        movie_subtitles_time = [word_dict['start_time_new'] for word_dict in movie_json]
        movie_info = [(word, time) for word,time in zip(movie_subtitles, movie_subtitles_time)]

    return movie_info

def split_subjects_by_films():
    """
    Splits subjects into groups based on the films they watched

    Returns
    -------
    all_film_paths_dict : dict
        Dictionary where keys are film lengths and values are lists of subject ids
    """
    #initialising a dictionary with keys as time of the films in seconds
    all_film_paths_dict = {'5470':[], '6804':[],'7715':[],'6674':[], '5900':[], '7515':[], '8882':[], '8181':[], '6739':[], '6102':[]}

    for i in range(1,87):
        data=loadmat(f'/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200/sub-{i}/full_ts.mat')
        data = data['data']
        data = np.mean(data, axis=0)

        #Splitting each subject based on the length of the film that they have watched
        for film_length in all_film_paths_dict.keys():
            if len(data) == int(film_length):
                all_film_paths_dict[film_length].append(i)
    return all_film_paths_dict



def get_corr_oinfo_attention(movie_context_path, subject_id, k):
    """
    Calculate Pearson correlation between o-information and attention values from movie context

    Parameters
    ----------
    movie_context_path : str
        local file of movie context information 
    subject_id : int
        subject id 
    k : int
        The number of random brain regions to sample

    Returns
    -------
    correlation : float
        The Pearson correlation coefficient between o-information and attention values
    """
    with open(movie_context_path, 'rb') as file:
        context_data = pickle.load(file)
        context_data_values = list(context_data.values())
        context_data_timings = list(context_data.keys())

    #We sample k random brain regions from the data
    sampled_data= random_sampling(subject_id=subject_id, k=k)

    #we get oinfo for it
    oinfo_local = get_oinfo(sampled_data)[1] #this has length equal to total time of the film

    #We want to keep only the datapoints where a word was spoken
    oinfo_local = list(oinfo_local)

    #we only want to keep the timings where a word has been spoken from the o-info list
    indices = [int(float(element)) for element in context_data_timings]

    attention_values = [value[1] for value in context_data_values]
    oinfo_local = [oinfo_local[i] for i in indices]


    return pearson_coef(list(oinfo_local), attention_values)


def k_pearson_coef_analysis(k):
    """
    Performs an analysis on pearson correlation coefficients

    Parameters
    ----------
    k : int
        The number of random brain regions to sample for each subject

    Returns
    -------
    pearson_dict : dict
        dictionary with subject ids as keys and pearson's coefficients as values
    """
    pearson_dict = {}
    all_film_paths_dict = split_subjects_by_films()

    #Separating the subjects by the film they have watched
    for film_length in all_film_paths_dict.keys():
        if film_length == '5470':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_500_days_of_summer.pkl'
        if film_length == '6804':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_citizenfour.pkl'
        if film_length == '7715':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_12_years_of_slave.pkl'
        if film_length == '6674':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_back_to_the_future.pkl'
        if film_length == '5900':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_little_miss_sunshine.pkl'
        if film_length == '7515':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_the_prestige.pkl'
        if film_length == '8882':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_pulp_fiction.pkl'
        if film_length == '8181':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_the_shawshank_redemption.pkl'
        if film_length == '6739':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_split.pkl'
        if film_length == '6102':
            movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surp_per_timepoint_the_usual_suspects.pkl'
        id_lists = all_film_paths_dict[film_length]
        for id in id_lists:
            pears_coef = get_corr_oinfo_attention(movie_context_path, subject_id = int(id), k = k)
            pearson_dict[str(id)] = pears_coef
    with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Results/O-info_analysis/{k}_pearson_coef_surprisals.json', 'w') as file:
        json.dump(pearson_dict, file)
    return pearson_dict

def count_negative_oinfo(id):
    """
    Count the number of negative o_info values for different subset sizes

    Parameters
    ----------
    id : int
        subject id

    Returns
    -------
    subset_sizes : range
        range of subset size
    fractions : list
        fractions of negative o-information subsets
    counts : list
        number of negative o-information subsets for each subset size
    """
    fractions = []
    counts = []
    subset_sizes = range(3, 16)
    
    #counting negative o-info values for each value of the subset
    for subset_size in subset_sizes: 
        oinfo_list = []
        for sub in range(subset_size):
            sampled_data = random_sampling(subject_id=id, k=3)
            oinfo = get_oinfo(sampled_data)[0]
            oinfo_list.append(oinfo)

        #counting the negative ones
        negative_oinfo_count = len([element for element in oinfo_list if element < 0])
        total_subsets = len(oinfo_list)
        fraction_negative_oinfo = negative_oinfo_count / total_subsets

        fractions.append(fraction_negative_oinfo)
        counts.append(negative_oinfo_count)
    
    return subset_sizes, fractions, counts


def plot_negative_oinfo(id):
    """
    Plots the fraction and number of negative o-information subsets for different subset sizes

    Parameters
    ----------
    id : int
        subject id
    """
    subset_sizes, fractions, counts = count_negative_oinfo(id)
    
    plt.figure(figsize=(7, 6))
    plt.plot(subset_sizes, fractions, 'o-', color='blue')
    plt.xlabel('Subset Size')
    plt.ylabel('Fraction of Negative Oinfo Subsets')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/rds/general/user/ab5621/home/Masters-Dissertation/Results/Plots/negative_oinfo_fraction_surprisals_{id}.png')
    plt.show()
    
    # Plot for number of negative o-info subsets
    plt.figure(figsize=(7, 6))
    plt.plot(subset_sizes, counts, 'o-', color='blue')
    plt.xlabel('Subset Size')
    plt.ylabel('Number of Negative Oinfo Subsets')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/rds/general/user/ab5621/home/Masters-Dissertation/Results/Plots/negative_oinfo_count_surprisals_{id}.png')
    plt.show()



#Predicting surprisal from o-info of 17 local oinfos from k ranging from 3 to 20
def o_info_random_forest(movie_data):
    """
    Uses Random Forest to predict surprisal values from o-information

    Parameters
    ----------
    movie_data : str
        local file containing movie surprisal data

    Returns
    -------
    mae : float
        Mean absolute error 
    mse : float
        Mean squared error
    r2 : float
        R-squared value 
    rmse : float
        Root mean squared error
    """
    with open(movie_data,'rb') as file:
        surprisal_data = pickle.load(file)
        surprisal_data_values = surprisal_data.values()
        surprisal_data_timings = list(surprisal_data.keys())

    surprisal_data_values = list(surprisal_data_values)
    o_info_list = []
    for k in range(3, 20):

        #We sample k random brain regions from the data
        sampled_data= random_sampling(subject_id=1, k=k)

        #we get oinfo for it
        oinfo_local = get_oinfo(sampled_data)[1] #this has length equal to total time of the film

        #We want to keep only the datapoints where a word was spoken
        oinfo_local = list(oinfo_local)

        #we only want to keep the timings where a word has been spoken from the o-info list
        indices = [int(float(element)) for element in surprisal_data_timings]

        oinfo_local = [oinfo_local[i] for i in indices]

        #append the locala o-info for this k to the list
        o_info_list.append(np.array(oinfo_local))


    #Preparing the predictor
    o_info_list = np.vstack(o_info_list)#concatenating the 17 o-info across the y-axis
    o_info_list = np.array(o_info_list)
    o_info_list = o_info_list.T

    #Preparing the target 
    surprisal_data_values = np.array(surprisal_data_values)
    surprisal_data_values = surprisal_data_values.ravel()

    #splitting data into traim and test set
    X_train, X_test, y_train, y_test = train_test_split(o_info_list, surprisal_data_values, test_size=0.4, random_state=10)


    # Random Forest model
    rf = RandomForestRegressor(random_state=11)

    # Grid search to find the best parameters
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300, 400],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None,5, 10, 20,  30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 6],
        'bootstrap': [True, False]
    }

    # GridSearchCV without KFold, using the train/test split directly
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid.best_params_

    # Train the Random Forest model with the best parameters
    best_rf = grid.best_estimator_

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    return mae, mse, r2, rmse


def peak_analysis():
    """
    Perform peak analysis of local O-information in brain data to compare positive and negative peaks with subtitle similarity

    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_film_paths_dict = split_subjects_by_films()
    for film_length in all_film_paths_dict.keys():
            if film_length == '5470':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/500_days_of_summer_words_new.json'
            if film_length == '6804':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/citizenfour_words_new.json'
            if film_length == '7715':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/12_years_a_slave_words_new.json'
            if film_length == '6674':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/back_to_the_future_words_new.json'
            if film_length == '5900':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/little_miss_sunshine_words_new.json'
            if film_length == '7515':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_prestige_words_new.json'
            if film_length == '8882':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/pulp_fiction_words_new.json'
            if film_length == '8181':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_shawshank_redemption_words_new.json'
            if film_length == '6739':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/split_words_new.json'
            if film_length == '6102':
                movie_context_path = '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_usual_suspects_words_new.json'
            id_lists = all_film_paths_dict[film_length]
            for id in id_lists:
                sampled_data= random_sampling(subject_id=id, k=20)

                #we get oinfo for it
                oinfo_local = get_oinfo(sampled_data)[1] #this has length equal to total time of the film

                oinfo_local = oinfo_local.flatten()

                poz_indices, _ = find_peaks(oinfo_local, height=(0, 15), prominence=6)

                # Find negative peaks (local minima) by inverting the data
                neg_indices, _ = find_peaks(-oinfo_local, height=(0, 15), prominence=6) 


                with open(movie_context_path, 'r') as file:
                    subtitles_json = json.load(file)

                subtitles = [word_dict['subtitle'] for word_dict in subtitles_json]

                movie_subtitles_time = [word_dict['end_time_new'] for word_dict in subtitles_json]
                movie_info = [(subtitle, time) for subtitle,time in zip(subtitles, movie_subtitles_time)]

                poz_similarity_scores = []
                for element in poz_indices:
                    word_list_before = []
                    word_list_after = []
                    for info in movie_info:
                        if element-6<info[1]<=element+2:
                            if info[1]>element-2:
                                word_list_after.append(info[0])
                            else:
                                word_list_before.append(info[0])
                    word_list_after = [" ".join(word_list_after)]
                    word_list_before = [" ".join(word_list_before)]
                    if word_list_after != [''] and word_list_before != ['']:
                        sentences = word_list_before + word_list_after
                        print(sentences)
                        #Compute embedding for both lists
                        embedding_1= model.encode(sentences[0], convert_to_tensor=True)
                        embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

                        similarity = util.pytorch_cos_sim(embedding_1, embedding_2)

                        poz_similarity_scores.append(similarity)

                neg_similarity_scores = []
                for element in neg_indices:
                    word_list_before = []
                    word_list_after = []
                    for info in movie_info:
                        if element-6<info[1]<=element+2:
                            if info[1]>element-2:
                                word_list_after.append(info[0])
                            else:
                                word_list_before.append(info[0])
                    word_list_after = [" ".join(word_list_after)]
                    word_list_before = [" ".join(word_list_before)]
                    if word_list_after != [''] and word_list_before != ['']:
                        sentences = word_list_before + word_list_after
                        print(sentences)
                        #Compute embedding for both lists
                        embedding_1= model.encode(sentences[0], convert_to_tensor=True)
                        embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

                        similarity = util.pytorch_cos_sim(embedding_1, embedding_2)

                        neg_similarity_scores.append(similarity)

                if poz_similarity_scores != []:
                    #getting the mean similarity values for that subject
                    tensor_stack_poz = torch.stack(poz_similarity_scores)
                    mean_value_poz = tensor_stack_poz.mean()
                if neg_similarity_scores!= []:
                    tensor_stack_neg = torch.stack(neg_similarity_scores)
                    mean_value_neg = tensor_stack_neg.mean()

                with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Results/O-info_analysis/{id}_poz_similarity_mean.txt', 'w') as file:
                    file.write(str(mean_value_poz))

                with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Results/O-info_analysis/{id}_neg_similarity_mean.txt', 'w') as file:
                    file.write(str(mean_value_neg))





def t_test_peaks(poz_list, neg_list):
    '''
    Computes a paired t-test to analyse if the means between the semantic similarities of positive and negative peaks are different

    Parameters
    ----------
    poz_list : list
            list of the semantic similarities for the words surrounding each positive peak
    neg_list : list
            list of the semantic similarities for the words surrounding each negative peak
    '''
    # Perform the paired t-test
    t_statistic, p_value = stats.ttest_rel(poz_list, neg_list)

    #confidence level of 95%
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The means are significantly different.")
    else:
        print("Fail to reject the null hypothesis: The means are not significantly different.")


def plot_peaks(oinfo_local):
    '''
    Function which plots o-info and marks the positive and negative peaks
    '''
    # Find positive peaks 
    positive_peaks, _ = find_peaks(oinfo_local, height=(0, 15), prominence=6)

    # Find negative peaks
    negative_peaks, _ = find_peaks(-oinfo_local, height=(0, 15), prominence=6) 


    plt.figure(figsize=(15, 8))
    plt.plot(oinfo_local)

    # Plot positive peaks
    plt.plot(positive_peaks, oinfo_local[positive_peaks], "x", label="Positive Peaks", markersize = 10)

    # Plot negative peaks
    plt.plot(negative_peaks, oinfo_local[negative_peaks], "o", label="Negative Peaks", color="red", markersize = 10)

    #plotting a line at 0 for reference
    plt.plot(np.zeros_like(oinfo_local), "--", color="black", linewidth=1)
    plt.xlabel('Time (Seconds)', fontsize=18)
    plt.ylabel('O-Info Value', fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.show()