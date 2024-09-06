
import json
import os
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
import scipy
from scipy.io import loadmat
import json
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_sscore, roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
import csv
import surprisal
from surprisal import AutoHuggingFaceModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def split_text(movie_path):
    '''
    Splits the subtitles from a JSON file into words with corresponding timestamps

    Parameters
    ----------
    movie_path : str
        The path to the JSON file containing subtitles

    Returns
    -------
    movie_info : list of tuples
        A list of tuples where each tuple contains a word and its corresponding end time
    '''
    with open(movie_path, 'r') as movie_file:
        movie_json = json.load(movie_file)
    
        movie_subtitles = ' '.join(word_dict['subtitle'] for word_dict in movie_json)
        movie_subtitles = movie_subtitles.split()
        movie_subtitles_time = [word_dict['end_time_new'] for word_dict in movie_json]
        movie_info = [(word, time) for word,time in zip(movie_subtitles, movie_subtitles_time)]

    return movie_info

def chunk_list(text, chunk_size, overlap):
    '''
    Splits the subtitle text into overlapping chunks of equal size

    Parameters
    ----------
    text : list
        The text of substitles
    chunk_size : int
        The number of words in each chunk
    overlap : int
        The number of words that should overlap between consecutive chunks

    Returns
    -------
    list of lists
        A list containing overlapping chunks of the subtitle text
    '''
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]




def bert_llm(text):
    '''
    Computes attention scores for the input text using the BERT model

    Parameters
    ----------
    text : str
        The subtitles for each film

    Returns
    -------
    words: list of tuples
        A list of tuples where each tuple contains each word from the film and its corresponding average attention score
    '''
    # BERT initialisation
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)

    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_ids = inputs['input_ids'][0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Getting attention scores
    outputs = model(**inputs)
    attentions = outputs.attentions

    attention_scores = [att.detach().numpy() for att in attentions]

    # Computing the norm of attention scores across all heads for each layer
    norm_attention_per_layer = [np.linalg.norm(att, axis=1) for att in attention_scores]

    # Computing the average norm across all layers
    norm_attention_all_layers = np.mean(norm_attention_per_layer, axis=0)
    token_attention = norm_attention_all_layers[0, 1:-1, 1:-1]

    # Aggregate attention scores to word level
    words= []
    current_word = ""
    current_att = []

    for i, token in enumerate(tokens[1:-1], 1):
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append((current_word, float(np.mean(current_att))))
            current_word = token
            current_att = []
        current_att.append(np.mean(token_attention[i-1]))

    if current_word:
        words.append((current_word, float(np.mean(current_att))))

    return words



def save_attention(movie_path, overlap):
    '''
    Saves word-level attention scores for the subtitles in the movie

    Parameters
    ----------
    movie_path : str
        Local path to the JSON file containing subtitles for the film
    overlap : int
        The number of words that should overlap between consecutive chunks

    Returns
    -------
    attention_dictionary : dict
        A dictionary with word timings as keys and corresponding attention scores as values
    '''

    # Splitting text into chunks so we can apply BERT
    output = split_text(movie_path)
    
    # Define chunk size and overlap
    chunk_size = 435
    
    # Create overlapping chunks
    words_list = chunk_list(output, chunk_size, overlap)

    attention_dictionary = {}
    
    for chunk in words_list:
        chunked = ' '.join(word[0] for word in chunk)
        chunk_attention = bert_llm(chunked)
        
        for attention, word_time in zip(chunk_attention, chunk):
            if word_time[1] not in attention_dictionary:
                attention_dictionary[word_time[1]] = attention

    with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/overlap_analysis/{overlap}_overlap_word_level_attention_{os.path.basename(movie_path)}.json', 'w') as json_file:
        json.dump(attention_dictionary, json_file)       
        
    return attention_dictionary



def aggregate_to_word_level(tokens, surprisals):
    '''
    Aggregates surprisal values at the token level to the word level

    Parameters
    ----------
    tokens : list
        A list of tokens
    surprisals : list
        A list of surprisal values corresponding to the tokens

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains a word and its corresponding average surprisal value
    '''
    
    word_list = []
    current_word = ""
    current_surp= []

    for token, surprisal in zip(tokens, surprisals):
        if token.startswith("Ġ"): 
          
            if current_word:
                word_list.append((current_word, float(np.mean(current_surp))))
            current_word = token[1:]
            current_surp = [surprisal]
        else:
            current_word += token
            current_surp.append(surprisal)

    if current_word:
        word_list.append((current_word, float(np.mean(current_surp))))

    return word_list


def token_surprisals(movie_path, overlap, chunk_size):
    '''
    Computes word-level surprisal values for the subtitles in the film
    Parameters
    ----------
    movie_path : str
        Local path to the JSON file containing subtitles
    overlap : int
        The number of words that should overlap between consecutive chunks
    chunk_size : int
        The number of words in each chunk

    Returns
    -------
    surprisal_list : list
        A list of surprisal values corresponding to each word in the movie
    '''
    surprisal_list = []
    #initialising surprisal 
    m = AutoHuggingFaceModel.from_pretrained('gpt2')

    # Splitting text into chunks
    output = split_text(movie_path)
    
    # Create overlapping chunks
    words_list = chunk_list(output, chunk_size, overlap)

    
    all_chunks = []
    for chunk in words_list:
        chunked = ' '.join(word[0] for word in chunk)
        chunked = [chunked]
        all_chunks += chunked


    for id, result in enumerate(m.surprise(all_chunks)):

        tokens = result.tokens
        surprisals = result.surprisals
        
        # Filter out infinite surprisals
        filtered_tokens = [token for token, surprisal in zip(tokens, surprisals) if surprisal != float('inf')]
        filtered_surprisals = [surprisal for surprisal in surprisals if surprisal != float('inf')]
        
        # Aggregate surprisal values to word level
        word_level_surprisals = aggregate_to_word_level(filtered_tokens, filtered_surprisals)
  
        chunk_surprisal = []
        for element in word_level_surprisals:
            chunk_surprisal.append(element[1])
            
        if id == 0:
    
            surprisal_list += chunk_surprisal

        if id != 0:
            surprisal_list += chunk_surprisal[overlap:]

    #Getting the timings of each word
    with open(movie_path, 'r') as file:
        movie_info = json.load(file)     
    
    word_timings = []
    for info in movie_info:
        word_timings.append(info['end_time_new'])
            
    surprisal_and_timings = list(zip(surprisal_list, word_timings)) #a list containing surprisal and timing of each word
    print(len(surprisal_and_timings))
    print(len(word_timings))
    print(len(surprisal_list))
 
    with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/surprisal_split', "w") as file:
        for item in surprisal_and_timings:
            file.write(f"{item}\n")   
        
    return surprisal_list
        


def random_forest(fmri_array,targets_list ):
    '''
    Trains a Random Forest model on fMRI data to predict surprisal values and evaluates the model's performance.

    Parameters
    ----------
    fmri_array (np.ndarray) : predictor (BOLD signals)
    targets_list (np.ndarray) : target variables (surprisal)

    Returns
    -------
    mae, mse, r2, rmse (float) : evaluation metricts of our model on the test set
    '''

    # Transpose the fmri_array
    fmri_array = fmri_array.T
    targets_list = targets_list.ravel()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(fmri_array, targets_list, test_size=0.4, random_state=42)

    # Random Forest model
    rf = RandomForestRegressor(random_state=11)

    # Grid search to find the best parameters
    param_grid = {
        'n_estimators': [1200, 1400, 1600, 2000, 2200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 100, 110, 120],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # gridsearch with 10 k-fold
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid.best_params_

    # train the model with the best parameters
    best_rf = grid.best_estimator_

    # Predict on the test set
    y_pred = best_rf.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    return mae, mse, r2, rmse


def language_regions():
    '''
    Filters and saves BOLD signals data by only taking the language-related regions of the brain

    Returns
    -------
    concat_all_info (np.ndarray) : BOLD signals of brain regions only associated with language
    '''
    #Getting only the language areas in the brain from the all info concat matrix
    extended_schaefer_200_data = pd.read_csv('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200 (1).csv')
    
    #creating a subset of regions related to language and speech
    language_related_regions = ["Aud 1", "Aud 2", "Aud 3", "FrOper 1", "FrOper 2", "TempPole 1", "TempPole 2", "TempPole 3", "TempPole 4", "Temp 1", "Temp 2", "Temp 3", "Temp 4", "IPL 1", "IPL 2", "TempPar 1", "TempPar 2", "TempPar 3", "TempPar 4", "IPS 1", "IPS 2", "ParOper 1", "ParMed 1", "ParMed 2", "PrC 1", "Cent 1", "Cent 2"]
    filtered_df = extended_schaefer_200_data[extended_schaefer_200_data['region'].isin(language_related_regions)]

    labels_list = filtered_df['label'].tolist()

    language_indexes = [x - 1 for x in labels_list]
    language_indexes_2 = [element+232 for element in language_indexes]
    language_indexes_3 = [element+232 for element in language_indexes_2]
    all_language_indexes = language_indexes + language_indexes_2+language_indexes_3

    #Loading the full BOLD signals data
    concat_all_info = pd.read_csv('/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/surprisal/allinfo_only_spoken_timepoints.csv', header=None)
    concat_all_info = np.array(concat_all_info)


    #only keeping regions that are involved in language
    concat_all_info = concat_all_info[all_language_indexes, :]

    return concat_all_info



def rnn(fmri_array, targets_list):
    '''
    Trains an RNN model on fMRI data to predict surprisal values and evaluates the model's performance

    Parameters
    ----------
    fmri_array (np.ndarray) : predictor (BOLD signals)
    targets_list (np.ndarray) : target variables (surprisal)

    Returns
    -------
    mae, mse, r2, rmse (float) : evaluation metricts of our model on the test set
    '''
    # Transpose the predictor
    fmri_array = fmri_array.T
    targets_list = targets_list.ravel()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(fmri_array, targets_list, test_size=0.4, random_state=42)

    # standardising data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # reshaping input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

    # Compiling the model with adam and learning rate 0.001
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # training the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    #predicting on test set
    y_pred = model.predict(X_test)

    # evaluating the model 
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mae, mse, r2, rmse