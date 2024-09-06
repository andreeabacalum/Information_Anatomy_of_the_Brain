import synergy_redundancy
import sys, os, subprocess
import numpy as np
import resting_state_synergy_redundancy
from scipy.io import loadmat
# import dynamic_information
import context_predict
import pickle
import pandas as pd
import dynamic_information

#Taking the index for array jobs for HPC
params = range(3, 20)
# params = ['/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/12_years_a_slave_words_new.json', '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/500_days_of_summer_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/back_to_the_future_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/citizenfour_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/little_miss_sunshine_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/pulp_fiction_words_new.json', '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/split_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_prestige_words_new.json', '/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_shawshank_redemption_words_new.json','/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/the_usual_suspects_words_new.json']
index = int(sys.argv[1]) 
i = params[index]

#surprisal
# context_predict.token_surprisals(movie_path='/rds/general/user/ab5621/home/Masters-Dissertation/movie_subtitles/split_words_new.json', overlap=300, chunk_size=500)

#GMM 
labels = dynamic_information.gmm_fit(data_path='/rds/general/user/ab5621/home/Masters-Dissertation/Results/New_GMM/resting_new_concat_all_info.csv', K= 5)
np.savetxt('/rds/general/user/ab5621/home/Masters-Dissertation/Results/New_GMM/resting_new_allinfo_labels.csv', labels, delimiter=',')


#Running the actual job and saving results
unique_data =synergy_redundancy.synergy_redundancy(i=86)

#Saving the redundancy and synergy data for each subject 
np.savetxt("/rds/general/user/ab5621/home/Masters-Dissertation/Results/Synergy_Redundancy_data/unique_corr_86.csv", unique_data, delimiter=',')
np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/synergy_corr_{i}.csv", synergy_data, delimiter=',')
combined_resting_state_data = resting_state_synergy_redundancy.open_resting_state_file('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/BOLD_timeseries_HCP.mat')
dynamic_synergy_matrix, dynamic_redundancy_matrix, dynamic_unique_matrix =resting_state_synergy_redundancy.dynamic_resting_info(i=i, combined_resting_state_data=combined_resting_state_data)
np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_synergy_resting.csv", dynamic_synergy_matrix, delimiter =',')
np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_redundancy_resting.csv", dynamic_redundancy_matrix, delimiter =',')
np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Results/Dynamic_Information/{i}_dynamic_unique_resting.csv", dynamic_unique_matrix, delimiter =',')


# #Saving the redundancy and synergy data for each subject 
np.savetxt("/rds/general/user/ab5621/home/Masters-Dissertation/Results/Synergy_Redundancy_data/resting_unique_corr_76.csv", resting_state_unique, delimiter=',')
np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/resting_synergy_corr_{i}.csv", resting_synergy_data, delimiter=',')

resting_state_entropy_rate_per_timeseries = np.zeros(shape=(100,232))

for row in range(len(combined_resting_state_data)):
    resting_state_entropy_rate_per_timeseries[i-1][row] = synergy_redundancy.entropy_rate(combined_resting_state_data[row])

np.savetxt(f"/rds/general/user/ab5621/home/Masters-Dissertation/Synergy_Redundancy_data/resting_state_entropy_rates.csv", resting_state_entropy_rate_per_timeseries, delimiter=',')

#dynamic information
score = dynamic_information.elbow_method_k_means(dynamic_information_path='/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/concat_all_info.csv', i=i)
with open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/score_{i}', 'w') as file:
    file.write(str(score))


#transition probability marices analysis
dynamic_redundancy_matrix, dynamic_synergy_matrix, dynamic_unique_matrix = dynamic_information.dynamic_information_all_regions(i)
output_syn = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_synergy.pkl', 'wb')
output_red = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_redundancy.pkl', 'wb')
output_uni = open(f'/rds/general/user/ab5621/home/Masters-Dissertation/Dynamic_Information/{i}_dynamic_unique.pkl', 'wb')


pickle.dump(dynamic_redundancy_matrix, output_red)
pickle.dump(dynamic_synergy_matrix, output_syn)
pickle.dump(dynamic_unique_matrix, output_uni)

output_syn.close()
output_red.close()
output_uni.close()


#context analysis

overlap = [5, 55, 105, 155, 205, 255, 305, 355, 405, 455]
for overlap_value in overlap:
    context_predict.save_attention(movie_path = i, overlap = overlap_value)


