#naive bayes classification from scratch

#importing the libraries
import pandas as pd

TRAINING_DATA_PRCT = 0.75 # % of data set used for training
testing_data_prct = 1 - TRAINING_DATA_PRCT # % of data set used for testing
SEED = 99

# Importing the dataset
dataset = pd.read_csv('chess_king_rook_weka_dataset.csv')

#preprocessing categorical 
dwkf = pd.get_dummies(dataset['white_king_file'], prefix = 'wkf')
dwkr = pd.get_dummies(dataset['white_king_rank'], prefix = 'wkr')
dwrf = pd.get_dummies(dataset['white_rook_file'], prefix = 'wrf')
dwrr = pd.get_dummies(dataset['white_rook_rank'], prefix = 'wrr')
dbkf = pd.get_dummies(dataset['black_king_file'], prefix = 'bkf')
dbkr = pd.get_dummies(dataset['black_king_rank'], prefix = 'bkr')

#concat new columns to original dataframe
dataset_concat = pd.concat([dwkf, dwkr, dwrf, dwrr, dbkf, dbkr, dataset['result']], axis = 1)

#encoding class label 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset_concat['result'] = labelencoder.fit_transform(dataset_concat['result'])

pd_data = dataset_concat

#train test
pd_training_data = pd_data.sample(frac=TRAINING_DATA_PRCT, random_state=SEED)
pd_testing_data = pd_data.drop(pd_training_data.index)

# Calculate the number of instances, columns, and attributes in the
# training data set. Assumes 1 column for the instance ID and 1 column
# for the class. Record the index of the column that contains 
# the actual class
no_of_instances_train = len(pd_training_data.index) # number of rows
no_of_columns_train = len(pd_training_data.columns) # number of columns
no_of_attributes = no_of_columns_train - 2
actual_class_column = no_of_columns_train - 1
 
# Store class values in a column, sort them, then create a list of unique
# classes and store in a dataframe and a Numpy array
unique_class_list_df = pd_training_data.iloc[:,actual_class_column]
unique_class_list_df = unique_class_list_df.sort_values()
unique_class_list_np = unique_class_list_df.unique() #Numpy array
unique_class_list_df = unique_class_list_df.drop_duplicates()#Pandas df
 
# Record the number of unique classes in the data set
num_unique_classes = len(unique_class_list_df)
 
# Record the frequency counts of each class in a Numpy array
freq_cnt_class = pd_training_data.iloc[:,actual_class_column].value_counts(
    sort=True)
 
# Record the frequency percentages of each class in a Numpy array
# This is a list of the class prior probabilities
class_prior_probs = pd_training_data.iloc[:,actual_class_column].value_counts(
    normalize=True, sort=True)
 
# Add 2 additional columns to the testing dataframe
pd_testing_data = pd_testing_data.reindex(
                  columns=[*pd_testing_data.columns.tolist(
                  ), 'Predicted Class', 'Prediction Correct?'])
 
# Calculate the number of instances and columns in the
# testing data set. Record the index of the column that contains the 
# predicted class and prediction correctness (1 if yes; 0 if no)
no_of_instances_test = len(pd_testing_data.index) # number of rows
no_of_columns_test = len(pd_testing_data.columns) # number of columns
predicted_class_column = no_of_columns_test - 2
prediction_correct_column = no_of_columns_test - 1
 
######################### Training Phase of the Model ########################
# Create a an empty dictionary
my_dict = {}
 
# Calculate the likelihood tables for each attribute. If an attribute has
# four levels, there are (# of unique classes x 4) different probabilities 
# that need to be calculated for that attribute.
# Start on the first attribute and make your way through all the attributes
for col in range(1, no_of_attributes + 1):
 
    # Record the name of this column 
    colname = pd_training_data.columns[col]
 
    # Create a dataframe containing the unique values in the column
    unique_attribute_values_df = pd_training_data[colname].drop_duplicates()
    # Create a Numpy array containing the unique values in the column
    unique_attribute_values_np = pd_training_data[colname].unique()
     
    # Calculate likelihood of the attribute given each unique class value
    for class_index in range (0, num_unique_classes):
         
        # For each unique attribute value, calculate the likelihoods 
        # for each class
        for attr_val in range (0, unique_attribute_values_np.size) :
            running_sum = 0
 
            # Calculate N(unique attribute value and class value)
            # Where N means "number of" 
            # Go through each row of the training set
            for row in range(0, no_of_instances_train):
                if (pd_training_data.iloc[row,col] == (
                    unique_attribute_values_df.iloc[attr_val])) and (
                    pd_training_data.iloc[row, actual_class_column] == (
                    unique_class_list_df.iloc[class_index])):
                        running_sum += 1
 
            # With N(unique attribute value and class value) as the numerator
            # we now need to divide by the total number of times the class
            # appeared in the data set
            try:
                denominator = freq_cnt_class[class_index]
            except:
                denominator = 1.0
             
            likelihood = running_sum / denominator
             
            # Add a new likelihood to the dictionary
            # Format of search key is 
            # <attribute_name><attribute_value><class_value>
            search_key = str(colname) + str(unique_attribute_values_df.iloc[
                         attr_val]) + str(unique_class_list_df.iloc[
                         class_index])
            my_dict[search_key] = likelihood
  
# Print the likelihood table to the console
# print(pd.DataFrame.from_dict(my_dict, orient='index'))
 
################# End of Training Phase of the Naive Bayes Model ########
 
################# Testing Phase of the Naive Bayes Model ################
 
# Proceed one instance at a time and calculate the prediction
for row in range(0, no_of_instances_test):
 
    # Initialize the prediction outcome
    predicted_class = unique_class_list_df.iloc[0]
    max_numerator_of_bayes = 0.0
 
    # Calculate the Bayes equation numerator for each test instance
    # That is: P(E1|CL0)P(E2|CL0)P(E3|CL0)...P(E#|CL0) * P(CL0),
    # P(E1|CL1)P(E2|CL1)P(E3|CL1)...P(E#|CL1) * P(CL1)...
    for class_index in range (0, num_unique_classes):
 
        # Reset the running product with the class
        # prior probability, P(CL)
        try:
            running_product = class_prior_probs[class_index]
        except:
            running_product = 0.0000001 # Class not found in data set
         
        # Calculation of P(CL) * P(E1|CL) * P(E2|CL) * P(E3|CL)...
        # Format of search key is 
        # <attribute_name><attribute_value><class_value>
        # Record each search key value
        for col in range(1, no_of_attributes + 1):
            attribute_name = pd_testing_data.columns[col]
            attribute_value = pd_testing_data.iloc[row,col]
            class_value = unique_class_list_df.iloc[class_index]
 
            # Set the search key
            key = str(attribute_name) + str(
                      attribute_value) + str(class_value)
 
            # Update the running product
            try:
                running_product *= my_dict[key]
            except:
                running_product *= 0
 
        # Record the prediction if we have a new max
        # Bayes numerator
        if running_product > max_numerator_of_bayes:
            max_numerator_of_bayes = running_product
            predicted_class = unique_class_list_df.iloc[
                         class_index] # New predicted class
 
    # Store the prediction in the dataframe
    pd_testing_data.iloc[row,predicted_class_column] = predicted_class
     
    # Store if the prediction was correct
    if predicted_class == pd_testing_data.iloc[row,actual_class_column]:
        pd_testing_data.iloc[row,prediction_correct_column] = 1
    else: 
        pd_testing_data.iloc[row,prediction_correct_column] = 0
 
# Print the revamped dataframe
print(pd_testing_data)
 
# Print the relevant stats to the console
print()
print("Number of Test Instances : " +
    str(no_of_instances_test))
 
# accuracy = (total correct predictions)/(total number of predictions)
accuracy = (pd_testing_data.iloc[
    :,prediction_correct_column].sum())/no_of_instances_test
accuracy_prcnt = accuracy * 100
s = str(accuracy_prcnt)
print("Accuracy : " + s + "%")
