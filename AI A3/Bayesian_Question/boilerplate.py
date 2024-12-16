#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

import time

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")

    train_df = pd.read_csv("train_data.csv")
    val_df = pd.read_csv("validation_data.csv")
    return train_df, val_df

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    features = ['Route_Type', 'Zones_Crossed', 'Start_Stop_ID', 'End_Stop_ID', 'Distance', 'Fare_Category']
    DAG = []

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            DAG.append((features[i], features[j]))

    model = bn.make_DAG(DAG= DAG)
    # bn.plot(model)
    model = bn.parameter_learning.fit(model, df)
    return model
    
def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""

    features = ['Start_Stop_ID', 'End_Stop_ID', 'Distance', 'Zones_Crossed', 'Route_Type', 'Fare_Category']
    DAG = []
    pruned_DAG = []

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            DAG.append((features[i], features[j]))
 
    model = bn.make_DAG(DAG= DAG)
    adjmat = bn.independence_test(model= model, df= df, prune= True)['adjmat']

    for source in adjmat.index:
        for target in adjmat.columns:
            if adjmat.loc[source, target] != 0:
                pruned_DAG.append((source, target))

    pruned_model = bn.make_DAG(DAG= pruned_DAG)
    # bn.plot(pruned_model)
    pruned_model = bn.parameter_learning.fit(pruned_model, df)
    return pruned_model    

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    optimized_DAG = []
    adjmat = bn.structure_learning.fit(df= df, methodtype= 'hc')['adjmat']

    for source in adjmat.index:
        for target in adjmat.columns:
            if adjmat.loc[source, target] != 0:
                optimized_DAG.append((source, target))

    optimized_model = bn.make_DAG(DAG= optimized_DAG)
    # bn.plot(optimized_model)
    optimized_model = bn.parameter_learning.fit(optimized_model, df)
    return optimized_model    

def save_model(fname, model):
    with open(fname, 'wb') as file:
        pickle.dump(model, file)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    # start_time = time.time()
    base_model = make_network(train_df)
    # end_time = time.time()
    # base_time = end_time - start_time
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    # start_time = time.time()
    pruned_network = make_pruned_network(train_df)
    # end_time = time.time()
    # pruned_time = end_time - start_time
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    
    # start_time = time.time()
    optimized_network = make_optimized_network(train_df)
    # end_time = time.time()
    # optimized_time = end_time - start_time
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

#     print(f'''
# Execution time for base model: {base_time}\n
# Execution time for pruned model: {pruned_time}\n
# Execution time for optimized model: {optimized_time}\n''')

    print("[+] Done")

if __name__ == "__main__":
    main()