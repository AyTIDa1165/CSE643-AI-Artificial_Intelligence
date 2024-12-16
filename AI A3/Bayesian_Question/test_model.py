import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from sklearn.model_selection import train_test_split

def test_model(graph, val_df):
    # Extract CPD states from DAG to verify variable names and states
    defined_states = {}

    # Check if DAG has a model attribute and CPDs are available
    if 'model' in graph and hasattr(graph['model'], 'cpds'):
        for cpd in graph['model'].cpds:
            if hasattr(cpd, 'state_names') and cpd.variable in cpd.state_names:
                # Store the state names for each variable
                defined_states[cpd.variable] = cpd.state_names[cpd.variable]
    else:
        return 0, 0, 0  # Return zero accuracy if no CPDs are found

    # Define valid Start_Stop_ID and End_Stop_ID states based on CPD data
    valid_start_ids = defined_states.get('Start_Stop_ID', [])
    valid_end_ids = defined_states.get('End_Stop_ID', [])

    # Filter test cases from DataFrame
    filtered_test_cases = [
        {
            'Start_Stop_ID': row['Start_Stop_ID'],
            'End_Stop_ID': row['End_Stop_ID'],
            'Distance': row['Distance'],
            'Zones_Crossed': row['Zones_Crossed'],
            'Expected_Fare_Category': row['Fare_Category']
        }
        for _, row in val_df.iterrows()
        if row['Start_Stop_ID'] in valid_start_ids and row['End_Stop_ID'] in valid_end_ids
    ]

    correct_predictions = 0
    total_cases = len(filtered_test_cases)

    # Iterate over each test case for inference
    for evidence in filtered_test_cases:
        evidence_for_inference = {
            'Start_Stop_ID': evidence['Start_Stop_ID'],
            'End_Stop_ID': evidence['End_Stop_ID'],
            'Distance': evidence['Distance'],
            'Zones_Crossed': evidence['Zones_Crossed']
        }

        try:
            # Perform inference
            inference_result = bn.inference.fit(graph, variables=['Fare_Category'], evidence=evidence_for_inference)
            probabilities = inference_result.values

            # Find the predicted category with the highest probability
            predicted_fare_category = inference_result.state_names['Fare_Category'][probabilities.argmax()]

            # Check if the prediction matches the expected category
            if predicted_fare_category == evidence['Expected_Fare_Category']:
                correct_predictions += 1

        except KeyError as e:
            pass  # Ignore KeyErrors in this context
        except Exception as e:
            pass  # Ignore other unexpected errors in this context

    # Calculate accuracy
    accuracy = (correct_predictions / total_cases) * 100 if total_cases > 0 else 0

    return correct_predictions, total_cases, accuracy
