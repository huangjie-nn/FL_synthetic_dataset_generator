
import sys
from contrib_calc import Contribution_Calculation
from FL_env import *
from synth_data_prep import *

##################
# Configurations #
##################

dataset_idx = sys.argv[1]
BASE_PATH =  './data/' + str(dataset_idx)

train_path = BASE_PATH + '/data.json'
test_path = BASE_PATH + '/test_data.json'
training_datasets = prep_synth_data(train_path)
testing_dataset = prep_synth_test_data(test_path)

# ============
# Hyperparams for FL model.

#============
# Binary Class hyperparams
binary_model_hyperparams = {
    "batch_size": 45,
    "lr": 0.1,
    "decay": 0.01,
    "rounds":3,
    "epochs": 2,
    "criterion": nn.BCELoss,
    "is_condensed": True
}

#============
# Multi Class hyperparams
multiclass_model_hyperparams = {
    "batch_size": 45,
    "lr": 0.01,
    "decay": 0.1,
    "rounds":1,
    "epochs": 1,
    "criterion": nn.CrossEntropyLoss,
    "is_condensed": False
}

binary_model_structure =  {
    '0':{'layer_size_mapping': {"in_features": 5,
                                "out_features": 64},
        "layer_type": 'linear',
        "activation": 'sigmoid'},
    '1':{'layer_size_mapping': {"in_features": 64,
                                "out_features": 1},
        "layer_type": 'linear',
        "activation": 'sigmoid'},
}

# multiclass_model_structure =  {
#     '0':{"in_features": 20,
#         "out_features": 200,
#         "layer_type": 'linear',
#         "activation": 'sigmoid'},
#     '1':{"in_features": 200,
#         "out_features": 4,
#         "layer_type": 'linear',
#         "activation": 'nil'},
# }

# {'in_channels': 10,
#      'out_channels': 10,
#      'kernel_size': 10,
# }

model_hyperparams = binary_model_hyperparams
model_structure = binary_model_structure

#============
# Set up Federated Learning environment.
# Produce points to the datasets stored on workers
# and ttp. Also produce pointers to workers and ttp.

(training_pointers,
 testing_pointer,
 workers,
 crypto_provider) = setup_FL_env(
    training_datasets,
    testing_dataset
)

#=============
# Produce individual trainlaoders for each client
# which make use of the full data available to them.
trainloaders = {}
for worker_id in list(training_pointers.keys()):
    train = {worker_id:training_pointers[worker_id]}
    #============
    # Convert training datasets into syft dataloaders.
    train_loader, test_loader = convert_to_FL_batches(
        model_hyperparams,
        train,
        testing_pointer
    )
    trainloaders.update({worker_id : train_loader})

trained_model, global_states, client_states, scale_coeffs, global_model_state_dicts = perform_FL_training(
    model_hyperparams,
    model_structure,
    trainloaders,
    workers,
    crypto_provider
)

cc = Contribution_Calculation(global_states, model_hyperparams, client_states, testing_dataset, scale_coeffs)
cc.contribution_calculation('Aggregate')

contributions = cc.aggregate_contribution_matrices()

with open(BASE_PATH + '/Q_dict.json') as json_file:
    Q_dict = json.load(json_file)
with open(BASE_PATH + '/x_stats.json') as json_file:
    x_stats = json.load(json_file)
with open(BASE_PATH + '/params.json') as json_file:
    params = json.load(json_file)

# print(params)
# print(x_stats)
# print(Q_dict)
print(contributions)

import csv
with open(BASE_PATH + '/CC_report_summary.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for param_header in params:
        writer.writerow(["========================"])
        writer.writerow([param_header])
        writer.writerow(["========================"])
        if type(params[param_header]) is list:
            writer.writerow(params[param_header])
        else:
            for param in params[param_header]:
                writer.writerow([param])
                writer.writerow([params[param_header][param]])

    for user in contributions:
        contribution = contributions[user]
        aggregate = contribution["aggregate"]
        alignment_arr = contribution["alignment_arr"]
        accuracy_arr = contribution["accuracy_arr"]
        ROC_arr = contribution["ROC_arr"]
        alignment_component = contribution["alignment_component"]
        accuracy_component = contribution["accuracy_component"]
        ROC_component = contribution["ROC_component"]

        writer.writerow(["========================"])
        writer.writerow([f"User Id: {user}"])
        writer.writerow([f"Aggregate Contribution: {aggregate}"])
        writer.writerow([f"Alignment Contribution: {alignment_arr}"])
        writer.writerow([f"Accuracy Contribution: {accuracy_arr}"])
        writer.writerow([f"ROC Contribution: {ROC_arr}"])
        writer.writerow([f"Summed Alignment Contribution: {alignment_component}"])
        writer.writerow([f"Summed Accuracy Contribution: {accuracy_component}"])
        writer.writerow([f"Summed ROC Contribution: {ROC_component}"])

        # "alignment_arr": alignment,
        # "accuracy_arr": accuracy,
        # "ROC_arr": ROC,
        # "alignment_component": alignment_component,
        # "accuracy_component": accuracy_component,
        # "ROC_component": ROC_component,
        # "num_rounds": num_rounds}

with open(BASE_PATH + '/CC_report_detailed.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['test'])
