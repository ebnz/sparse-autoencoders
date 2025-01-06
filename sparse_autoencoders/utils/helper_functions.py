import torch
from itertools import product

"""
Helpers for Correlation-Calculation
"""
def calculate_correlation_from_kv_dict(kv_dict_gt, kv_dict_simulated):
    """
    Calculates the correlation between two Activation-Samples.
    :param kv_dict_gt: Ground Truth Activation Sample
    :param kv_dict_simulated: Simulated Activation Sample
    :return: Correlation Score of Activation Samples
    """

    datapoints = []
    for key_gt, key_simulated in product(kv_dict_gt, kv_dict_simulated):
        if key_gt == key_simulated:
            datapoints.append([kv_dict_gt[key_gt], kv_dict_simulated[key_simulated]])

    datapoints_tensor = torch.Tensor(datapoints).T
    corr_mat = torch.corrcoef(datapoints_tensor)

    return float(corr_mat[0, 1])


"""
Helpers for Interpretation/Simulation-Parsing
"""
def apply_dict_replacement(inp_str, replacement_dict):
    out_str = inp_str

    for key in replacement_dict.keys():
        out_str = out_str.replace(key, replacement_dict[key])

    return out_str


def remove_leading_asterisk(line):
    if line.startswith("* "):
        return line[2:]
    return line
