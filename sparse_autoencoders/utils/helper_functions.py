import torch
from numpy import isnan
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

    if len(datapoints) == 0:
        return 0

    datapoints_tensor = torch.Tensor(datapoints).T
    corr_mat = torch.corrcoef(datapoints_tensor)

    return float(corr_mat[0, 1]) if not isnan(float(corr_mat[0, 1])) else 0


"""
Helpers for Interpretation/Simulation-Parsing
"""
def apply_dict_replacement(inp_str, replacement_dict):
    """
    Applies a Dictionary of String-Replacements to a String.
    :type inp_str: str
    :type replacement_dict: dict
    :param inp_str: String, to apply the Replacements to
    :param replacement_dict: Dictionary of String-Replacements
    :return: String with applied replacements
    """
    out_str = inp_str

    for key in replacement_dict.keys():
        out_str = out_str.replace(key, replacement_dict[key])

    return out_str


def remove_leading_asterisk(line):
    """
    Removes a leading Asterisk from a Line.
    :type line: str
    :param line: Line to remove the leading Asterisk from
    :return: line without leading Asterisk
    """
    if line.startswith("* "):
        return line[2:]
    return line
