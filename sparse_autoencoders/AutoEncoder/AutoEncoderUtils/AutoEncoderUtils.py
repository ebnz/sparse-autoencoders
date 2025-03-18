import plotly.express as px
import plotly
import pickle

"""
Helpers
"""
def generate_model_conf(model, ACT_VEC_SIZE, DICT_VEC_SIZE, LEARNING_RATE,
                        L1_COEFFICIENT, BATCH_SIZE_AE, LAYER_TYPE, LAYER_INDEX):
    """
    Generates the Model-Cofiguration Dictionary with the specified Parameters.
    :type model: AutoEncoderBase
    :type ACT_VEC_SIZE: int
    :type DICT_VEC_SIZE: int
    :type LEARNING_RATE: float
    :type L1_COEFFICIENT: float
    :type BATCH_SIZE_AE: int
    :type LAYER_TYPE: str
    :type LAYER_INDEX: int
    :param model: Model
    :param ACT_VEC_SIZE: Size of the Activation Vectors
    :param DICT_VEC_SIZE: Size of the Dictionary Vectors
    :param LEARNING_RATE: Learning Rate, Training of Autoencoder
    :param L1_COEFFICIENT: L1-Coefficient, Sparsity Penalty Coefficient
    :param BATCH_SIZE_AE: Batch Size, Training of Autoencoder
    :param LAYER_TYPE: Hookpoint-Name, Training of Autoencoder
    :param LAYER_INDEX: Layer-ID, Training of Autoencoder
    :return: Model-Configuration
    """
    return {
        "MODEL_TYPE": model.__class__.__name__,
        "STATE_DICT": model.state_dict(),
        "ACT_VEC_SIZE": ACT_VEC_SIZE,
        "DICT_VEC_SIZE": DICT_VEC_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "L1_COEFFICIENT": L1_COEFFICIENT,
        "BATCH_SIZE_TRAINING": BATCH_SIZE_AE,
        "LAYER_TYPE": LAYER_TYPE,
        "LAYER_INDEX": LAYER_INDEX,
        "INTERPRETATIONS": None,
        "MINS": None,
        "MAXS": None
    }


"""
PLOTS
"""
def generate_loss_curve(reconstruction_losses, sparsity_losses):
    """
    Returns a Plotly Figure, containing the Loss Curves.
    :type reconstruction_losses: list[float]
    :type sparsity_losses: list[float]
    :param reconstruction_losses: List of Reconstruction Losses
    :param sparsity_losses: List of Sparsity Losses
    :return: Plotly Figure of Loss Curves
    """
    recons_line = plotly.graph_objs.Scatter(y=reconstruction_losses, name="Reconstruction")
    sparsity_line = plotly.graph_objs.Scatter(y=sparsity_losses, name="Sparsity")
    fig = plotly.graph_objs.Figure([recons_line, sparsity_line], layout={"title": "Loss Curves"})
    fig.update_layout(yaxis_type="log")
    return fig

def generate_histogram(activation_counts_log10, no_dead_neurons):
    """
    Returns a Plotly Figure, containing a Histogram of the Feature Frequencies.
    :type activation_counts_log10: list[float]
    :type no_dead_neurons: list[int]
    :param activation_counts_log10: List of Log10 of Activation-Frequency
    :param no_dead_neurons: Number of Dead Neurons
    :return: Plotly Figure of Histogram of Feature Frequencies
    """
    fig = px.histogram(
        activation_counts_log10,
        histnorm="percent",
        title=f"{no_dead_neurons} dead Neurons"
    )
    return fig

def save_plotly_graph(fig, save_path):
    """
    Saves a Plotly Figure to disk.
    :type fig: plotly.Figure
    :type save_path: str
    :param fig: Plotly Figure
    :param save_path: Path to save the Figure to
    """
    plotly.offline.plot(fig, filename=save_path)

"""
CHECKPOINTING
"""
def save_autoencoder_checkpoint(model_conf, save_path):
    """
    Saves a Checkpoint of an Autoencoder, generated during Training.
    :type model_conf: dict
    :type save_path: str
    :param model_conf: Model-Configuration Dictionary
    :param save_path: Path, the Checkpoint is saved to
    """
    # Move State-Dict to CPU, for successful loading on a CPU-only machine
    state_dict_cpu = {k: v.cpu() for k, v in model_conf["STATE_DICT"].items()}
    model_conf["STATE_DICT"] = state_dict_cpu

    # Save to disk
    with open(save_path, "wb") as f:
        pickle.dump(model_conf, f)
