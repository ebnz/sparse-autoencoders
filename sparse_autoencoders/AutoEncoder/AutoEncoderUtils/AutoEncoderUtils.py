import plotly.express as px
import plotly
import pickle

"""
Helpers
"""
def generate_model_conf(model, ACT_VEC_SIZE, DICT_VEC_SIZE, LEARNING_RATE,
                        L1_COEFFICIENT, BATCH_SIZE_AE, LAYER_TYPE, LAYER_INDEX):
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
    recons_line = plotly.graph_objs.Scatter(y=reconstruction_losses, name="Reconstruction")
    sparsity_line = plotly.graph_objs.Scatter(y=sparsity_losses, name="Sparsity")
    fig = plotly.graph_objs.Figure([recons_line, sparsity_line], layout={"title": "Loss Curves"})
    fig.update_layout(yaxis_type="log")
    return fig

def generate_histogram(activation_counts_log10, no_dead_neurons):
    fig = px.histogram(
        activation_counts_log10,
        histnorm="percent",
        title=f"{no_dead_neurons} dead Neurons"
    )
    return fig

def save_plotly_graph(fig, save_path):
    plotly.offline.plot(fig, filename=save_path)

"""
CHECKPOINTING
"""
def save_autoencoder_checkpoint(model_conf, save_path):
    # Move State-Dict to CPU, for successful loading on a CPU-only machine
    state_dict_cpu = {k: v.cpu() for k, v in model_conf["STATE_DICT"].items()}
    model_conf["STATE_DICT"] = state_dict_cpu

    # Save to disk
    with open(save_path, "wb") as f:
        pickle.dump(model_conf, f)


