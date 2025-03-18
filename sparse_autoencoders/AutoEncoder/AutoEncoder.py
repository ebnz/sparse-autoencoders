import torch
import os

from sparse_autoencoders.AutoEncoder import AutoEncoderUtils


class AutoEncoderBase(torch.nn.Module):
    def __init__(self, n, m):
        """
        Base-Class of an Autoencoder.
        :type n: int
        :type m: int
        :param n: Activation-Vector Size
        :param m: Dictionary-Vector Size
        """
        super().__init__()

        self.n = n
        self.m = m

        self.training_runs = 0
        self.BATCH_SIZE = None
        self.L1_COEFFICIENT = None
        self.LEARNING_RATE = None
        self.LAYER_TYPE = None
        self.LAYER_INDEX = None
        self.SAVE_PATH = None
        self.IMAGE_PATH = None

        # Neuron Resampling
        self.neuron_resampling_enabled = False

        self.neuron_resampling_method = None
        self.neuron_resampling_interval = None
        self.summed_f_resampling = None

        # Checkpointing
        self.checkpointing_enabled = False

        self.checkpoint_interval = None
        self.summed_f_checkpoint = torch.zeros(self.m)
        self.count_f_checkpoint = torch.zeros(self.m)

        self.reconstruction_losses = []
        self.sparsity_losses = []

    def forward_encoder(self, x):
        """
        Forward-Pass of the Encoder of the Model.
        :type x: torch.Tensor
        :param x: Input of the Encoder
        :rtype: torch.Tensor
        """
        raise NotImplementedError("AutoEncoderBase is an interface!")

    def forward_decoder(self, x):
        """
        Forward-Pass of the Decoder of the Model.
        :type x: torch.Tensor
        :param x: Input of the Decoder
        :rtype: torch.Tensor
        """
        raise NotImplementedError("AutoEncoderBase is an interface!")

    def forward(self, x):
        """
        Forward-Pass of the Autoencoder.
        :type x: torch.Tensor
        :param x: Input of the Model
        :rtype: (torch.Tensor, torch.Tensor)
        :return: Output of the Model and Dictionary-Vector
        """
        if not self.training:
            f = self.forward_encoder(x)
            x_hat = self.forward_decoder(f)

            return x_hat, f

        self.training_runs += 1
        f = self.forward_encoder(x)
        x_hat = self.forward_decoder(f)

        # Checkpointing
        if self.checkpointing_enabled:
            self.checkpointing(x, x_hat, f)

        # Neuron Resampling
        if not self.neuron_resampling_enabled:
            pass
        elif self.neuron_resampling_method == "replacement":
            self.neuron_resampling_by_replacement(f)
        else:
            print("WARN: Neuron-Resampling Method not recognized")

        return x_hat, f

    """
    Loss Methods
    """
    def loss(self, x, x_hat, f, l1=1):
        """
        Loss-Function of the Model.
        :type x: torch.Tensor
        :type x_hat: torch.Tensor
        :type f: torch.Tensor
        :type l1: float
        :param x: Ground Truth Model Input
        :param x_hat: Resembled Model Output
        :param f: Dictionary-Vector
        :param l1: L1-Coefficient, Sparsity Penalty Coefficient
        :rtype: float | torch.Tensor
        :return: Value of Loss-Function
        """
        raise NotImplementedError("AutoEncoderBase is an interface!")


    """
    Neuron Resampling Methods
    """
    def enable_neuron_resampling(self, neuron_resampling_method=None, neuron_resampling_interval=25_000):
        """
        Enables Neuron Resampling for the Training of an Autoencoder.
        This resets all dead Neuron's Weights in an interval of Training Epochs.
        :type neuron_resampling_method: str
        :type neuron_resampling_interval: int
        :param neuron_resampling_method: Name of the Neuron Resampling Method
        :param neuron_resampling_interval: Interval of Neuron Resampling
        """
        self.neuron_resampling_enabled = True

        self.neuron_resampling_method = neuron_resampling_method
        self.neuron_resampling_interval = neuron_resampling_interval
        self.summed_f_resampling = torch.zeros(self.m)

    def neuron_resampling_by_replacement(self, f):
        """
        Implementation of the Neuron-Resampling Method 'replacement'.
        :type f: torch.Tensor
        :param f: Dictionary-Vector
        """
        self.summed_f_resampling += torch.sum(f.detach().cpu(), dim=0)

        if self.training_runs % self.neuron_resampling_interval == 0:
            device = f.device
            with torch.no_grad():
                # Reinit self.weight_encoder
                shape_new_weight_encoder = self.weight_encoder[::, torch.argwhere(self.summed_f_resampling == 0)].shape
                weight_encoder_new_part = torch.nn.init.kaiming_uniform_(torch.zeros(shape_new_weight_encoder))
                new_weight_encoder = self.weight_encoder.cpu().clone()
                new_weight_encoder[::, torch.argwhere(self.summed_f_resampling == 0)] = weight_encoder_new_part
                self.weight_encoder.data = new_weight_encoder.to(device)
                # Reinit self.weight_decoder
                shape_new_weight_decoder = self.weight_decoder[torch.argwhere(self.summed_f_resampling == 0)].shape
                weight_decoder_new_part = torch.nn.init.kaiming_uniform_(torch.zeros(shape_new_weight_decoder))
                new_weight_decoder = self.weight_decoder.cpu().clone()
                new_weight_decoder[torch.argwhere(self.summed_f_resampling == 0)] = weight_decoder_new_part
                self.weight_decoder.data = new_weight_decoder.to(device)

                print(f"Resampled {torch.sum(self.summed_f_resampling == 0)} Neurons!")

            # Reset self.summed_f_resampling
            self.summed_f_resampling = torch.zeros(self.m)


    """
    Checkpointing Methods
    """
    def enable_checkpointing(self, learning_rate, l1_coefficient, batch_size, layer_type, layer_index, save_path, image_path, checkpoint_interval=12_500):
        """
        Enables Checkpointing of the Model's Training Process.
        :type learning_rate: float
        :type l1_coefficient: float
        :type batch_size: int
        :type layer_type: str
        :type layer_index: int
        :type save_path: str
        :type image_path: str
        :type checkpoint_interval: int
        :param learning_rate: Learning Rate
        :param l1_coefficient: L1-Coefficient, Sparsity Penalty Coefficient
        :param batch_size: Batch Size
        :param layer_type: Name of Hookpoint
        :param layer_index: Layer-ID
        :param save_path: Path to save the Model to
        :param image_path: Path to save the Histogram to
        :param checkpoint_interval: Interval, in which the Checkpoints should be generated
        """
        self.checkpointing_enabled = True

        self.LEARNING_RATE = learning_rate
        self.L1_COEFFICIENT = l1_coefficient
        self.BATCH_SIZE = batch_size
        self.LAYER_TYPE = layer_type
        self.LAYER_INDEX = layer_index
        self.SAVE_PATH = save_path
        self.IMAGE_PATH = image_path

        self.checkpoint_interval = checkpoint_interval
        self.summed_f_checkpoint = torch.zeros(self.m)
        self.count_f_checkpoint = torch.zeros(self.m)

    def calculate_activation_counts(self, log10=True, num_samples=None):
        """
        Calculates the Activation Counts (Frequency of Dictionary-Feature Activations).
        :type log10: bool
        :type num_samples: int
        :param log10: Whether to return the Activations in Log10
        :param num_samples: Number of Samples, used for Activation Count Calculation
        :return: Activations Counts and Number of dead Neurons
        """
        if num_samples is None:
            activation_counts = self.count_f_checkpoint / (self.checkpoint_interval * self.BATCH_SIZE)
        else:
            activation_counts = self.count_f_checkpoint / num_samples
        activation_counts_log10 = activation_counts.log10()
        no_dead_neurons = torch.sum(self.summed_f_checkpoint == 0)

        if log10:
            return activation_counts_log10, no_dead_neurons
        return activation_counts, no_dead_neurons

    def checkpointing(self, x, x_hat, f):
        """
        Checkpointing Precedure. Saves a Checkpoint if Interval-Condition is met.
        :type x: torch.Tensor
        :type x_hat: torch.Tensor
        :type f: torch.Tensor
        :param x: Ground Truth Autoencoder Input
        :param x_hat: Reconstructed Autoencoder Output
        :param f: Dictionary-Vector
        """
        # Shape [Tokens, Activations]
        if len(f.shape) == 2:
            self.summed_f_checkpoint += torch.sum(f.detach().cpu(), dim=0)
            self.count_f_checkpoint += torch.sum(f.detach().cpu() > 0, dim=0)
        # Shape [Batch, Tokens, Activations]
        elif len(f.shape) == 3:
            self.summed_f_checkpoint += torch.sum(f.detach().cpu(), dim=(0, 1))
            self.count_f_checkpoint += torch.sum(f.detach().cpu() > 0, dim=(0, 1))

        self.reconstruction_losses.append(
            self.reconstruction_loss(x, x_hat).detach().cpu().tolist()
        )

        self.sparsity_losses.append(
            self.sparsity_loss(f, l1=self.L1_COEFFICIENT).detach().cpu().tolist()
        )

        if self.training_runs % self.checkpoint_interval == 0:
            activation_counts_log10, no_dead_neurons = self.calculate_activation_counts(log10=True)

            self.summed_f_checkpoint = torch.zeros(self.m)
            self.count_f_checkpoint = torch.zeros(self.m)

            # Save Checkpoint
            fig = AutoEncoderUtils.generate_loss_curve(self.reconstruction_losses, self.sparsity_losses)
            AutoEncoderUtils.save_plotly_graph(fig, os.path.join(self.IMAGE_PATH, "loss_curve.html"))

            fig = AutoEncoderUtils.generate_histogram(activation_counts_log10, no_dead_neurons)
            AutoEncoderUtils.save_plotly_graph(fig, os.path.join(self.IMAGE_PATH, f"{self.training_runs}.html"))
            model_conf = AutoEncoderUtils.generate_model_conf(self, self.n, self.m, self.LEARNING_RATE, self.L1_COEFFICIENT,
                                                              self.BATCH_SIZE, self.LAYER_TYPE, self.LAYER_INDEX)
            AutoEncoderUtils.save_autoencoder_checkpoint(
                model_conf,
                os.path.join(self.SAVE_PATH, f"{self.training_runs}.pt")
            )

class AutoEncoderAnthropic(AutoEncoderBase):
    def __init__(self, n, m):
        """
        Implementation of the Anthropic Autoencoder.
        Link: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
        """
        super().__init__(n, m)

        self.bias_encoder = torch.nn.Parameter(torch.zeros(m))
        self.bias_decoder = torch.nn.Parameter(torch.zeros(n))

        # Init Encoder-Weights (Kaiming)
        self.weight_encoder = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(self.n, self.m)))
        # self.weight_encoder = torch.nn.Parameter((self.weight_encoder.T / torch.norm(self.weight_encoder, dim=1, p=2)).T)

        # Init Decoder-Weights and set them to Unit Norm
        self.weight_decoder = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.zeros(self.m, self.n)))
        self.decoder_unit_norm()

        self.relu = torch.nn.ReLU()

    @torch.no_grad()
    def decoder_unit_norm(self):
        self.weight_decoder.data = (self.weight_decoder.T / torch.norm(self.weight_decoder, dim=1, p=2)).T


    """
    Forward Methods
    """
    def forward_encoder(self, x):
        self.decoder_unit_norm()
        x_bar = x - self.bias_decoder
        f = self.relu(x_bar @ self.weight_encoder + self.bias_encoder)
        return f

    def forward_decoder(self, f):
        x_hat = f @ self.weight_decoder + self.bias_decoder
        return x_hat

    """
    Loss Methods
    """
    def loss(self, x, x_hat, f, l1=1):
        return (1 / x.shape[0]) * (self.reconstruction_loss(x, x_hat) + self.sparsity_loss(f, l1=l1))

    def reconstruction_loss(self, x, x_hat):
        return torch.sum((x - x_hat) ** 2)

    def sparsity_loss(self, f, l1=1):
        return torch.sum(l1 * torch.abs(f))


class AutoEncoderAnthropicImproved(AutoEncoderBase):
    def __init__(self, n, m, l2_decoder_init=0.1):
        super().__init__(n, m)

        # Init both Biases with zeros
        self.bias_encoder = torch.nn.Parameter(torch.zeros(m))
        self.bias_decoder = torch.nn.Parameter(torch.zeros(n))

        # Init Decoder Weights with Kaiming-Uniform and setting all rows to L2-Norm of l2_decoder_init
        self.weight_decoder = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(self.m, self.n))
        self.weight_decoder = l2_decoder_init * (self.weight_decoder.T / torch.norm(self.weight_decoder, dim=1, p=2)).T

        # Init Encoder Weights to Transpose of Decoder Weights
        self.weight_encoder = torch.nn.Parameter(self.weight_decoder.T)

        self.relu = torch.nn.ReLU()

    """
    Forward Methods
    """

    def forward_encoder(self, x):
        f = self.relu(x @ self.weight_encoder + self.bias_encoder)
        return f

    def forward_decoder(self, f):
        x_hat = f @ self.weight_decoder + self.bias_decoder
        return x_hat

    """
    Loss Methods
    """

    def loss(self, x, x_hat, f, l1=1):
        return (1 / x.shape[0]) * (self.reconstruction_loss(x, x_hat) + self.sparsity_loss(f, l1=l1))

    def reconstruction_loss(self, x, x_hat):
        return torch.sum((x - x_hat) ** 2)

    def sparsity_loss(self, f, l1=1):
        l1_loss_batches = f @ torch.norm(self.weight_decoder, dim=1, p=2)
        return l1 * torch.sum(l1_loss_batches)

    """
    Neuron Resampling Methods
    """
    def neuron_resampling_by_replacement(self, f):
        self.training_runs += 1
        if self.training_runs % self.neuron_resampling_interval == 0:
            with torch.no_grad():
                # Reinit self.bias_encoder
                self.bias_encoder[torch.argwhere(self.summed_f_resampling == 0)] = torch.zeros_like(
                    self.bias_encoder[torch.argwhere(self.summed_f_resampling == 0)])
                # Reinit self.weight_encoder
                self.weight_encoder[::, torch.argwhere(self.summed_f_resampling == 0)] = torch.nn.init.kaiming_uniform_(
                    self.weight_encoder[::, torch.argwhere(self.summed_f_resampling == 0)])
                # Reinit self.weight_decoder
                self.weight_decoder[torch.argwhere(self.summed_f_resampling == 0)] = torch.nn.init.kaiming_uniform_(
                    self.weight_decoder[torch.argwhere(self.summed_f_resampling == 0)])

                print(f"Resampled {torch.sum(self.summed_f_resampling == 0)} Neurons!")

            # Reset self.summed_f_resampling
            self.summed_f_resampling = torch.zeros(self.m)


"""
Model-Loading
"""
def load_model_from_config(config_dict):
    """
    Loads a Model from its Model-Configuration Dictionary.
    :type config_dict: dict
    :param config_dict: Model-Configuration Dictionary
    :rtype: AutoEncoderBase
    """
    if config_dict["MODEL_TYPE"] == "AutoEncoderAnthropic":
        autoencoder = AutoEncoderAnthropic(config_dict["ACT_VEC_SIZE"], config_dict["DICT_VEC_SIZE"])
    elif config_dict["MODEL_TYPE"] == "AutoEncoderAnthropicImproved":
        autoencoder = AutoEncoderAnthropic(config_dict["ACT_VEC_SIZE"], config_dict["DICT_VEC_SIZE"])
    autoencoder.load_state_dict(config_dict["STATE_DICT"])
    return autoencoder
