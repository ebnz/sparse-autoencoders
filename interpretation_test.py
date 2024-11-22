import os

from sparse_autoencoders.AutoInterpretation import *

interpretation_config = InterpretationConfig(
    "/nfs/data/students/ebenz_bsc2024/tokenized_dataset",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "/nfs/data/students/ebenz_bsc2024/autoenc_2/autoenc_lr2e-4_0.5_32_nr/50000.pt"
)

interpreter = AutoInterpreter(interpretation_config)

# interpreter.load_dataset()
# interpreter.load_target_model("cuda:2")
# interpreter.load_autoencoder("cuda:3")
# interpreter.obtain_interpretation_samples(2000)
# interpreter.save_interpretation_samples("/nfs/data/students/ebenz_bsc2024/interp_samples_2000.pkl")




local_rank = int(os.getenv("LOCAL_RANK", "0"))

interpreter.load_dataset()
interpreter.load_interpretation_model_deepspeed(4)

interpreter.load_interpretation_samples("/nfs/data/students/ebenz_bsc2024/interp_samples_2000.pkl")

print(interpreter.mean_feature_activations.shape)
print(interpreter.mean_feature_activations.dtype)
print(interpreter.activation_counts_log10.shape)
print(interpreter.interpretable_neuron_indices)
print(interpreter.activation_counts_log10[interpreter.interpretable_neuron_indices])





# if local_rank == 0:
#     file = open("/nfs/home/ebenz_bsc2024/ints.txt", "w")
#
# for idx, feature_index in enumerate(interpreter.interpretable_neuron_indices):
#     user_prompt_interpretation = interpreter.generate_interpretation_prompt(feature_index, 3)
#     interpretation = interpreter.get_explanation(user_prompt_interpretation, raw=True)
#     if local_rank == 0:
#         #print(interpretation)
#         #print("---------------------------------------------------------")
#         file.write(interpretation)
#         file.write("---------------------------------------------------------")
#
#     if idx >= 100:
#         if local_rank == 0:
#             file.close()
#         break




if local_rank == 0:
    file = open("/nfs/home/ebenz_bsc2024/sims.txt", "w")

for idx, feature_index in enumerate(interpreter.interpretable_neuron_indices):
    user_prompt_interpretation = interpreter.generate_interpretation_prompt(feature_index, 3)
    interpretation = interpreter.get_explanation(user_prompt_interpretation)
    user_prompt_simulation = interpreter.generate_simulation_prompt(feature_index, 3, interpretation)
    simulation = interpreter.get_simulation(user_prompt_simulation, return_raw_simulation=True)
    if local_rank == 0:
        #print(interpretation)
        #print("---------------------------------------------------------")
        file.write(simulation)
        file.write("---------------------------------------------------------")

    if idx >= 100:
        if local_rank == 0:
            file.close()
        break


# interpreter.load_dataset()
# interpreter.load_interpretation_samples("/nfs/data/students/ebenz_bsc2024/interp_samples_2000.pkl")
#
# for idx, feature_index in enumerate(interpreter.interpretable_neuron_indices):
#     user_prompt = interpreter.generate_interpretation_prompt(feature_index, 5)
#     print("----------------------------------------------------------")
#
#     if idx > 10:
#         break

# user_prompt = interpreter.generate_interpretation_prompt(1535, 5)
# print(user_prompt)
# print("---------------------------------------------------------")
