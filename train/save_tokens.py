import numpy as np
from train.preprocess import PreTokens


dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")


input_list = []
target_list = []
# training data
for input in dataset.get_tokens(mode="t"):
    input_list.append(input[:-1])
    target_list.append(input[1:])


input_arr = np.array(input_list, dtype=np.int32)
target_arr = np.array(target_list, dtype=np.int32)

np.save("datasets/input.npy", input_arr)
np.save("datasets/target.npy", target_arr)

# valdiation data
val_input_lst = []
val_target_lst = []
for input in dataset.get_tokens(mode="v"):
    val_input_lst.append(input[:-1])
    val_target_lst.append(input[1:])

val_input_arr = np.array(val_input_lst, dtype=np.int32)
val_target_arr = np.array(val_target_lst, dtype=np.int32)

np.save("datasets/val_input.npy", val_input_arr)
np.save("datasets/val_target.npy", val_target_arr)
