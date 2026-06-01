import numpy as np
from train.preprocess import PreTokens


dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")

# -----------------------------------------------------------------------------------------
#                             Fine Tune Tokens 
# -----------------------------------------------------------------------------------------
input_list = []
target_list = []
# training data
for input in dataset.get_fine_tune_tokens(mode="t"):
    input_list.append(input[:-1])
    target_list.append(input[1:])


input_arr = np.array(input_list, dtype=np.int32)
target_arr = np.array(target_list, dtype=np.int32)

np.save("datasets/pretokens/ft_input.npy", input_arr)
np.save("datasets/pretokens/ft_target.npy", target_arr)

# valdiation data
val_input_lst = []
val_target_lst = []
for input in dataset.get_fine_tune_tokens(mode="v"):
    val_input_lst.append(input[:-1])
    val_target_lst.append(input[1:])

val_input_arr = np.array(val_input_lst, dtype=np.int32)
val_target_arr = np.array(val_target_lst, dtype=np.int32)

np.save("datasets/pretokens/ft_val_input.npy", val_input_arr)
np.save("datasets/pretokens/ft_val_target.npy", val_target_arr)

# -----------------------------------------------------------------------------------------
#                             Foundational tokenss
# -----------------------------------------------------------------------------------------
all_input_list = []
# training data
for input in dataset.get_fd_tokens():
    all_input_list.extend(input)
print(f"[LOG] all input len : {len(all_input_list)}")
print(f"[LOG] 0 : \n{all_input_list[0]}")
# sldiing window
input_list = []
target_list = []
n : int = len(all_input_list)
idx : int = 0
SEQ_LEN : int = 55
while idx + SEQ_LEN < n:
    input_list.append(all_input_list[idx : idx + SEQ_LEN ])
    target_list.append(all_input_list[idx + 1 : idx + SEQ_LEN + 1])
    idx += 1 

input_arr = np.array(input_list, dtype=np.int32)
target_arr = np.array(target_list, dtype=np.int32)

np.save("datasets/pretokens/fd_input.npy", input_arr)
np.save("datasets/pretokens/fd_target.npy", target_arr)
