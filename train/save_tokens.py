import numpy as np
from train.preprocess import PreTokens


dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")


encoder_list = []
decoder_in_list = []
decoder_out_list = []

for enc, dec in dataset.get_tokens():
    encoder_list.append(enc)
    decoder_in_list.append(dec[:-1])
    decoder_out_list.append(dec[1:])


encoder_arr = np.array([x for x in encoder_list], dtype=np.int32)
decoder_in_arr = np.array([x for x in decoder_in_list], dtype=np.int32)
decoder_out_arr = np.array([x for x in decoder_out_list], dtype=np.int32)

np.save("datasets/encoder.npy", encoder_arr)
np.save("datasets/decoder_in.npy", decoder_in_arr)
np.save("datasets/decoder_out.npy", decoder_out_arr)
