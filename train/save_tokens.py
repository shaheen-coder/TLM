import numpy as np
from train.preprocess import PreTokens


dataset = PreTokens("datasets/", "tokenizer/tiny_lm_tokenizer.json")


encoder_list = []
decoder_in_list = []
decoder_out_list = []
# training data
for enc, dec in dataset.get_tokens(mode="t"):
    encoder_list.append(enc)
    decoder_in_list.append(dec[:-1])
    decoder_out_list.append(dec[1:])


encoder_arr = np.array(encoder_list, dtype=np.int32)
decoder_in_arr = np.array(decoder_in_list, dtype=np.int32)
decoder_out_arr = np.array(decoder_out_list, dtype=np.int32)

np.save("datasets/encoder.npy", encoder_arr)
np.save("datasets/decoder_in.npy", decoder_in_arr)
np.save("datasets/decoder_out.npy", decoder_out_arr)

# valdiation data
val_enc_lst = []
val_dec_in_lst = []
val_dec_out_lst = []

for enc, dec in dataset.get_tokens(mode="v"):
    val_enc_lst.append(enc)
    val_dec_in_lst.append(dec[:-1])
    val_dec_out_lst.append(dec[1:])

val_enc_arr = np.array(val_enc_lst, dtype=np.int32)
val_dec_in_arr = np.array(val_dec_in_lst, dtype=np.int32)
val_dec_out_arr = np.array(val_dec_out_lst, dtype=np.int32)

np.save("datasets/val_encoder.npy", val_enc_arr)
np.save("datasets/val_decoder_in.npy", val_dec_in_arr)
np.save("datasets/val_decoder_out.npy", val_dec_out_arr)
