import argparse
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from model.transformer import TinyLM


class TinyLMInference:
    def __init__(self, model, start_token_id, end_token_id, max_len=50):
        self.model = model
        self.bos_id = start_token_id
        self.eos_id = end_token_id
        self.max_len = max_len
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = Tokenizer.from_file("tokenizer/tiny_lm_tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
        )

    def _greedy_sampling(self, logits):
        tok_id = int(tf.argmax(logits, axis=-1).numpy()[0])
        return tok_id

    def _top_sampling(self, logits, topk=None, topp=None):
        logits = tf.squeeze(logits, axis=0)  # [vocab_size]

        # --- Top-K filtering ---
        if topk is not None and topk > 0:
            # Find the kth largest value as a threshold
            values, _ = tf.math.top_k(logits, k=topk)
            min_top_k_val = values[-1]  # smallest value among top-k
            # Zero out (mask) anything below the top-k threshold
            mask = logits < min_top_k_val
            logits = tf.where(mask, tf.fill(tf.shape(logits), float("-inf")), logits)

        # --- Top-P (nucleus) filtering ---
        if topp is not None and 0.0 < topp < 1.0:
            sorted_indices = tf.argsort(logits, direction="DESCENDING")
            sorted_logits = tf.gather(logits, sorted_indices)

            sorted_probs = tf.nn.softmax(sorted_logits)
            cumulative_probs = tf.cumsum(sorted_probs, exclusive=False)

            sorted_mask = cumulative_probs - sorted_probs > topp
            sorted_logits = tf.where(
                sorted_mask,
                tf.fill(tf.shape(sorted_logits), float("-inf")),
                sorted_logits,
            )

            original_order = tf.argsort(sorted_indices)
            logits = tf.gather(sorted_logits, original_order)

        logits = tf.expand_dims(logits, axis=0)
        tok_id = int(tf.squeeze(tf.random.categorical(logits, num_samples=1)).numpy())
        return tok_id

    def generate(
        self,
        input_text: str,
        samp_mode: str = "greedy",
        temp: float = 1.0,
        topk: int = None,
        topp: float = None,
    ) -> str:

        input_text = f"[PROMPT] {input_text}"
        input_ids = self.tokenizer(input_text)["input_ids"]
        encoder_input = tf.constant([input_ids])
        encoder_outputs, states = self.model.encoder(encoder_input, training=False)

        decoder_input = tf.constant([[self.bos_id]])

        generated = []

        for _ in range(self.max_len):
            logits, states = self.model.decoder(
                encoder_outputs, decoder_input, states, training=False
            )

            next_tok_log = logits[:, -1, :]

            if samp_mode == "greedy":
                next_tok = tf.argmax(next_tok_log, axis=-1)
                tok_id = int(next_tok.numpy()[0])

                if tok_id == self.eos_id:
                    break

                generated.append(tok_id)
                decoder_input = tf.expand_dims(
                    tf.cast(next_tok, dtype=tf.int32), axis=1
                )  # [1, 1]

            elif samp_mode == "top":
                scaled_logits = next_tok_log / temp

                tok_id = self._top_sampling(scaled_logits, topk=topk, topp=topp)

                if tok_id == self.eos_id:
                    break

                generated.append(tok_id)
                decoder_input = tf.constant([[tok_id]])

        text = self.tokenizer.decode(generated)
        return text


if __name__ == "__main__":
    model = load_model("tlm.keras", compile=False)
    raw_tok = Tokenizer.from_file("tokenizer/tiny_lm_tokenizer.json")
    bos_id = raw_tok.token_to_id("[AI]")
    eos_id = raw_tok.token_to_id("[END]")

    inferencer = TinyLMInference(model, bos_id, eos_id, max_len=50)

    samples = [
        "vanakkam shaAI, epdi irukka?",
        "hello shaAI",
        "hi shaAI, good morning",
        "hey shaAI",
        "shaAI, unga peru epdi varum?nee yaaru shaAI?",
    ]
    print("--------------------------- greedy -------------------------- ")
    for sample in samples:
        response = inferencer.generate(sample, "greedy")
        print(f"Prompt  : {sample}")
        print(f"Response: {response}")
        print()

    print("--------------------------- Top    -------------------------- ")
    for sample in samples:
        response = inferencer.generate(sample, "top", 0.7, 20, 0.9)
        print(f"Prompt  : {sample}")
        print(f"Response: {response}")
        print()
