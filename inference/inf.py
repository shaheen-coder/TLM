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

    def _load_tokenizer(self):
        tokenizer = Tokenizer.from_file("tokenizer/tiny_lm_tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
        )

    def _top_sampling(self, logits, topk=None, topp=None):
        logits = tf.squeeze(logits, axis=0)

        if topk is not None and topk > 0:
            values, _ = tf.math.top_k(logits, k=topk)
            min_top_k_val = values[-1]
            mask = logits < min_top_k_val
            logits = tf.where(mask, tf.fill(tf.shape(logits), float("-inf")), logits)

        if topp is not None and 0.0 < topp < 1.0:
            sorted_indices = tf.argsort(logits, direction="DESCENDING")
            sorted_logits = tf.gather(logits, sorted_indices)

            probs = tf.nn.softmax(sorted_logits)
            cumulative_probs = tf.cumsum(probs)

            mask = cumulative_probs > topp
            mask = tf.concat([[False], mask[:-1]], axis=0)

            sorted_logits = tf.where(
                mask,
                tf.fill(tf.shape(sorted_logits), float("-inf")),
                sorted_logits,
            )

            original_order = tf.argsort(sorted_indices)
            logits = tf.gather(sorted_logits, original_order)

        logits = tf.expand_dims(logits, axis=0)
        tok_id = int(tf.squeeze(tf.random.categorical(logits, 1)).numpy())
        return tok_id

    def generate(
        self,
        input_text: str,
        samp_mode="greedy",
        temp=1.0,
        topk=None,
        topp=None,
    ):
        # ---- Tokenize ----
        input_text = f"[PROMPT] {input_text} [AI]"
        input_ids = self.tokenizer(input_text)["input_ids"]

        # Start sequence with prompt
        generated = input_ids.copy()

        for _ in range(self.max_len):
            # ---- Model forward ----
            inputs = tf.constant([generated])
            logits = self.model(inputs, training=False)

            next_tok_logits = logits[:, -1, :] / temp

            # ---- Sampling ----
            if samp_mode == "greedy":
                tok_id = int(tf.argmax(next_tok_logits, axis=-1).numpy()[0])

            elif samp_mode == "top":
                tok_id = self._top_sampling(next_tok_logits, topk=topk, topp=topp)

            else:
                raise ValueError("Unknown sampling mode")

            # ---- Stop condition ----
            if tok_id == self.eos_id:
                break

            generated.append(tok_id)

        # Remove prompt tokens before decoding (optional)
        output_tokens = generated[len(input_ids) :]

        return self.tokenizer.decode(output_tokens)


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
