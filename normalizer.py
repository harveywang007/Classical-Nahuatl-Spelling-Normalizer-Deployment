import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformer import Seq2SeqTransformer


SAVED_MODEL_PATH = "./models/"
TOKENIZER_PATH = "./tokenizers/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL = ["<START>", "<PAD>", "<END>", "<MASK>"]

NEG_INFTY = -1e9

EMB_SIZE = 256
MAX_LEN = 4096
NUM_HEADS = 4
FFN_HID_DIM = 1024
BATCH_SIZE = 20
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.3


def load_trained_transformer(file_name, source_vocab_size=271, target_vocab_size=271):
    """Loads a trained transformer using the Seq2SeqTransformer class and the saved weights."""
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, MAX_LEN,
                                     NUM_HEADS, source_vocab_size, target_vocab_size, FFN_HID_DIM,
                                     DROPOUT)

    transformer.load_state_dict(torch.load(SAVED_MODEL_PATH + file_name + ".pt",
                                           map_location=torch.device(DEVICE)))

    transformer.to(DEVICE)

    return transformer


def load_tokenizer(dir_name):
    """Loads the wrapped and saved tokenizer for use."""
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH + dir_name)


def create_tokens(sentence, tokenizer):
    sentence = tokenizer(sentence)["input_ids"]

    def add_tokens(sentence, tokenizer):
        sentence = torch.cat((torch.tensor(tokenizer(SPECIAL[0])["input_ids"]),
                              torch.tensor(sentence),
                              torch.tensor(tokenizer(SPECIAL[2])["input_ids"])))
        return sentence
    return add_tokens(sentence, tokenizer)


def square_masks(size):
    mask = torch.ones((size, size), device=DEVICE)
    mask = torch.triu(mask).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(NEG_INFTY)).masked_fill(mask == 1, float(0.0))
    return mask


def greedy_decode(model, source, source_mask,
                  target_tokenizer, max_len):
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = model.encode(source, source_mask)

    normalization = torch.ones(1, 1)
    normalization = normalization.fill_(torch.tensor(target_tokenizer(SPECIAL[0])["input_ids"][0])).type(torch.long).to(DEVICE)

    for _ in range(max_len-1):
        memory = memory.to(DEVICE)

        output_mask = (square_masks(normalization.size(0)).type(torch.bool)).to(DEVICE)

        output = model.decode(normalization, memory, output_mask)
        output = output.transpose(0, 1)

        prob = model.linear(output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        normalization = torch.cat([normalization, torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)

        if next_word == target_tokenizer(SPECIAL[2])["input_ids"][0]:
            break

    return normalization


def normalize(source_sentence, model, source_tokenizer, target_tokenizer):
    model.eval()

    source = create_tokens(source_sentence, source_tokenizer)
    source = source.view(-1, 1)

    num_tokens = source.shape[0]
    source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    output_tokens = greedy_decode(model, source, source_mask,
                                  target_tokenizer, max_len=num_tokens + 5).flatten()
    output_tokens.tolist()

    return "".join(target_tokenizer.decode(output_tokens)
                   .replace(SPECIAL[0], "")
                   .replace(SPECIAL[2], "")
                   .strip())
