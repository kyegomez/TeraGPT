[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# TeraGPT
Zeta present TeraGPT – the simplest implementation for training large language models with tens or hundreds of billions of parameters. This work was inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). However, while nanoGPT is designed to train medium sized models up to around the 1B parameter range, TeraGPT leverages the over-powered Zeta framework to use a single simple model definition and training loop to scale to GPT-3 sized models run across zetascale clusters. 

As in nanoGPT, the main training logic is split between [`train.py`](./teragpt/train.py) and [`model.py`](./teragpt/model.py), with a total of 350 lines of simple, readable pytorch code combined. While nanoGPT can replicate GPT-2, gigaGPT is built to be able to replicate something of the scale of GPT-4 (albeit possibly with a dataset upgrade compared to the nanoGPT support). We have tested that models up to 175b parameters in size run functionally correctly at high throughput and have no reason to suspect that you can't scale significantly larger.

The combination of the scale of the hardware, the weight streaming execution mode, and the data parallel scale-out across machines is what provides the magic required for easy scale-out to larger models and larger clusters.

## Install
`pip3 install teragpt `


## Usage
```python
import torch
from teragpt.main import TeraGPT

model = TeraGPT(
    dim=4096,
    depth=6,
    heads=8,
    num_tokens=20000,
)

x = torch.randint(0, 20000, (1, 4096))

out = model(x)
print(out.shape)

```

### Tokenizer
```python
from teragpt import Tokenizer

tokenizer_name = "hf-internal-testing/llama-tokenizer"
tokenizer = Tokenizer(tokenizer_name=tokenizer_name)
encoded_text = tokenizer.encode("This is a sample text")
decoded_text = tokenizer.decode(encoded_text)
print("Encoded text:", encoded_text)
print("Decoded text:", decoded_text)

```


### Train
`trainer.py` sets up the environment for distributed training and then initializes a `Trainer` object to start the training process.

## Environment Variables

The script uses the following environment variables:

- `MASTER_ADDR`: The address of the master node. This is typically 'localhost'.
- `MASTER_PORT`: The port that the master node is listening on. This is typically '9994'.
- `RANK`: The rank of the current node in the distributed training setup. This is typically '0' for the master node.
- `WORLD_SIZE`: The total number of nodes participating in the distributed training. This is typically the number of GPUs available.

## How to Train the Model

1. Set the environment variables `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` appropriately for your distributed training setup.

2. Run the script with any additional arguments required by the `Trainer` object.

```bash
python train.py
```

Please note that the exact arguments required by the `Trainer` object will depend on your specific training setup and the model you are training.

## Note

The comment `[CRITICAL] Pay attention to this when scaling to multiple GPUs and clusters` indicates that the settings for `RANK` and `WORLD_SIZE` are particularly important when scaling the training process to multiple GPUs and clusters. Make sure to set these variables correctly to ensure efficient distributed training.


---

## Codebase comparison
The standard way to train a GPT-3 sized model is to use frameworks such as Nvidia Megatron. Megatron however is a large and complex framework that’s challenging to implement. This is what motivated the creation of nanoGPT – a light, readable, hackable framework. To quantify the complexity of these frameworks, we counted the lines of code in reach repo. Megatron has 20,507, lines of code while nanoGPT and Teragpt have 639 and 350 lines of code respectively. This supports our primary claim that TeraGPT trains GPT-3 sized models while retaining the simplicity of nanoGPT.

Megatron-LM

| Language                  | files |        blank |      comment |         code|
| ------------------------- | ----- | ------------ | ------------ | ----------- |
| Python                    |    99 |         4710 |         4407 |       18395 |
| C/C++ Header              |     4 |          146 |           90 |        1118 |
| C++                       |     4 |          137 |          117 |         649 |
| CUDA                      |     3 |           41 |           20 |         220 |
| HTML                      |     1 |           15 |            2 |         107 |
| Bourne Shell              |     1 |            1 |            0 |           9 |
| make                      |     1 |            2 |            0 |           7 |
| SUM:                      |   115 |         5052 |         4636 |       20507 |


nanoGPT

| Language                  | files |        blank |      comment |         code|
| ------------------------- | ----- | ------------ | ------------ | ----------- |
| Python                    |     5 |           90 |          187 |         639 |
| SUM:                      |     5 |           90 |          187 |         639 |

TeraGPT

| Language                  | files |        blank |      comment |         code|
| ------------------------- | ----- | ------------ | ------------ | ----------- |
| Python                    |     3 |          109 |            1 |         350 |
| SUM:                      |     6 |          109 |            1 |         350 |


# License
Apache