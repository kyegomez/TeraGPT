[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# TeraGPT
Train a production grade GPT in less than 400 lines of code. Better than Karpathy's verison and GIGAGPT



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
```python
from teragpt import train

train()

```


# License
MIT



