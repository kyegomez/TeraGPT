from teragpt.main import TeraGPT
from zeta.training import Trainer

def train(
    model = TeraGPT,
    dataset = None,
    optimizer = None,
    scheduler = None,
    device = None,
    epochs = 10,
    
)