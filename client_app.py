
import numpy as np

import json 
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from models.models import Net
from client.client import FlowerClient


with open('client/client_config.json') as client_config_file:
    client_config = json.load(client_config_file)

#Leggi questo dal file di configurazione del client
DEVICE = None


def load_datasets():
    # questa funzione mi deve ritornare il train-loader e il validation loader 
    pass

def client_fn(context: Context):
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader, config= client_config).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)