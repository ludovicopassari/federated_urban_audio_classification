from flwr.client import NumPyClient, ClientApp
from models.models import get_parameters, set_parameters, test
from models.models import Net

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, config):
        self.partition_id = partition_id
        self._net = net
        self._trainloader = trainloader
        self._valloader = valloader
        self._device = config["device"] if "device" in config else 'cpu'
        self._epoch = config['epochs'] if "epoch" in config else 30
        

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self._net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self._net, parameters)
        self.train(self._net, self.trainloader, epochs=1)
        return get_parameters(self._net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self._net, parameters)
        loss, accuracy = test(self._net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
    def train(self, net, trainloader, epochs: int):
        pass

    def test(self, net, testloader):
        pass

