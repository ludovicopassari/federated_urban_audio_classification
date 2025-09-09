from flwr.simulation import run_simulation

# importa server e client già definiti
from server_app import server
from client_app import client


# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {
    "client_resources": {"num_cpus": 1, "num_gpus": 0},
    "num_clients": 10, 
}


if __name__ == "__main__":
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=5,
        backend_config=backend_config,
    )