import flwr as fl
from flwr.server.strategy import FedAvg

def server_fn(context):
    strategy = FedAvg(
    fraction_fit = 1.0,
    min_fit_clients = 2,
    min_available_clients = 2,
    )

    config = fl.server.ServerConfig(num_rounds=3)
    return fl.server.Server(config=config, strategy=strategy)

if __name__ == "__main__":
    fl.server.run_app(
        server_fn= server_fn,
        server_address = "0.0.0.0:8080",
    )
    
