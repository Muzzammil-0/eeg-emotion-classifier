import flwr as fl
from flwr.server.strategy import FedAvg

def main():

    strategy = FedAvg(
        fraction_fit=0.1,  # Sample 10% of clients for training
        min_fit_clients=2,  # Minimum number of clients to be sampled for training
        min_available_clients=2,  # Minimum number of clients that need to be connected to the server before training can start
    )

    fl.server.start_server(
        server_address= "0.0.0.0:8080",
        config = fl.server.ServerConfig(num_rounds=3), #number of rounds of federated learning
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
    