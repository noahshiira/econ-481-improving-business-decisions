from my_sim.config import load_config
from my_sim.data_simulation import DataSimulator

def main(config_path="configs/default.yaml"):
    config = load_config(config_path)
    simulator = DataSimulator(config)
    simulator.simulate()

if __name__ == "__main__":
    main()
