"""Configuration loader test."""

from utils.config import Config


def main():

    config = Config("configs/default.yaml")

    print("=" * 60)

    print("PROJECT ATLAS CONFIGURATION")

    print("=" * 60)

    print("Dataset:", config.get("dataset", "name"))

    print("Clients:", config.get("partition", "num_clients"))

    print("Rounds:", config.get("training", "num_rounds"))

    print("Learning rate:", config.get("training", "learning_rate"))

    print("Privacy Enabled:", config.get("privacy", "enabled"))

    print("Epsilon:", config.get("privacy", "epsilon"))

    print("=" * 60)

    print("Configuration loader passed.")


if __name__ == "__main__":

    main()