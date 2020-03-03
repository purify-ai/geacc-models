import geacc.training.train_inceptionv3 as model

import argparse


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", metavar="PATH", required=True,
                        help="Path to the dataset containing training and validation data")
    parser.add_argument("-m", "--models", metavar="PATH", required=True,
                        help="Path to store trained intermediate and final models")
    parser.add_argument("-t", "--tensorboard", metavar="PATH", required=False,
                        help="Path to store TensorBoard logs")
    parser.add_argument("-ds", "--distribution_strategy", metavar="STRATEGY", required=False, default="off",
                        help="The Distribution Strategy to use for training. Case insensitive. "
                             "Accepted values are 'off', 'tpu', 'one_device', 'mirrored', 'parameter_server', 'collective'."
                             "'off' means not to use Distribution Strategy; 'tpu' to use TPU.")
    parser.add_argument("--tpu-address", metavar="TPU", required=False,
                        help="The Cloud TPU to use for training. This should be either the name "
                             "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url. Passing `local` will use the "
                             "CPU of the local instance instead. (Good for debugging.)")
    
    args = parser.parse_args()
    return args


def main():
    args = input_arguments()
    model.train(dataset_path=args.dataset, model_path=args.models, tb_path=args.tensorboard,
                distribution_strategy=args.distribution_strategy, tpu_address=args.tpu_address)


if __name__ == "__main__":
    # logging.set_verbosity(logging.INFO)
    main()
