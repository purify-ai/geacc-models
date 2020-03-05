import geacc.training.train_inceptionv3 as model

import argparse
import tensorflow as tf


def get_gpu_num():
    gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     print("Name:", gpu.name, " Type:", gpu.device_type)
    return len(gpus)


def input_arguments():
    gpus_detected = get_gpu_num()
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
    parser.add_argument("--gpu-num", metavar="GPU", required=False, default=gpus_detected,
                        help="How many GPUs to use at each worker with the DistributionStrategies API. "
                             f"The default is auto-detect on local host (detected: {gpus_detected}).")
    parser.add_argument("-b", "--batch-size", required=False, default=128,
                        help="Batch size for training and evaluation. When using multiple gpus, this is the global batch size for "
                              "all devices. For example, if the batch size is 32 and there are 4 GPUs, each GPU will get 8 examples on "
                              "each step. For TPU, batch size of 1024 is recommended.")

    args = parser.parse_args()
    return args


def main():
    args = input_arguments()
    print('Tensorflow version: ', tf.__version__)

    model.train(dataset_path=args.dataset, model_path=args.models, tb_path=args.tensorboard, batch_size=args.batch_size,
                distribution_strategy=args.distribution_strategy, gpu_num=args.gpu_num, tpu_address=args.tpu_address)


if __name__ == "__main__":
    # logging.set_verbosity(logging.INFO)
    main()
