import argparse
import config


def get_trainer_opt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', choices=config.DATASETS.keys(), required=True)
    # parser.add_argument('--setting', '-s', choices=config.EXPERIMENT_SETTINGS, required=True)

    parser.add_argument('--seed', type=int, help='manually set the seeds for torch')

    parser.add_argument('--device', type=int, default=-1, help="the gpu device number to override")
    parser.add_argument('--batch', type=int, help="the batch size to override")

    parser.add_argument("--quiet", action="store_true", help="mute the log")
    parser.add_argument("--debug", action="store_true", help="print the debugging log")

    parser.add_argument('--memo', type=str, default="", help="used to remember some runtime configurations")
    parser.add_argument('--epoch', type=int, help="the training limit in number of epochs")
    parser.add_argument('--fine-tune', action="store_true")

    return parser


def get_test_opt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', choices=config.DATASETS.keys(), required=True)
    # parser.add_argument('--setting', '-s', choices=config.EXPERIMENT_SETTINGS, required=True)
    parser.add_argument('--device', type=int, help="the gpu device number to override")
    parser.add_argument('--batch', type=int, help="the batch size to override")

    parser.add_argument("--quiet", action="store_true", help="mute the log")
    parser.add_argument("--debug", action="store_true", help="print the debugging log")

    return parser
