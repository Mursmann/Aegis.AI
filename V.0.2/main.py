import argparse
from train import train_models
from monitor import monitor_system
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Super Smart Antivirus")
    parser.add_argument('--mode', choices=['train', 'monitor'], default='monitor',
                        help="Mode to run: 'train' for training models, 'monitor' for system monitoring")
    args = parser.parse_args()

    if args.mode == 'train':
        train_models()
    else:
        monitor_system()