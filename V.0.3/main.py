import argparse
import sys
from monitor import monitor_system

def main(directory=''):
    parser = argparse.ArgumentParser(description="AI-based Antivirus")
    parser.add_argument('--mode', choices=['train', 'monitor'], default='monitor', help='Mode: train or monitor')
    parser.add_argument('--dir', default=None, help='Directory to monitor')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training mode is disabled in this version.")
        sys.exit(1)
    elif args.mode == 'monitor':
        monitor_dir = args.dir if args.dir else "C:/Users/Vovaaaan/Downloads/"
        monitor_system(monitor_dir)

if __name__ == "__main__":
    main()