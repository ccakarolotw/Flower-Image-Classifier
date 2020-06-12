import argparse
import sys

def parse_predict(argv):
    parser = argparse.ArgumentParser(
        description='Parse command lines')

    parser.add_argument(action="store",
                        dest="image", default=None)
    parser.add_argument(action="store",
                        dest="checkpoint",default=None)
    arg = parser.parse_args(argv)
    if arg.image == None or arg.checkpoint == None:
        print('enter image path followed by the checkpoint file name')
    return arg.checkpoint, arg.image 

def parse_train(argv):
    parser = argparse.ArgumentParser(
        description='Parse command lines')

    parser.add_argument(action="store",
                        dest="data_dir", default=None)
    parser.add_argument('-epoch', action="store", dest="epoch", type=int)
    arg = parser.parse_args(argv)
    return arg.data_dir, arg.epoch 