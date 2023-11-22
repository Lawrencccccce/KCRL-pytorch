import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed