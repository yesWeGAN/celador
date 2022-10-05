#!/usr/bin/env python
import argparse
import sys

# see also code.data_collection.bing_collect for **kwargs implementation


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-l', '--limit', type=int, default=100, help='limit of items to download')
    parser.add_argument('-q', '--query_list', type=str, nargs='+', default=[], help='query strings')
    parser.add_argument('--force_replace', action='store_true', help='overwrite previous (default: False)')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    return args


def the_actual_task(parameters):
    output = parameters
    return output


def main():
    print("this is the main function")
    inputs = parse_args()
    endresult = the_actual_task(inputs)


if __name__ == '__main__':
    main()
