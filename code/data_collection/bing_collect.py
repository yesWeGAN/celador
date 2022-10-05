#!/usr/bin/env python
import argparse
import os
import sys
from tqdm import tqdm

from bing_image_downloader import downloader


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def scrape(query_string, **kwargs) -> None:
    """execute the bing download with given parameters
    filter : (optional, default is "") filter, choose from [line, photo, clipart, gif, transparent]"""

    print(f"Downloading query {query_string}")
    with HiddenPrints():
        downloader.download(query_string, **kwargs)


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-q', '--query_list', type=str, nargs='+', default=[], help='query strings')
    parser.add_argument('-l', '--limit', type=int, default=10, help='limit of items to download')
    parser.add_argument('-d', '--output_dir', type=str, default="", help='output directory')

    parser.add_argument('-t', '--timeout', type=int, default=60, help='timeout for connection')

    parser.add_argument('--adult_filter_off', action='store_false', default=True,
                        help='adult filter off (default: True)')
    parser.add_argument('--force_replace', action='store_true', default=False, help='overwrite (default: False)')
    # parser.add_argument('-f', '--filter', type=str, default='photo', help='type of data to dl')
    # parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose (default: False')

    args = parser.parse_args()
    args = vars(args)   # convert namespace to dictionary
    queries = args.pop('query_list')
    return queries, args


def main() -> None:
    queries, params = parse_args()
    for query in tqdm(queries):
        scrape(query, **params)


if __name__ == '__main__':
    main()
