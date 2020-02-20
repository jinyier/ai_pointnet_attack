#!/usr/bin/env python3


import os
import sys


def get_input_files(dir_or_files, followlinks=True):
    inputfiles = []
    for path in dir_or_files:
        if not os.path.exists(path):
            print('{}: no such file or directory'.format(path), file=sys.stderr)
            continue
        if os.path.isdir(path):
            print('Processing files in directory {}'.format(path))
            for root, dirs, files in os.walk(path, followlinks=followlinks):
                for fn in files:
                    fullpath = os.path.join(root, fn)
                    if os.path.isfile(fullpath):
                        inputfiles.append(fullpath)
                    else:
                        print('{}: not a regular file, ignored.'.format(fullpath), file=sys.stderr)
        elif os.path.isfile(path):
            inputfiles.append(path)
        else:
            print('{}: not a regular file or directory, ignored.'.format(path), file=sys.stderr)
    return sorted(list(set(inputfiles)))
