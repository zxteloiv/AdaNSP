# Load files for specific formats

import json
import bz2
import gzip
import os


def open_json(infile):
    if isinstance(infile, str):
        infile = open_file(infile)

    for json_str in infile:
        try:
            yield json.loads(json_str)

        except json.JSONDecodeError as e:
            continue


def open_file(filename: str):
    assert isinstance(filename, str)
    if filename.endswith('.bz2'):
        f = bz2.BZ2File(filename)
    elif filename.endswith('.gz'):
        f = gzip.GzipFile(filename)
    else:
        f = open(filename)

    return f


def open_dir(dirname: str):
    for file in os.listdir(dirname):
        for line in open_file(file):
            yield line


def open_dir_or_file(objname: str):
    assert objname
    if os.path.isdir(objname):
        return open_dir(objname)
    else:
        return open_file(objname)

