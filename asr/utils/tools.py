#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

def empty_dir(src):
    if os.path.isfile(src):
        try:
            os.remove(src)
        except:
            pass
    elif os.path.isdir(src):
        for item in os.listdir(src):
            itemsrc = os.path.join(src, item)
            empty_dir(itemsrc)
        try:
            os.rmdir(src)
        except:
            pass

def value_to_key(temp_dict, value):
    return list(temp_dict.keys())[list(temp_dict.values()).index(value)]

