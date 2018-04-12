#!/usr/bin/env python2.7
#coding=utf-8

import ConfigParser
import sys
import json

all = ["init_config", "get", "set_section"]

_CR_ = None

class ConfigRead():

    def __init__(self, filename):
        self._cp = ConfigParser.SafeConfigParser()
        self._cp.read(filename)
        self._section = self._cp.sections()[0]

    def set_section(self, section):
        if self._cp.has_section(section):
            self._section = section
            return True
        else:
            return False

    def get(self, key, section=None):
        if section == None:
            section = self._section
        if self._cp.has_option(section, key):
            res = self._cp.get(section, key)
            try:
                if res.find('#') < 0:
                    res = res.strip()
                else:
                    res = res[:res.index("#")].strip()
                # verify boolean
                bool_res = res.upper()
                if bool_res == "FALSE":
                    return False
                elif bool_res == "TRUE":
                    return True
                return res
            except:
                return res
        else:
            return False

    def sections(self):
        return self._cp.sections();


def init_config(config_path):
    global _CR_
    _CR_ = ConfigRead(config_path)

def set_section(section):
    return _CR_.set_section(section)

def get(key, section=None):
    return _CR_.get(key, section)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: config.py conf"
        exit(1)
    conf_file = sys.argv[1]
    init_config(conf_file)
    print get("small_dataset_test", "dvector")
