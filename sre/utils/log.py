#!/usr/bin/env python2.7
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import logging.handlers

def get_logger(log_path):
    print("log_path:", log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.handlers.TimedRotatingFileHandler(log_path, 'D')
        fmt = logging.Formatter("%(asctime)s - %(pathname)s - %(filename)s - %(lineno)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

