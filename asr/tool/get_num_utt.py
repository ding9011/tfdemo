#!/usr/bin/env python2.7

import sys

def get_num_utt(spk2utt, num_utt):
    with open(spk2utt, 'r') as spk:
        with open(num_utt, 'w') as utt:
            for spk_line in spk:
                spk_items = spk_line.strip().split(' ')
                spk_id = spk_items[0]
                spk_num = len(spk_items) - 1
                utt.write(spk_id + ' ' + str(spk_num) + ' \n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: " + sys.argv[0] + " <spk2utt:string> <num_utt:string>"
        exit(1)
    spk2utt = sys.argv[1]
    num_utt = sys.argv[2]
    get_num_utt(spk2utt, num_utt)
