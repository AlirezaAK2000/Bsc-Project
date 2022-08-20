#!/bin/sh

python config.py --map Town02 --no-rendering

python spawn_npc.py -n 0 -w 150
