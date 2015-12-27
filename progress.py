#!python3
"""
Created on 2015-12-27
@author: yuqiang
Tool for showing progress in console
"""

import sys


def update(percent):
    """Update and show the progress in console

    Update and show the progress in console.
    Remember to call "update(1)" when all progress finished!

    Args:
        percent(Type: float): percent(0~1) of the part finished
    """

    if percent < 0 or percent > 1:
        print("Error input of progress.update()!")
        return

    bar_length = 20
    hashes = '#' * int(percent * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, percent * 100))
    if percent == 1:
        sys.stdout.write("\n")
    sys.stdout.flush()
