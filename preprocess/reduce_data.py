import sys
import math
from collections import deque


def encode_examples(infile, outfile, dicts):
    userdict, itemdict, catedict = dicts
    fw = open(outfile, "w")

    source = []
    target = []

    with open(infile, "r") as fr:
        for line in fr:
            ss = line.strip("\n").split("\t")
            uid = userdict[ss[1]] if ss[1] in userdict else 0
            mid = itemdict[ss[2]] if ss[2] in itemdict else 0
            cat = catedict[ss[3]] if ss[3] in catedict else 0
            timestepnow = float(ss[4])

            tmp = []
            for fea in ss[5].split(""):
                m = itemdict[fea] if fea in itemdict else 0
                tmp.append(m)
            mid_list = tmp

            tmp1 = []
            for fea in ss[6].split(""):
                c = catedict[fea] if fea in catedict else 0
                tmp1.append(c)
            cat_list = tmp1

            tmp2 = []
            for fea in ss[7].split(""):
                tmp2.append(float(fea))
            time_list = tmp2

            # Time-LSTM-123
            tmp3 = []
            for i in range(len(time_list) - 1):
                deltatime_last = (time_list[i + 1] - time_list[i]) / (3600 * 24)
                if deltatime_last <= 0.5:
                    deltatime_last = 0.5
                tmp3.append(math.log(deltatime_last))
            deltatime_now = (timestepnow - time_list[-1]) / (3600 * 24)
            if deltatime_now <= 0.5:
                deltatime_now = 0.5
            tmp3.append(math.log(deltatime_now))
            timeinterval_list = tmp3

            # Time-LSTM-4
            tmp4 = []
            tmp4.append(0.0)
            for i in range(len(time_list) - 1):
                deltatime_last = (time_list[i + 1] - time_list[i]) / (3600 * 24)
                if deltatime_last <= 0.5:
                    deltatime_last = 0.5
                tmp4.append(math.log(deltatime_last))
            timelast_list = tmp4

            tmp5 = []
            for i in range(len(time_list)):
                deltatime_now = (timestepnow - time_list[i]) / (3600 * 24)
                if deltatime_now <= 0.5:
                    deltatime_now = 0.5
                tmp5.append(math.log(deltatime_now))
            timenow_list = tmp5

            source.append(
                [
                    uid,
                    mid,
                    cat,
                    mid_list,
                    cat_list,
                    timeinterval_list,
                    timelast_list,
                    timenow_list,
                ]
            )
            target.append([float(ss[0]), 1 - float(ss[0])])


def reduce_examples(infile, outfile):
    """
    The original train_data contains all sub-sequences for a particular user
    resulting in a large number of samples. The objective of this code is to
    reduce the same to only the last example - one positive and one negative.

    Note: test_data is already in the required shape, only two examples per user

    """
    fw = open(outfile, "w")

    last_user = None
    out_list = deque(maxlen=2)
    count_line = 0
    count_user = 0
    with open(infile, "r") as fr:
        for line in fr:
            ss = line.strip("\n").split("\t")
            user = ss[1]
            if not last_user:
                last_user = user
                count_user += 1
            if user != last_user:
                if len(out_list) == 2:
                    for example in out_list:
                        fw.write(example)
                        count_line += 1
                out_list.clear()
                last_user = user
                count_user += 1
            else:
                out_list.append(line)
    fw.close()
    print(f"Wrote {count_line} examples for {count_user} users in {outfile}")


if __name__ == "__main__":
    inp_file, out_file = "data/train_data", "data/train_data_reduced"
    reduce_examples(inp_file, out_file)
