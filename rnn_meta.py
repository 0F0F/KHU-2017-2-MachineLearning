import os
import datetime


exe = "python3 rnn_train.py "
arg_skeleton = "-s {} -h {} -r {} -c {}"

cnt = 0
for hidden_dim in range(3, 10):
    for seq_length in range(3, 16):
        arg = arg_skeleton.format(seq_length, hidden_dim, '6m', cnt)
        os.system(exe + arg)
        arg = arg_skeleton.format(seq_length, hidden_dim, '1y', cnt)
        os.system(exe + arg)
        arg = arg_skeleton.format(seq_length, hidden_dim, '2y', cnt)
        os.system(exe + arg)

        print("[{}] {}_{}_{}".format(datetime.datetime.now(), seq_length, hidden_dim, cnt)
        cnt += 1
