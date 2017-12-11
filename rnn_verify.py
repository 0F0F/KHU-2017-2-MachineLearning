import rnn_predict
import numpy as np
import math
import matplotlib.pyplot as plt

import sys

def progress_bar(value, endvalue, bar_length = 20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{2}/{3}: [{0}] {1:.8f}%".format(arrow + spaces, 100. * percent, value, endvalue))
    sys.stdout.flush()

period = '6m'
seq_length = 3
hidden_dim = 6
path = './{}_{}_{}.ckpt'.format(period, seq_length, hidden_dim)

predictor = rnn_predict.predictor(path, seq_length, hidden_dim, 8, 1.6)
#xy = np.loadtxt('./6m_raw.csv', delimiter=',')
xy = np.loadtxt('real.csv', delimiter=',')


bot_benefit = 1
bot_krw_now = True
bot_bought_price = 0

optimal_benefit = 1
optimal_krw_now = True
optimal_bought_price = 0

x = [math.log(i) for i in xy]
y = []
pred = []

tp, tn, fp, fn = 0,0,0,0 
for i in range(seq_length):
    predictor.renew_data(x[i])

for i in range(seq_length, len(x) - 1):
    progress_bar(i, len(x) - 1)
    predictor.renew_data(x[i])
    predicted_price = predictor.predict()
    y.append(x[i+1])
    pred.append(predicted_price)

    if len(y) > 2:
        if y[-1] > y[-2] and optimal_krw_now:
            optimal_bought_price = x[i]
            optimal_krw_now = False
        if y[-1] < y[-2] and not optimal_krw_now:
            optimal_benefit *= math.exp(x[i]) / math.exp(optimal_bought_price)
            optimal_krw_now = True

    if len(pred) > 2:
        if pred[-1] <=  pred[-2]: # positive
            if y[-1] > y[-2]:
                tp += 1
            else:
                fp += 1

            if bot_krw_now:
                bot_bought_price = x[i]
                bot_krw_now = False
        else: # negative
            if y[-1] > y[-2]:
                fn += 1
            else:
                tn += 1
            if not bot_krw_now:
                bot_benefit *= math.exp(x[i]) / math.exp(bot_bought_price)
                bot_krw_now = True
print(tp, fp, tn, fn)
print("precision : {}".format(tp/(tp + fp)))
print("recall : {}".format(tp/(tp+fn)))

print(bot_benefit)
print(optimal_benefit)
plt.plot(pred)
plt.plot(y)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.grid()

plt.show()
