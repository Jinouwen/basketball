import pickle
import matplotlib.pyplot as plt
import os

src_dir = "./model_and_curve"
color = ['red', 'blue', 'black', 'green', 'orange', 'grey', 'pink', 'purple']
# plt.figure(figsize=(4, 4))
i = 1


for fn in os.listdir(src_dir):
    if fn.endswith('curve.pkl'):
        with open(src_dir + '/' + fn, 'rb') as f:
            cur = pickle.load(f)
            recall, FAR = cur
            MPN = 1 - recall
            plt.plot(MPN, FAR, color=color[(i-1) % 5], linewidth=0.3, label='model_'+str(i))
            i = i + 1

plt.legend()
plt.ylabel("Missed Positive Number")
plt.xlabel("False Alarm Number")
plt.title("ROC")
plt.show()

