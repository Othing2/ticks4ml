import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans


root_dir = '/home/worker/report_tmp/'
file_tran = 'trans3.txt'


idx, max_idx = 0, 100000
max_im = 0.0
unit_means = 0.0
trains_dt = []
with open(root_dir+file_tran, 'r') as fp:
    for line in fp:
        idx += 1
        name, pt, wt = line.strip().split('\t')
        if int(pt) > 0:
            unit = float(wt)*324305/int(pt)
            max_im = unit if unit > max_im else max_im
            if unit < 0.1:
                unit_means += unit
                trains_dt.append({'name': name, 'value': [unit, unit], 'line': idx})
            if max_idx and idx >= max_idx:
                break
    unit_means = float(unit_means)/len(trains_dt)

prob_list = list(filter(lambda x: x['value'][0]*10 < unit_means, trains_dt))

print('SUM: %s' % len(prob_list))
for dt in prob_list:
    print('    %s' % dt)


# ------------------------------------------------------------------------------

# trains_arr = np.array([dt['value'] for dt in trains_dt])
#
# y_pred = KMeans(n_clusters=100, max_iter=50).fit_predict(trains_arr)
#
# y_samples, y_means = {}, {}
# y_label = defaultdict(list)
# for i, l in enumerate(y_pred.tolist()):
#     y_samples[l] = y_samples[l] + 1 if y_samples.get(l, None) else 1
#     y_means[l] = y_means[l] + trains_dt[i]['value'][0] if y_means.get(l, None) else trains_dt[i]['value'][0]
#
# ss = sorted(y_samples.items(), key=lambda x: x[1])[:60]
# pre_label = set(i[0] for i in ss)
# for i, l in enumerate(y_pred.tolist()):
#     if l in pre_label:
#         y_label[l].append({'name': trains_dt[i]['name'], 'line': trains_dt[i]['line']})
#
# print(y_samples)
# print({k: float(v)/y_samples[k] for k, v in y_means.items()})
# for k, ll in y_label.items():
#     print(k)
#     for l in ll:
#         print('    %s' % l)
#
# plt.scatter(trains_arr[:, 0], trains_arr[:, 1], c=y_pred)
# plt.show()



