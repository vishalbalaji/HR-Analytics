from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

cf_matrix = np.array([[39.2, 5.8], [10.8, 44.2]])

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
cf = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
cf.get_figure().savefig("imgs/xbg_test.png")
