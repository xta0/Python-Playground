#discrete_probability_exercise

import matplotlib.pyplot as plt

count_data = [54,111,163,222,277,336,276,220,171,111,59] 
total_count = sum(count_data);

normalized_counts = []
for index,value in enumerate(count_data):
    p = value / total_count
    normalized_counts.append(p)

plt.bar(range(len(normalized_counts)),normalized_counts)
plt.show()
