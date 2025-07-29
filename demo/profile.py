import csv
import pandas as pd

def summarize_profiling(csv_path: str) -> dict:
    totals = {}
    total_nums = {}
    avgs = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    val = float(v)
                except ValueError:
                    continue
                if val != 0.0:
                    if k not in total_nums:
                        total_nums[k] = 0
                    total_nums[k] += 1
                totals[k] = totals.get(k, 0.0) + val
    print(pd.DataFrame([totals]).T)
    for k, v in totals.items():
        if k in total_nums and total_nums[k] > 0:
            avgs[k] = v / total_nums[k]
        else:
            avgs[k] = 0.0
    print(pd.DataFrame([avgs]).T)

summarize_profiling('attention_profile.csv')