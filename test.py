import os
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle

data_dir = "data"

def read_file_to_dataset(fp):
    ds = DataSet()
    with open(fp, 'r') as f:
        f.readline()  # 第一行是title名称，忽略掉
        for line in f:
            # print(line)
            line = line.strip()
            _, target, chars = line.split(',')
            ins = Instance(target=chars, raw_chars=target)
            ds.append(ins)
    return ds

data_bundle = DataBundle()
for name in ['test1.tsv']:
    fp = os.path.join(data_dir, name)
    ds = read_file_to_dataset(fp)
    data_bundle.set_dataset(name="test", dataset=ds)

print(data_bundle)  # 查看以下数据集的情况
# In total 3 datasets:
#    train has 9600 instances.
#    dev has 1200 instances.
#    test has 1200 instances.

data_bundle.get_dataset("test").save()