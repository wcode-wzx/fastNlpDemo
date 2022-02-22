import os
from fastNLP import DataSet, Instance
from fastNLP.io import DataBundle


def read_file_to_dataset(fp):
    ds = DataSet()
    with open(fp, 'r') as f:
        f.readline()  # 第一行是title名称，忽略掉
        for line in f:
            # print(line)
            line = line.strip()
            _, target, chars = line.split('\t')
            ins = Instance(target=target, raw_chars=chars)
            ds.append(ins)
    return ds


def data_load(data_dir):
    # 
    if data_dir:
        data_dir = data_dir
    else:
        data_dir = "../data/"
        

    name_list = ['train.tsv','dev.tsv','test.tsv']
    data_bundle = DataBundle()
    for name in name_list:
        fp = os.path.join(data_dir, name)
        ds = read_file_to_dataset(fp)
        data_bundle.set_dataset(name=name.split(".")[0], dataset=ds)

    print(data_bundle)  # 查看以下数据集的情况
    # In total 3 datasets:
    #    train has 9600 instances.
    #    dev has 1200 instances.
    #    test has 1200 instances.
    
    return data_bundle


if __name__=="__main__":
    data_load("")