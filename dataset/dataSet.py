from fastNLP import Vocabulary
from fastHan import FastHan
from dataset.dataLoad import data_load

# 分词
def data_bundle_cut(data_bundle):
    model=FastHan()
    # model.set_device('cuda')

    # 定义分词处理操作
    def word_seg(ins):
        raw_chars = ins['raw_chars']
        # 由于有些句子比较长，我们只截取前128个汉字
        raw_words = model(raw_chars[:128], target='CWS')[0]#target 参数可在 'Parsing'、'CWS'、'POS'、'NER' 四个选项中取值，模型将分别进行依存分析、分词、词性标注、命名实体识别任务
        return raw_words

    for name, ds in data_bundle.iter_datasets():
        # apply函数将对内部的instance依次执行word_seg操作，并把其返回值放入到raw_words这个field
        ds.apply(word_seg, new_field_name='raw_words')
        # 除了apply函数，fastNLP还支持apply_field, apply_more(可同时创建多个field)等操作
        # 同时我们增加一个seq_len的field
        ds.add_seq_len('raw_words')
    
    return data_bundle


def to_vocab(data_bundle, vocab_path, target_vocab_path):
    
    vocab = Vocabulary()

    # 对raw_words列创建词表, 建议把非训练集的dataset放在no_create_entry_dataset参数中
    vocab.from_dataset(data_bundle.get_dataset('train'), field_name='raw_words', 
                    no_create_entry_dataset=[data_bundle.get_dataset('dev'), 
                                                data_bundle.get_dataset('test')]) 

    # 将建立好词表的Vocabulary用于对raw_words列建立词表，并把转为序号的列存入到words列
    vocab.index_dataset(data_bundle.get_dataset('train'), data_bundle.get_dataset('dev'), 
                    data_bundle.get_dataset('test'), field_name='raw_words', new_field_name='words')
    
    vocab.save(vocab_path)
    # 建立target的词表，target的词表一般不需要padding和unknown
    target_vocab = Vocabulary(padding=None, unknown=None) 
    # 一般情况下我们可以只用训练集建立target的词表
    target_vocab.from_dataset(data_bundle.get_dataset('train'), field_name='target') 
    # 如果没有传递new_field_name, 则默认覆盖原词表
    target_vocab.index_dataset(data_bundle.get_dataset('train'), data_bundle.get_dataset('dev'), 
                    data_bundle.get_dataset('test'), field_name='target')
    target_vocab.save(target_vocab_path)
    # 我们可以把词表保存到data_bundle中，方便之后使用
    data_bundle.set_vocab(field_name='words', vocab=vocab)
    data_bundle.set_vocab(field_name='target', vocab=target_vocab)

    # 我们把words和target分别设置为input和target，这样它们才会在训练循环中被取出并自动padding, 有关这部分更多的内容参考
    data_bundle.set_target('target')
    data_bundle.set_input('words', 'seq_len')  # DataSet也有这两个接口
    # 如果某些field，您希望它被设置为target或者input，但是不希望fastNLP自动padding或需要使用特定的padding方式，请参考

    # 由于之后需要使用之前定义的BiLSTMMaxPoolCls模型，所以需要将words这个field修改为chars(因为该模型的forward接受chars参数)
    data_bundle.rename_field('words', 'chars')

    print(data_bundle.get_dataset('train')[:2])  # 我们可以看一下当前dataset的内容

    return data_bundle

def data_loader(vocab_path, target_vocab_path):
    # 加载数据
    data_bundle = data_load("data")
    data_bundle = data_bundle_cut(data_bundle)
    data_bundle = to_vocab(data_bundle, vocab_path, target_vocab_path)
    return data_bundle

if __name__=="__main__":
    data_loader()