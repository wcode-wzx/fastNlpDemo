import torch

from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric
from fastNLP.io.model_io import ModelSaver
from model.biLstmMaxPoolCls import BiLSTMMaxPoolCls
from dataset.dataSet import data_loader

# path_config
from config import get_train
path = get_train()
vocab_path, target_vocab_path, model_sava_path, embedding_path = path[0],path[1],path[2],path[3]

# 加载数据
data_bundle = data_loader(vocab_path, target_vocab_path)

from fastNLP.embeddings import StaticEmbedding

word2vec_embed = StaticEmbedding(data_bundle.get_vocab('chars'), model_dir_or_name=embedding_path)
# 初始化模型
model = BiLSTMMaxPoolCls(word2vec_embed, len(data_bundle.get_vocab('target')))

# 开始训练
loss = CrossEntropyLoss() 
optimizer = Adam(model.parameters(), lr=0.001)
metric = AccuracyMetric()
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss, 
                  optimizer=optimizer, batch_size=32, dev_data=data_bundle.get_dataset('dev'),
                  metrics=metric, device=device, n_epochs=1)


if  __name__=="__main__":
    trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型
    saver = ModelSaver(model_sava_path)
    saver.save_pytorch(model)

# 在测试集上测试一下模型的性能
from fastNLP import Tester
print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device, verbose=1)
tester.test()