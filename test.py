import torch
from fastNLP.core.predictor import Predictor
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import ModelLoader
from model.biLstmMaxPoolCls import BiLSTMMaxPoolCls
from dataset.dataTest import data_loader

# path_config
from config import get_test
path = get_test(2)
vocab_path, target_vocab_path, model_sava_path, embedding_path = path[0],path[1],path[2],path[3]

data_bundle = data_loader(vocab_path,target_vocab_path, name="test.tsv")

# embedding
word2vec_embed = StaticEmbedding(data_bundle.get_vocab('chars'), model_dir_or_name=embedding_path)

# 初始化模型
model = BiLSTMMaxPoolCls(word2vec_embed, len(data_bundle.get_vocab('target')))
model_ = ModelLoader()
model_.load_pytorch(model,model_sava_path)


predictor = Predictor(network=model)
predictors = predictor.predict(data=data_bundle.get_dataset('test'), seq_len_field_name="seq_len")

for i in predictors:
  print(i)
  for j in predictors["pred"]:
    print(torch.tensor(j).argmax())
  print(len(predictors["pred"]))