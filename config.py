import os

def mkd(path):
    if not os.path.exists(path):
      os.makedirs(path)

# 创建 test文件夹
root = os.getcwd()+"/test/"
mkd(root)

def get_train():
    # 获取test文件列表
    rootlist = [int(i) for i in os.listdir(root)]
    try:
        x = max(rootlist)
        local = str(x+1)
    except:
        local = str(1)

    s_vocab = root+local+"/vocab/"
    s_model = root+local+"/model/"
    mkd(s_vocab)
    mkd(s_model)

    vocab_path = s_vocab+"vocab_.txt"
    target_vocab_path = s_vocab+"target_vocab_.txt"
    model_sava_path = s_model+"model_ckpt_200.pkl"
    embedding_path = "data/embddings/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"

    return [vocab_path, target_vocab_path, model_sava_path, embedding_path]


def get_test(num):
    vocab_path = root+str(num)+"/vocab/vocab_.txt"
    target_vocab_path = root+str(num)+"/vocab/target_vocab_.txt"
    model_sava_path = root+str(num)+"/model/model_ckpt_200.pkl"
    embedding_path = "data/embddings/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
    return [vocab_path, target_vocab_path, model_sava_path, embedding_path]

if __name__=="__main__":
    def hello():
        print("1")

    # print(callable(hello), callable(vocab_path))