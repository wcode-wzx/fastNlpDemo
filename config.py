from unicodedata import name


vocab_path = "data/vocab/vocab_.txt"
target_vocab_path = "data/vocab/target_vocab_.txt"
embedding_path = "data/embddings/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
model_sava_path = "data/model/model_ckpt_200.pkl"


if __name__=="__main__":
    def hello():
        print("1")

    print(callable(hello), callable(vocab_path))