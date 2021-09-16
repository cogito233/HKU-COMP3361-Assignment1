"""
This File is used to load the data
"""
DATA_PATH = '../data/'

class DataFliter(object):
    def __init__(self, data_path, data_name):
        self.data = []
        with open(data_path+data_name, 'r') as f:
            for line in f:
                for sentence in line.split('.'):
                    # for subsentence in sentence.split(','):
                    if (self.check_valid(sentence)):
                        self.data.append(sentence)
    def check_valid(self, s):
        return (' at ' in s) or (' in ' in s) or (' of ' in s) or (' for ' in s) or (' on ' in s) or (' **GW** ' in s)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

train_data = DataFliter(DATA_PATH, 'train.fo1')

def make_tokenizer(files):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "**GW**", ])

    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)

    tokenizer.save("tokenizer.json")

    tokenizer = Tokenizer.from_file("tokenizer.json")

    return tokenizer

def load_tokenizer():
    tokenizer = Tokenizer.from_file("tokenizer.json")
    return tokenizer

def pertrained_tokenizer():
    from transformers import AutoTokenizer

    model_checkpoint = "distilbert-base-uncased"
    tokenizer_pertrained = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer_pertrained

class DataGenerator(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, tokenizer):
        # 分词器
        self.tokenizer = tokenizer
        self.label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on']
        self.label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5}

    def tokenizerFunc(self,str):
        x = self.tokenizer.encode(str)
        return {"input_ids":x.ids, "tokens":x.tokens, "attention_mask":x.attention_mask}

    def maskFunc(self, str):
        data_dict = self.tokenizerFunc(str)
        n = len(data_dict["input_ids"])
        data_dict["label"] = [-100]*n
        for i in range(n):
            if data_dict["tokens"][i] in self.label_list:
                data_dict["attention_mask"][i] = 0
                data_dict["label"][i] = self.label_dict[data_dict["tokens"][i]]
        data_dict['input_ids'] += [0]*(512-n)
        data_dict['tokens'] += ['']*(512-n)
        data_dict['attention_mask'] += [0]*(512-n)
        data_dict['label'] += [-100]*(512-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)

class DataGenerator_pertrained(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, tokenizer):
        # 分词器
        self.tokenizer = tokenizer
        self.label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on']
        self.label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5}

    def tokenizerFunc(self,str):
        x = self.tokenizer(str)
        return {"input_ids":x.input_ids, "attention_mask":x.attention_mask,
                "tokens":self.tokenizer.convert_ids_to_tokens(x["input_ids"])}

    def maskFunc(self, str):
        data_dict = self.tokenizerFunc(str)
        n = len(data_dict["input_ids"])
        data_dict["label"] = [-100]*n
        for i in range(n):
            if data_dict["tokens"][i] in self.label_list:
                data_dict["attention_mask"][i] = 0
                data_dict["label"][i] = self.label_dict[data_dict["tokens"][i]]
        data_dict['input_ids'] += [0]*(512-n)
        data_dict['attention_mask'] += [0]*(512-n)
        data_dict['tokens'] += ['']*(512-n)
        data_dict['label'] += [-100]*(512-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)


def generate_dataset(dataGenerator):
    dict_data = {'input_ids': [], 'tokens': [], 'attention_mask': [], 'labels': []}
    for i in train_data:
        example = dataGenerator(i)
        dict_data['input_ids'].append(example['input_ids'])
        dict_data['tokens'].append(example['tokens'])
        dict_data['attention_mask'].append(example['attention_mask'])
        dict_data['labels'].append(example['label'])

    from datasets import Dataset
    import datasets
    dataset = Dataset.from_dict(dict_data)

    dataset.shuffle()
    sub_dataset = [dataset.shard(num_shards=5, index=i) for i in range(5)]
    train_set = datasets.concatenate_datasets([sub_dataset[j] for j in range(5) if j != 0])
    text_set = sub_dataset[0]
    return train_set, text_set


def test():
    print(2333)

if __name__ == '__main__':
    test()