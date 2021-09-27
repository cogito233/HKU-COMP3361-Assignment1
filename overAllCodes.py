DATA_PATH = '/content/drive/MyDrive/COMP3361/Assignment1/data/'
!head -n 10 /content/drive/MyDrive/COMP3361/Assignment1/data/valid.fo1
class DataFliter(object):
    def __init__(self, data_path, data_name):
        self.data = []
        ans = 0
        with open(data_path+data_name, 'r') as f:
            for line in f:
                for sentence in line.split('.'):
                    for subsentence in sentence.split(';'):
                        if len(subsentence)>300:
                            for subsubsentence in subsentence.split(','):
                                if (self.check_valid(subsubsentence)):
                                    self.data.append(subsubsentence)
                        else:
                            if (self.check_valid(subsentence)):
                                self.data.append(subsentence)
    def check_valid(self, s):
        return (' at ' in s) or (' in ' in s) or (' of ' in s) or (' for ' in s) or (' on ' in s) or ('**GW**' in s)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
class DataFliter_subsentence(object):
    def __init__(self, data_path, data_name):
        self.data = []
        ans = 0
        with open(data_path+data_name, 'r') as f:
            for line in f:
                for sentence in line.split('.'):
                    for subsentence in sentence.split(';'):
                        for subsubsentence in subsentence.split(','):
                            if subsubsentence[len(subsubsentence)-1]=='\n':
                                subsubsentence=subsubsentence[0:len(subsubsentence)-1]
                            if subsubsentence!='':
                                self.data.append(subsubsentence)
    def check_valid(self, s):
        return ('at' in s) or (' in ' in s) or (' of ' in s) or (' for ' in s) or (' on ' in s) or (' **GW** ' in s)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

train_data = DataFliter(DATA_PATH, 'train.fo1')

print(train_data[0])
print(train_data[1])
print(train_data[2])
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

files = ["/content/drive/MyDrive/COMP3361/Assignment1/data/valid.fo1",
         "/content/drive/MyDrive/COMP3361/Assignment1/data/train.fo1"]
paths = files

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=10000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
    "**GW**",
])
!mkdir tokenizer
tokenizer.save_model("./tokenizer")
x=tokenizer.encode(" **GW** in forest in **GW** 2011 in japan , it is the third game in the valkyria series ")
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(vocab_size=10000,special_tokens=["[UNK]", "[CLS]", "[SEP]",
                                "[PAD]", "[MASK]","**GW**","at", 'in', 'of', 'for', 'on'])


from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
tokenizer.pre_tokenizer = Whitespace()


files = ["/content/drive/MyDrive/COMP3361/Assignment1/data/valid.fo1",
         "/content/drive/MyDrive/COMP3361/Assignment1/data/train.fo1"]

tokenizer.train(files, trainer)

tokenizer.save("tokenizer-wiki.json")

tokenizer = Tokenizer.from_file("tokenizer-wiki.json")

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
tokenizer = Tokenizer.from_file("tokenizer-wiki.json")
x.tokens
def tokenizerFunc(str):
    x = tokenizer.encode(str)
    return {"input_ids":x.ids, "attention_mask":x.attention_mask}
class DataGenerator(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, tokenizer):
        # 分词器
        self.tokenizer = tokenizer
        self.label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on', 'Ġat', 'Ġin', 'Ġof', 'Ġfor', 'Ġon']
        self.label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5,
                           'Ġat':1, 'Ġin':2, 'Ġof':3, 'Ġfor':4, 'Ġon':5
                           }
        self.label_stat = [0,0,0,0,0,0]

    def tokenizerFunc(self,str):
        x = self.tokenizer.encode(str)
        return {"input_ids":x.ids, "tokens":x.tokens, "attention_mask":x.attention_mask}

    def maskFunc(self, str):
        data_dict = self.tokenizerFunc(str)
        n = len(data_dict["input_ids"])
        if (n>128):
            print(n)
        data_dict["label"] = [-100]*n
        for i in range(n):
            if data_dict["tokens"][i] in self.label_list:
                data_dict["attention_mask"][i] = 0
                data_dict["label"][i] = self.label_dict[data_dict["tokens"][i]]
                self.label_stat[self.label_dict[data_dict["tokens"][i]]] += 1
        data_dict['input_ids'] += [0]*(128-n)
        data_dict['tokens'] += ['']*(128-n)
        data_dict['attention_mask'] += [0]*(128-n)
        data_dict['label'] += [-100]*(128-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)


dataGenerator = DataGenerator(tokenizer)
print(dataGenerator(" released in january 2011 in japan , it is the third game in the valkyria series ")["attention_mask"])
dataGenerator = DataGenerator(tokenizer)
dict_data={'input_ids':[]}
for i in train_data:
    example = dataGenerator(i)
    dict_data['input_ids'].append(example['input_ids'])

from datasets import Dataset
dataset = Dataset.from_dict(dict_data)
dataGenerator = DataGenerator(tokenizer)
dict_data={'input_ids':[], 'tokens':[], 'attention_mask':[], 'label':[]}
for i in train_data:
    example = dataGenerator(i)
    dict_data['input_ids'].append(example['input_ids'])
    dict_data['tokens'].append(example['tokens'])
    dict_data['attention_mask'].append(example['attention_mask'])
    dict_data['label'].append(example['label'])

from datasets import Dataset
train_dataset = Dataset.from_dict(dict_data)
import datasets
dataset = train_dataset
dataset.shuffle()
sub_dataset = [dataset.shard(num_shards =5 , index = i) for i in range(5)]
train_set = datasets.concatenate_datasets([sub_dataset[j] for j in range(5) if j!=0])
text_set = sub_dataset[0]
valid_data = DataFliter(DATA_PATH, 'valid.fo1')
valid_data_label = DataFliter_subsentence(DATA_PATH,'valid.fo2')
len(valid_data)
class DataGenerator_valid(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, tokenizer, labels):
        # 分词器
        self.tokenizer = tokenizer
        self.label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on', 'Ġat', 'Ġin', 'Ġof', 'Ġfor', 'Ġon']
        self.label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5,
                           'Ġat':1, 'Ġin':2, 'Ġof':3, 'Ġfor':4, 'Ġon':5}
        self.label_stat = [0,0,0,0,0,0]
        self.labels = labels
        self.label_num = 0

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
                # list the labels
                if (data_dict["tokens"][i] == '**GW**'):
                    data_dict["label"][i] = self.label_dict[self.labels[self.label_num]]
                    self.label_num += 1
        # padding
        data_dict['input_ids'] += [0]*(128-n)
        data_dict['tokens'] += ['']*(128-n)
        data_dict['attention_mask'] += [0]*(128-n)
        data_dict['label'] += [-100]*(128-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)

dataGenerator = DataGenerator_valid(tokenizer, valid_data_label)
dict_data={'input_ids':[], 'tokens':[], 'attention_mask':[], 'label':[]}
for i in valid_data:
    example = dataGenerator(i)
    dict_data['input_ids'].append(example['input_ids'])
    dict_data['tokens'].append(example['tokens'])
    dict_data['attention_mask'].append(example['attention_mask'])
    dict_data['label'].append(example['label'])
print(dataGenerator.label_num)
from datasets import Dataset
valid_dataset = Dataset.from_dict(dict_data)
from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
tokenizer_pertrained = AutoTokenizer.from_pretrained(model_checkpoint)
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
        data_dict['input_ids'] += [0]*(200-n)
        data_dict['attention_mask'] += [0]*(200-n)
        data_dict['tokens'] += ['']*(200-n)
        data_dict['label'] += [-100]*(200-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)


class DataGenerator_valid_pertrained(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, tokenizer, labels):
        # 分词器
        self.tokenizer = tokenizer
        self.label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on']
        self.label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5}
        self.labels = labels
        self.label_num = 0

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
                # list the labels
                if (data_dict["tokens"][i] == '**GW**'):
                    data_dict["label"][i] = self.label_dict[self.labels[self.label_num]]
                    self.label_num += 1
        # padding
        data_dict['input_ids'] += [0]*(128-n)
        data_dict['tokens'] += ['']*(128-n)
        data_dict['attention_mask'] += [0]*(128-n)
        data_dict['label'] += [-100]*(128-n)
        return data_dict

    def __call__(self, example):
        return self.maskFunc(example)

dataGenerator = DataGenerator_pertrained(tokenizer_pertrained)
dict_data={'input_ids':[], 'tokens':[], 'attention_mask':[], 'labels':[]}
for i in train_data:
    example = dataGenerator(i)
    dict_data['input_ids'].append(example['input_ids'])
    dict_data['tokens'].append(example['tokens'])
    dict_data['attention_mask'].append(example['attention_mask'])
    dict_data['labels'].append(example['label'])

from datasets import Dataset
dataset = Dataset.from_dict(dict_data)
len(dataset[0]['tokens'])
train_dataset=dataset
dataGenerator = DataGenerator_valid_pertrained(tokenizer_pertrained, valid_data_label)
dict_data={'input_ids':[], 'tokens':[], 'attention_mask':[], 'labels':[]}
for i in valid_data:
    example = dataGenerator(i)
    dict_data['input_ids'].append(example['input_ids'])
    dict_data['tokens'].append(example['tokens'])
    dict_data['attention_mask'].append(example['attention_mask'])
    dict_data['labels'].append(example['label'])
print(dataGenerator.label_num)
from datasets import Dataset
valid_dataset = Dataset.from_dict(dict_data)
len(valid_data_label)
import datasets

dataset.shuffle()
sub_dataset = [dataset.shard(num_shards =10 , index = i) for i in range(10)]
train_set = datasets.concatenate_datasets([sub_dataset[j] for j in range(10) if j!=0])
text_set = sub_dataset[0]
!pip install seqeval
import numpy as np
from datasets import load_metric

metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[int(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[int(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
import numpy as np
!pip install seqeval
from datasets import load_metric

metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    print(true_predictions)
    print(true_labels)

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on']
model_checkpoint = "distilbert-base-uncased"
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"test-3361-pertrain",
    evaluation_strategy = "steps",
    learning_rate=2e-4,# TO steps
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.0001,# TO change
)

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
del model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def get_one_hot(label):
            size = label.size()
            # create one-hot vector for label map
            oneHot_size = (size[0], size[1])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().cuda()
            input_label = input_label.cuda().scatter_(1, label.data.long(), 1.0)
            return input_label
        labels = inputs.pop("labels")
        
        labels = labels.unsqueeze(2).repeat(1,1,6)

        outputs = model(**inputs)
        logits = outputs.logits
        # logits = F.softmax(logits, dim=2)


        logits = logits[labels!=-100]
        labels = labels[labels!=-100]

        logits = logits.view(-1, self.model.config.num_labels)
        labels = get_one_hot(labels.view(-1, self.model.config.num_labels))

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits,
                        labels.float())
        return (loss, outputs) if return_outputs else loss
del trainer
trainer = MultilabelTrainer(
    model,
    args,
    train_dataset=train_set,
    eval_dataset=text_set,
    compute_metrics=compute_metrics,
    
)

trainer.args=args
trainer.train()
trainer.eval_dataset
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("/content/drive/MyDrive/COMP3361/model20", max_len=512)
model=model.from_pretrained("/content/drive/MyDrive/COMP3361/model20")
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=10000,
    max_position_embeddings=130,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)
model.num_parameters()
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/COMP3361",
    learning_rate = 5e-4,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=64,
    save_steps=10000,
    prediction_loss_only=True,
    warmup_steps = 10000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model("/content/drive/MyDrive/COMP3361/model_pertrain")
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="/content/drive/MyDrive/COMP3361/checkpoint-10000",
    tokenizer="/content/drive/MyDrive/COMP3361/model_pertrain",

)
def generate_data(st):
    li=[]
    while st.find("**GW**")!=-1:
        s = st[:]
        s=s.replace("**GW**", "<mask>", 1)
        s=s.replace("**GW**", "<unk>")
        li.append(s)
        st=st.replace("**GW**", "<unk>", 1)
    return li
generate_data(" is a species **GW** <unk> lobster from the eastern atlantic ocean , mediterranean sea and parts **GW** the black sea ")
confution_matrix=np.zeros([6,6])
label_num = 0
label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on', 'Ġat', 'Ġin',
              'Ġof', 'Ġfor', 'Ġon',' at', ' in', ' of', ' for', ' on']
label_dict = {'**GW**':0, 'at':1, 'in':2, 'of':3, 'for':4, 'on':5,
                           'Ġat':1, 'Ġin':2, 'Ġof':3, 'Ġfor':4, 'Ġon':5,
                           ' at':1, ' in':2, ' of':3, ' for':4, ' on':5,
                }
preds=[]
pred_str=[]
acc_pred = 0
for i in valid_data:
    for s in generate_data(i):
        predict_list=fill_mask(s)
        pred_str.append(s)
        flag = False
        for ele in predict_list:
            if (ele["token_str"] in label_list):
                if (not flag and label_dict[valid_data_label[label_num]]
                        ==label_dict[ele["token_str"]]):
                    acc_pred += 1
                flag = True
                confution_matrix[label_dict[valid_data_label[label_num]]][label_dict[ele["token_str"]]] += 1
                preds.append(label_dict[ele["token_str"]])
                break
        if (not flag and label_dict[valid_data_label[label_num]]==3):
            acc_pred += 1
            confution_matrix[label_dict[valid_data_label[label_num]]][0] += 1
        if (not flag):
            preds.append(0)
        label_num += 1
        if label_num % 100==0:
            print(label_num)
        if label_num==20000:
            break
    if label_num==20000:
        break

print(acc_pred/label_num)
preds_new=[]
for i in preds_lists:
    flag = False
    for ele in i:
        if (ele["token_str"] in label_list):
            flag = True
            preds_new.append(label_dict[ele["token_str"]])
            break
    if (not flag):
        preds_new.append(0)

import torch.nn as nn
from transformers import AutoModel 
class myModel(nn.Module):
    def __init__(self, freeze_bert=False, model_name="/content/drive/MyDrive/COMP3361/checkpoint-30000", hidden_size=768, num_classes=6):
        super(myModel, self).__init__()
        # 返回输出的隐藏层
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.config = RobertaConfig(
            vocab_size=10000,
            max_position_embeddings=130,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            num_labels=6
        )
        for p in self.bert.parameters():
            p.requires_grad = False
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*4, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )
        '''
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size*4, num_classes, bias=False),
        )
        self.hidden_size=hidden_size
  
    def forward(self, input_ids, attention_mask):
    	#其实只需要input_ids也可以
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 将最后输入的隐藏层后四个进行拼接
        size=input_ids.shape
        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1) # [bs, seq_len, hidden_dim*4]
        #first_hidden_states = hidden_states[:, 0, :] # [bs, hidden_dim*4] # 这里取的是cls的embedding表示，
        #它代表了一句话的embedding
        hidden_states = torch.reshape(hidden_states, [size[0]*size[1],self.hidden_size*4])
        
        logits = self.fc(hidden_states)
        return torch.reshape(logits, [size[0],size[1],6])
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def get_one_hot(label):
            size = label.size()
            # create one-hot vector for label map
            oneHot_size = (size[0], size[1])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().cuda()
            input_label = input_label.cuda().scatter_(1, label.data.long(), 1.0)
            return input_label
        labels = inputs.pop("labels")
        
        labels = labels.unsqueeze(2).repeat(1,1,6)

        outputs = model(**inputs)
        logits = outputs
        # logits = F.softmax(logits, dim=2)
        
        logits = logits[labels!=-100]
        labels = labels[labels!=-100]

        logits = logits.view(-1, self.model.config.num_labels)
        labels = get_one_hot(labels.view(-1, self.model.config.num_labels))

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits,
                        labels.float())
        return (loss, {"logics":outputs}) if return_outputs else loss
from transformers import TrainingArguments, Trainer

label_list = ['**GW**', 'at', 'in', 'of', 'for', 'on']
model_checkpoint = "distilbert-base-uncased"
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"test-3361-pertrain",
    evaluation_strategy = "epoch",
    learning_rate=5e-4,# TO steps
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps = 10000,
    num_train_epochs=5,
    weight_decay=0.01,# TO change
)

del trainer
trainer = MultilabelTrainer(
    Mmodel,
    args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    
)

trainer.train()
trainer.evaluate()













"""
Follows are from n-gram
"""


label_list = ['at', 'in', 'of', 'for', 'on', '**GW**']
label_dict = {'at':0, 'in':1, 'of':2, 'for':3, 'on':4, '**GW**':5, '<unk>':6}
n = 5
class DataFliter(object):
    def __init__(self, data_path, data_name):
        self.data = []
        ans = 0
        with open(data_path+data_name, 'r') as f:
            for line in f:
                self.data.append(line)
    def check_valid(self, s):
        return (' at ' in s) or (' in ' in s) or (' of ' in s) or (' for ' in s) or (' on ' in s) or ('**GW**' in s)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
train_data = DataFliter(DATA_PATH, 'train.fo1')
len(train_data)

from typing import List
from collections import Counter
class Ngram(object):
    def __init__(self, n):
        self.n = n
        self.stat = [None] * (n+1)
        self.stat_final = [None] * (n+1)
        self.ans = 0
    def add_sentence(self, str_list):
        padding = [" "]*(self.n+1)
        str_list =  padding + str_list + padding
        self.ans += len(str_list)-self.n-1
        for i in range(len(str_list)-self.n-1):
            for j in range(self.n+1):
                if self.stat[j] == None:
                    self.stat[j] = []
                self.stat[j].append(str(str_list[i:i+j+1]))

    def finish(self):
        for i in range(self.n+1):
            self.stat_final[i] = Counter(self.stat[i])

ngram=Ngram(5)
for i in train_data[0:14670]:
    ngram.add_sentence(i.split(' '))
    ngram.finish()
nngram = ngram

counted = sorted(Counter(labels_num).items())
Sum=0
for i in counted:
    Sum+=i[1]
print(counted, Sum)

def generate_Datasets(str_list):
    global Sum
    # str_list list[2*n+1]
    feature = []# will be list[2*n*5]
    for i in range(n):
        subStrList = str_list[n+1:n+i+2]
        a = [0]*5
        sum = 0
        for j in range(5):
            a[j] = ngram.stat_final[i+1].get(str([label_list[j]]+subStrList), 0)
            a[j] += 0.1
            sum += a[j]
        a = [i/sum for i in a]
        feature += a
    
    for i in range(n):
        subStrList = str_list[n-i-1:n]
        a = [0]*5
        sum = 0
        for j in range(5):
            a[j] = ngram.stat_final[i+1].get(str(subStrList+[label_list[j]]),  0)
            a[j] += 0.1
            sum += a[j]
        a = [i/sum for i in a]
        feature += a
    return feature, str_list[n]

padding = [' ']*n
features, labels = [], []
qwq=0
for i in train_data[13670:]:
    sequence = padding + i.split(' ') + padding
    for j in range(len(sequence)):
        if sequence[j] in label_list:
            feature, label = generate_Datasets(sequence[j-n:j+n+1])
            features.append(feature)
            labels.append(label)

labels_num=[label_dict[i] for i in labels]

len(features)

#用随机森林快速对数字进行分类
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(features, labels_num)
ypre = model.predict(valid_features)

#查看分类器的分类结果报告
from sklearn import metrics
print(metrics.classification_report(ypre, valid_labels_num,digits=4))

class DataFliter_subsentence(object):
    def __init__(self, data_path, data_name):
        self.data = []
        ans = 0
        with open(data_path+data_name, 'r') as f:
            for line in f:
                for sentence in line.split('.'):
                    for subsentence in sentence.split(';'):
                        for subsubsentence in subsentence.split(','):
                            if subsubsentence[len(subsubsentence)-1]=='\n':
                                subsubsentence=subsubsentence[0:len(subsubsentence)-1]
                            if subsubsentence!='':
                                self.data.append(subsubsentence)
    def check_valid(self, s):
        return ('at' in s) or (' in ' in s) or (' of ' in s) or (' for ' in s) or (' on ' in s) or (' **GW** ' in s)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

valid_data = DataFliter(DATA_PATH, 'valid.fo1')
valid_data_label = DataFliter_subsentence(DATA_PATH, 'valid.fo2')
label_num = 0

padding = [' ']*n
valid_features, valid_labels = [], []

for i in valid_data:
    sequence = padding + i.split(' ') + padding
    for j in range(len(sequence)):
        if sequence[j] == "**GW**":
            feature, label = generate_Datasets(sequence[j-n:j+n+1])
            valid_features.append(feature)
            valid_labels.append(valid_data_label[label_num])
            label_num += 1

valid_labels_num=[label_dict[i] for i in valid_labels]

import numpy as np
pred = model.predict(np.array(valid_features))

from sklearn import metrics
print(metrics.classification_report(pred, np.ascontiguousarray(np.array(valid_labels_num)),digits=4))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn
mat = confusion_matrix(pred, np.ascontiguousarray(np.array(valid_labels_num)))
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')



test_data = DataFliter(DATA_PATH, 'test.fo1')
padding = [' ']*n
test_features, test_labels = [], []
for i in test_data:
    sequence = padding + i.split(' ') + padding
    test_label = []
    for j in range(len(sequence)):
        if sequence[j] == "**GW**":
            feature, label = generate_Datasets(sequence[j-n:j+n+1])
            test_features.append(feature)
            test_label.append(label)
    test_labels.append(test_label)
import numpy as np
pred = model.predict(np.array(test_features))

import numpy as np
label_num = 0
pred = model.predict(np.array(test_features))
for i in test_labels:
    for j in range(len(i)):
        i[j] = label_list[pred[label_num]]
        label_num += 1
def processlist(x):
    str=""
    for i in x:
        str=str+i+","
    return str[:len(str)-1]+"\n"
with open(DATA_PATH+"test.fo3",'w') as f:
    for i in test_labels:
        f.write(processlist(i))
