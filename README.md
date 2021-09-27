

# HKU-COMP3361-Assignment1

Name: Zhiheng LYU

Email: 3035772432

UID: 3036772432

## 1. Motivation

Previous research have shown that training from scratch on a large enough dataset can just as well as pre-training [1]. So I tried two different models: one is Bert from scratch. The other uses N-gram to extract features and uses the random forest that is most commonly used on Kaggle for machine learning.



## 2.Dataset analysis

The training data set is not balanced, and there are some differences in the distribution under different labels. 

In train set, there are `140015 ` words in `['at', 'in', 'of', 'for', 'on']` in total， and `[('at', 9070), ('in', 44981), ('of', 57030), ('for', 13794), (on, 15140)]` , respectively.

In vaild set, there are `14634 ` words in `['at', 'in', 'of', 'for', 'on']` in total， and `[('at', 1005), ('in', 4751), ('of', 5926), ('for', 1272), ('on', 1680)]` , respectively.

We can observed that the data distribution of the two datasets was not exactly the same, but very similar. Since our final goal is to improve the accuracy of all samples, we do not need to do over-sampling.



## 3. Train Bert from scratch

### 3.1 Generate the data[2]

Considering the lack of corpus text, using the whole word list will lead to the sparse distribution of samples in probability space, which will further result in the generation of over-fitting. I used a BPE tokenizer to reduce the number of words.



### 3.2 How to train[3],[4],[5]

Like traditional Bert, we use MaskLM for unsupervised training. To speed up the training, I put the data through` . ; , ` 's prioritized to short sentences of maximum length `128`.

#### Pretrain with MaskLM

 I tried several hyperparameters and finally found that `lr=5e-4; batch_size=64; warmup_steps=10000` is the most favorable for the loss function to decline, and the following is the result.

| train steps | loss function | accuracy in valid set |
| ----------- | ------------- | --------------------- |
| 20000       | 3.303300      | 0.6119                |
| 30000       | 2.912500      | 0.6381                |
| 40000       | 2.650100      | 0.6076                |

We can simply observe that after 30,000 steps, the performance of the model starts to get worse and worse, indicating that the data set is small enough to be remembered by the model and it loses its generalization ability. In short, it is not large enough to prevent overfitting.

#### Finetuning with remain set

We used checkpoint-30000 for finetuning, but the effect was not satisfactory: when we fixed Bert's parameters, using valid set as train set only brought 0.6 accuracy.



### 3.3 Some possible further explorations

Here are some possible ways to improve the overfit. Due to the lack of time, I didn't try any further.

#### Regularization

Modify the L2 regularized hyperparameters of the model to achieve a balance between overfitting and unlearning.

#### Data argument

Used the trained Bert model to add and modify some data and expand the size of the training set.



## 4. N-gram with Random forest

### 4.1 Generate the feature by Bi-directional N-gram

We use `class Ngram()` to count all n-tuples, and the end result is that we have a dict for all k-tuples, take n=5 to balance the amount of calculation and accuracy. To make sure that strings of less than n lengths can be counted, we add two padding lengths of n on each side of the string.

In order to better count the semantic information from the front and the back, we do n-gram classification for both directions. For each K-Gram classifier, we output features with a length of 5, representing the probability of each category respectively. Due to `k=1..n`, and we have forward and backward direction, the number of  total feature dimensions is `2*5*5=50`.

For each K-Gram classifier, we do use label smoothing and normalization.(Because n gram is used to text generation but we only need classification.) And the final value is:
$$
\text{Set }occur(x) \text{ as the occurance times in the Training text.}\\
\\
\\
P_i(X)=\frac{occur(i)+0.1}{\sum_k occur(k)+0.1}
$$


### 4.2 Naive training in Whole train set.[6]

We simply use random forest, a learning method that performs better when features are more regular.  And extract the feature for the Whole train set.  However the accuracy is 0.59.

In fact, after generating the training data with train Data, the text of trainset and validset is not identically distributed: In probability space, the occurrence probability of 5-gram is very low, but since trainset is used to generate 5-gram, we can generally observe this phenomenon in 5-gram, whereas in validset, this phenomenon is not observed.

So we considered two solutions: divide part of the validSet for training, or divide part of the Trainset for not constructing the n-Gram.



### 4.3 Split the validset to train

We randomly split the Validset into training set and test set, and then run the code.

```
              precision    recall  f1-score   support

           0       0.36      0.61      0.45       144
           1       0.82      0.71      0.76      1377
           2       0.87      0.81      0.84      1613
           3       0.40      0.57      0.47       220
           4       0.58      0.76      0.66       305

    accuracy                           0.75      3659
   macro avg       0.60      0.69      0.64      3659
weighted avg       0.78      0.75      0.76      3659
```



### 4.4 Split the trainset and test on valid set

We used the last 2200 lines of trainset as the data of training random forest, the results are as follows:

```
              precision    recall  f1-score   support

           0     0.3383    0.7219    0.4607       471
           1     0.7897    0.7013    0.7429      5350
           2     0.8920    0.7630    0.8225      6928
           3     0.3797    0.6184    0.4705       781
           4     0.5286    0.8043    0.6379      1104

    accuracy                         0.7345     14634
   macro avg     0.5857    0.7218    0.6269     14634
weighted avg     0.7820    0.7345    0.7490     14634

```

![img](data:image/png;base64,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)

**Since we are not allowed to train with validSet, the final answer will be generated by this model.**

## 5. Reference

[1] https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf

[2] https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

[3] https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=imY1oC3SIrJf

[4] https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=G-kkz81OY6xH

[5] https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=imY1oC3SIrJf

[6] https://blog.csdn.net/jasonzhoujx/article/details/81911799

## Appendix 

### A. Generate `test.fo2` from huggingface pretrained bert

The notebooks `COMP3361Assignment1GenerateTestData.ipynb`  is a simple example of generating labels from Bert.

**Although I directly used Bert to obtain a label with high accuracy, I did not use it for fitting when training my model.**

I also shared it with `Xijia Tao`, `Jiayi Xin`

