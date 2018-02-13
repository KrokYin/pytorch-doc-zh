# -*- coding: utf-8 -*-
r"""
词嵌入 : 编码词汇语义
===========================================

词嵌入是每一个词汇的实数密集向量.在自然语言处理中，特征往往是单词!
但是如何在计算机中表示一个单词 ?你可以存储它的ascii字符表示，但是
它只能告诉你单词 *是* 什么，它没有多说它的 *含义*（你可以从它的词
缀或者大写字母中得到它的词性，但不多）.更何况，如何将这些表示结合在
一起？我们常常想从神经网络中得到稠密的输出，当输入是 :math:`|V|` 维
向量, :math:`V` 是我们的词典大小,但是我们的输出只有一小部分是维度是
实数。(如果我们只是预测一部分标签，例如).如何从一个庞大的向量空间
到一个较小的向量空间?如何代替 ascii 表示词汇，我们使用 one-hot 编码?
也就是说, 我们表示单词 :math:`w` 通过

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

1 出现在特定的位置 :math:`w`.任何单词在其对应的位置都是 1 ，其他的都是 0 .

除了向量巨大以外 , 还有一个致命的缺陷.这基本上将所有的单词之间当做独立无关的实体对待。
我们真正想要的是词汇之间的 *similarity* 相似性这一概念。为什么? 请看以下示例.

假设我们正在建立一个语言模型. 假设我们看过这些句子在我们训练数据中.

* The mathematician ran to the store.
* The physicist ran to the store.
* The mathematician solved the open problem.

现在假设我们得到了一个我们训练数据中前所未见的新句子:

* The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn't it be much
better if we could use the following two facts:

我们的语言模型在这句话上可能会行得通，但是如果我们能够使用以下两个示例则会差得多:

* We have seen  mathematician and physicist in the same role in a sentence. Somehow they
  have a semantic relation.
* We have seen mathematician in the same role  in this new unseen sentence
  as we are now seeing physicist.

然后推断物理学家实际上很适合新的未见过的句子吗？这就是我们所说的相似概念 :
我们的意思是 *semantic similarity* ,而不仅仅是具有相似的字形表示.
这是一种通过连接我们所看到的和我们所没有的之间的点来对抗语言数据的稀疏性的技术.
这个例子依赖于一个基本的语言学假设 : 出现在相似语境中的词在语义上彼此相关.
被称为`distributional hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__.


获取密集词嵌入
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们如何解决这个问题? 也就是说，我们如何才能真正编码语义相似性呢？ 也许我们想
出了一些语义属性.例如，我们看到数学家和物理学家都可以运行，所以也许我们会给这些
词赋予 "is able to run" 语义属性的高分。想想其他一些属性，想象一下你可能在这些
属性上得到一些常用词。如果每个属性都是一个维度，那么我们可能会给每个单词一个向量，
如下所示:

.. math::

    q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
   \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]

.. math::

    q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
   \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]

然后我们可以通过这样做来获得这些词的相似度:

.. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}

尽管通过长度进行归一化更为常见:

.. math::

    \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
   {\| q_\text{\physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)

其中 :math:`\phi` 是两个向量之间的角度. 这样,及其相似的词 (词嵌入指向相同的方向) 
相似度为 1. 完全相反的词相似度为 -1.

你可以从本节开头考虑稀疏的 one-hot 编码 , 作为我们已经定义的这些新矢量的一个特例,
其中每个单词基本上具有相似性0 , 并且我们给每个单词一些独特的语义属性. 
这些新的向量是* dense *，这就是说它们的入口（通常）是非零的。


但是这些新的向量是一个巨大的痛苦 : 你可以想象成千上万个可能与确定相似性有关
的不同语义属性 , 以及如何设置不同属性的值?深度学习理念的核心是神经网络学习特
征的表征，而不是要求程序员亲自设计它们.那么为什么不让这个词嵌入在我们的模型
中作为参数，然后在训练中更新?这正是我们要做的.我们将拥有一些原则上网络可以学
习的 *latent semantic attributes*.

注意，词嵌入无法解释.也就是说,虽然用上面的手工制作的向量,我们可以看到数学家和
物理学家的相似之处在于,他们都喜欢咖啡,如果我们允许神经网络学习嵌入,并且看到数
学家和物理学家都有很大的价值在第二个方面，目前尚不清楚这意味着什么 . 它们在一
些潜在的语义维度上是相似的，但是这可能对我们没有任何解释.

总之，**单词嵌入是一个单词的 * 语义 * 的表示，有效地编码可能与手头任务** 相关的语义
信息.也可以嵌入其他的东西 : 词性标签,解析树,任何东西! 特征嵌入的想法是该领域的核心.


Pytorch中的词嵌入
~~~~~~~~~~~~~~~~~~~~~~~~~~

在我们开始一个工作的例子和练习之前，先简单介绍一下如何在Pytorch中使用嵌
入以及一般的深度学习编程。类似于我们在制作一个one-hot编码时为每个单词定
义唯一索引的方式，我们还需要在使用嵌入时为每个单词定义一个索引.
这些将成为查找表的关键.也就是说，词嵌入存储为 :math:`|V| \times D` 矩阵，
其中 :math:`D` 是嵌入的维数,使得赋予索引 :math:`i` 的单词将其嵌入存储在
矩阵的第 :math:`i` 行中.

在我所有的代码中, 从单词到索引的映射是一个名为 word\_to\_ix 的字典.
允许使用嵌入的模块是torch.nn.Embedding，它有两个参数：词汇大小和嵌入的维数.
要索引到此表中，您必须使用torch.LongTensor（因为索引是整数，而不是浮点数）.

"""

# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

######################################################################

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))
print(hello_embed)


######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


######################################################################
# Exercise: Computing Word Embeddings: Continuous Bag-of-Words
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Continuous Bag-of-Words model (CBOW) is frequently used in NLP deep
# learning. It is a model that tries to predict words given the context of
# a few words before and a few words after the target word. This is
# distinct from language modeling, since CBOW is not sequential and does
# not have to be probabilistic. Typcially, CBOW is used to quickly train
# word embeddings, and these embeddings are used to initialize the
# embeddings of some more complicated model. Usually, this is referred to
# as *pretraining embeddings*. It almost always helps performance a couple
# of percent.
#
# The CBOW model is as follows. Given a target word :math:`w_i` and an
# :math:`N` context window on each side, :math:`w_{i-1}, \dots, w_{i-N}`
# and :math:`w_{i+1}, \dots, w_{i+N}`, referring to all context words
# collectively as :math:`C`, CBOW tries to minimize
#
# .. math::  -\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
#
# where :math:`q_w` is the embedding for word :math:`w`.
#
# Implement this model in Pytorch by filling in the class below. Some
# tips:
#
# * Think about which parameters you need to define.
# * Make sure you know what shape each operation expects. Use .view() if you need to
#   reshape.
#

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


make_context_vector(data[0][0], word_to_ix)  # example
