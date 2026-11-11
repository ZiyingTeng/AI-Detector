# P7-B Video Script | 中英文演讲稿

## Slide 1: Method

**English**

Hi everyone, my part focuses on AI text detection using GPT-2 and GLTR.

I extract PPL and GLTR features and use Logistic Regression for classification.

Now, let's look at these features.

**中文翻译**

大家好，我负责的部分主要研究利用 GPT-2 和 GLTR 进行 AI 文本检测。我提取 PPL 和 GLTR 特征，并使用逻辑回归进行分类。接下来，让我们看看这两类特征。

## Slide 2: Features

**English**

PPL measures how predictable a text is. Lower PPL means higher predictability.

GLTR measures token rank distributions.

The left chart shows that **82.9%** of AI tokens are in the top 10, compared to **66.3%** for human text.

The right figure shows token ranks using different colors.

So, do these features work better together?

**中文翻译**

PPL 用来衡量一段文本的可预测性。PPL 越低，说明文本越容易被预测。GLTR 则衡量词元的预测排名分布。左侧图表显示，AI 文本中有 **82.9%** 的词元位于预测排名前十，而人类文本的这一比例是 **66.3%**。右侧图示通过不同颜色展示词元的排名。那么，将这两类特征结合起来，效果会不会更好呢？

## Slide 3: Ablation Study

**English**

To find out, I compared PPL, GLTR, and their combination.

The combined features achieved the best results, with an **AUROC of 0.9722** and an **F1 of 0.9444**.

But how well does the model work on other datasets?

**中文翻译**

为了验证这一点，我比较了单独使用 PPL、单独使用 GLTR，以及将两者结合时的效果。结果表明，结合两类特征的表现最好，**AUROC 达到 0.9722**，**F1 达到 0.9444**。不过，这个模型在其他数据集上的表现如何呢？

## Slide 4: Generalization

**English**

To test this, I trained the model on HC3 and tested it on MAGE and RAID.

AUROC dropped from **0.9622** on HC3 to **0.6444** and **0.7500**.

This shows that combining features improves detection, but generalization remains a challenge.

Thank you!

**中文翻译**

为了测试泛化能力，我在 HC3 数据集上训练模型，并在 MAGE 和 RAID 数据集上进行测试。AUROC 从 HC3 上的 **0.9622**，分别下降到 **0.6444** 和 **0.7500**。这说明结合两类特征能够提高检测效果，但模型的泛化能力仍然是一个挑战。谢谢大家！
