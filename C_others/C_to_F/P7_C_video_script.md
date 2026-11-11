# P7 · Part C 视频 / 录音文案（约 1 分钟）

**讲者**：YU ZIXUAN（G2605146C）　**对应幻灯片**：`P7_C_slides.pptx`（3 页，每页备注栏有同样的英文稿）
**长度**：英文约 145 词，正常语速约 1 分钟。

---

## Slide 1 · Theory（约 20 秒）

> Part C is a supervised detector. I fine-tuned RoBERTa-base end to end: each text is cut to 256 tokens, the start-token vector goes into a linear head, and softmax gives the probability that the text is AI-written. I trained one model each on HC3, MAGE and RAID, plus a main model on all three.

*中文参考*：C 部分是监督式检测器。我对 RoBERTa-base 做端到端微调：文本截断到 256 个 token，起始 token 的向量接线性层，softmax 输出 AI 概率。HC3、MAGE、RAID 各训一个模型，再用三者合并训练一个主模型。

---

## Slide 2 · Results（约 20 秒）

> The main model scores about 1.00 AUROC on HC3 and RAID, but only 0.90 on MAGE, which has 27 generators. The heatmap shows why: a model trained on HC3 or RAID falls to around 0.6 on MAGE. Fine-tuning learns the generators it sees, so training data diversity matters most.

*中文参考*：主模型在 HC3 和 RAID 上 AUROC 约为 1.00，但在含 27 个生成器的 MAGE 上只有 0.90。热力图说明了原因：只用 HC3 或 RAID 训练的模型在 MAGE 上掉到 0.6 左右。微调学到的是见过的生成器，所以训练数据的多样性最重要。

---

## Slide 3 · Where it fails（约 20 秒）

> Attacks break it in both directions. Extra line breaks or zero-width spaces get every human text flagged as AI, while homoglyphs let sixty-one percent of AI texts slip through. So a score from this detector is evidence, never a verdict.

*中文参考*：攻击会让它在两个方向上出错：多加换行或插入零宽字符，所有人类文本都会被判成 AI；同形字替换则让 61% 的 AI 文本漏检。所以这个检测器的分数只能作为参考证据，不能当作结论。

---

## 录制提示

- 直接用 PowerPoint 的「幻灯片放映 → 录制」边翻页边配音，导出为视频；只交录音也可以（要求是视频或录音）。
- 第 2 页说到 "around 0.6 on MAGE" 时，可以用鼠标指一下热力图里 MAGE 那一列。
