# P7 Part A – Video script (about 1 minute)

## Slide 1 · Method
Detector A reads writing style, not meaning. We cut every text to its first 60 words, so length cannot help. Then we compute 45 simple numbers, like sentence length, punctuation and common small words. An XGBoost model turns them into the chance that the text is AI. We train on HC3 only.

## Slide 2 · Results
On HC3 it works very well: AUROC 0.99. But on MAGE and RAID, AUROC falls to 0.56 and 0.66, close to guessing. On the new generators it is 0.76. At 1 percent false positives, it finds only 3 to 4 percent of AI texts.

## Slide 3 · Why it fails
In HC3, humans write short, informal answers, and the model learns that habit. Other datasets look different, so the habit stops working. Style features are fast and easy to explain, but they should be combined with other detectors.

---
参考中文：A 检测器看的是文风而不是语义……（HC3 内 0.99，换数据集接近随机，建议和其他检测器配合）。
