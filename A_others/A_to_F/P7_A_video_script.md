# P7 Part A – Video script (about 1 minute)

【第 1 页：方法】
Detector A reads writing style, not meaning. We keep only the first 60 words of each text, so length cannot give the answer away. From these words we compute 45 simple numbers, such as sentence length, punctuation, and common small words. An XGBoost model turns them into the chance that the text is AI. We train on HC3 only.

【翻到第 2 页：结果】
On HC3, it works very well: AUROC is 0.99. But on MAGE and RAID, AUROC falls to 0.56 and 0.66, close to guessing. On the new generators it is 0.76, and after paraphrasing it is 0.73. At a strict 1 percent false positive rate, it finds only 2 to 4 percent of AI texts.

【翻到第 3 页：失败原因】
Why does it fail? In HC3, human answers are short and informal, and the model learns this habit. Other datasets look different, so the habit stops working. Style features are fast and easy to explain, but they should be used together with other detectors.
