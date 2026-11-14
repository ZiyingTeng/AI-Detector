# P7 Part A – Video script (about 1 minute)

【第 1 页 Method：开始录制，念下面这段】
Hi, I am Yang Jiayue. I built Detector A. It reads writing style, not meaning. We cut every text to the first 60 words, so length cannot help. Then we compute 45 simple numbers, like sentence length, punctuation, and common small words. An XGBoost model turns them into the chance that the text is AI. We train on HC3 only.

【👉 翻到第 2 页 Results】
On HC3, it works very well: AUROC is 0.99. But on MAGE and RAID, AUROC falls to 0.56 and 0.66, close to guessing. On the new generators it is 0.76, and after paraphrasing it is 0.73. At a strict 1 percent false positive rate, it finds only 2 to 4 percent of AI texts.

【👉 翻到第 3 页 Why it fails】
In HC3, human answers are short and informal, and the model learns this habit. Other datasets look different, so the habit stops working. Style features are fast and easy to explain, but they should be used together with other detectors. Thank you.

【录制结束】
