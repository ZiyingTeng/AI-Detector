# AI Text Detector (EE6405 P7)

检测一段文字是人写的还是 AI 生成的，比较四类检测器，并测试改写攻击。

## 运行

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # 填入 API key
streamlit run app.py
```

还没上传 `predict` 文件的检测器会显示为 (mock)，返回随机分数，界面照常运行。

## 每人上传什么

| 成员 | 上传到仓库 | 说明 |
|---|---|---|
| A | `P7_A_predict.py`、`models/P7_A_model.pkl` | 风格特征检测器 |
| B | `P7_B_predict.py`、`models/P7_B_calib.pkl` | GPT-2 自动下载，不用上传 |
| C | `P7_C_predict.py` | 模型上传到 Hugging Face（超过 GitHub 100 MB 上限） |
| D | `P7_D_predict.py`、`P7_demo_texts.csv` | 不要上传 API key |
| E | `results/auroc.csv`、`results/attack.csv`、`results/attack_pairs.csv`、`results/findings.txt` | 前端只读取、画图 |
| 所有人 | `notebooks/P7_X.ipynb`、`requirements/P7_X.txt` | X 换成自己的字母 |

`predict()` 统一格式：

```python
def predict(text: str) -> dict:
    return {"prob_ai": 0.83, "label": "AI", "extra": {}}
```

## 上传规则

- 单个文件不能超过 100 MB。
- 视频、录音和幻灯直接发给 F，不放进仓库。

## 结构

```
app.py                 检测页（F）
pages/1_Results.py     实验结果页（F，读取 results/）
pages/2_Attack_Demo.py 攻击对比页（F，读取 results/attack_pairs.csv）
detectors.py           加载 A–D 的 predict()，缺失时用 mock
gltr.py                GPT-2 逐词排名着色
P7_A..D_predict.py     各检测器（A–D）
models/                A、B 的模型文件
results/               E 的结果文件
notebooks/             每人的实验 notebook
requirements/          每人的依赖清单
```
