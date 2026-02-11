# APTOS 2019 Blindness Detection

## Source
- **Website**: [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

## Instructions
1. **Install Kaggle CLI**: `pip install kaggle`
2. **Setup API Token**: Download `kaggle.json` from your Kaggle account (Settings -> API -> Create New API Token) and place it in `~/.kaggle/` (`C:\Users\<User>\.kaggle\` on Windows).
3. **Run Download Script**:
   ```bash
   bash scripts/download_aptos.sh
   ```

## Expected Structure
```text
datasets/raw/aptos2019/
├── train_images/
├── test_images/
├── train.csv
└── test.csv
```
