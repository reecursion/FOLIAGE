# Pairwise Persuasion Datasets

This directory contains 15 pairwise datasets for training binary classifiers on persuasion categories.

## Categories

1. **expressed_donate_did**: Expressed intention of donating but did not donate
2. **expressed_donate_donated**: Expressed intention and donated
3. **no_express_donated**: Did not express intention but donated
4. **no_express_no_donate**: Did not express intention and did not donate
5. **unclear_donated**: Unclear but donated
6. **unclear_no_donate**: Unclear but did not donate

## Dataset Structure

Each CSV file contains:
- All original columns from the main dataset
- `binary_label`: 0 for first category, 1 for second category
- `category`: The specific category name

## Pairwise Combinations

### pairwise_01_expresseddonatedid_vs_expresseddonatedonated.csv
- **Category 1 (Label 0)**: Expressed intention of donating but did not donate
  - Conversations: 78
  - Utterances: 2714
- **Category 2 (Label 1)**: Expressed intention and donated
  - Conversations: 64
  - Utterances: 2434
- **Balance Ratio**: 0.897
- **Total**: 142 conversations, 5148 utterances

### pairwise_02_expresseddonatedid_vs_noexpressdonated.csv
- **Category 1 (Label 0)**: Expressed intention of donating but did not donate
  - Conversations: 78
  - Utterances: 2714
- **Category 2 (Label 1)**: Did not express intention but donated
  - Conversations: 4
  - Utterances: 164
- **Balance Ratio**: 0.060
- **Total**: 82 conversations, 2878 utterances

### pairwise_03_expresseddonatedid_vs_noexpressnodonate.csv
- **Category 1 (Label 0)**: Expressed intention of donating but did not donate
  - Conversations: 78
  - Utterances: 2714
- **Category 2 (Label 1)**: Did not express intention and did not donate
  - Conversations: 33
  - Utterances: 1171
- **Balance Ratio**: 0.431
- **Total**: 111 conversations, 3885 utterances

### pairwise_04_expresseddonatedid_vs_uncleardonated.csv
- **Category 1 (Label 0)**: Expressed intention of donating but did not donate
  - Conversations: 78
  - Utterances: 2714
- **Category 2 (Label 1)**: Unclear but donated
  - Conversations: 3
  - Utterances: 114
- **Balance Ratio**: 0.042
- **Total**: 81 conversations, 2828 utterances

### pairwise_05_expresseddonatedid_vs_unclearnodonate.csv
- **Category 1 (Label 0)**: Expressed intention of donating but did not donate
  - Conversations: 78
  - Utterances: 2714
- **Category 2 (Label 1)**: Unclear but did not donate
  - Conversations: 22
  - Utterances: 682
- **Balance Ratio**: 0.251
- **Total**: 100 conversations, 3396 utterances

### pairwise_06_expresseddonatedonated_vs_noexpressdonated.csv
- **Category 1 (Label 0)**: Expressed intention and donated
  - Conversations: 64
  - Utterances: 2434
- **Category 2 (Label 1)**: Did not express intention but donated
  - Conversations: 4
  - Utterances: 164
- **Balance Ratio**: 0.067
- **Total**: 68 conversations, 2598 utterances

### pairwise_07_expresseddonatedonated_vs_noexpressnodonate.csv
- **Category 1 (Label 0)**: Expressed intention and donated
  - Conversations: 64
  - Utterances: 2434
- **Category 2 (Label 1)**: Did not express intention and did not donate
  - Conversations: 33
  - Utterances: 1171
- **Balance Ratio**: 0.481
- **Total**: 97 conversations, 3605 utterances

### pairwise_08_expresseddonatedonated_vs_uncleardonated.csv
- **Category 1 (Label 0)**: Expressed intention and donated
  - Conversations: 64
  - Utterances: 2434
- **Category 2 (Label 1)**: Unclear but donated
  - Conversations: 3
  - Utterances: 114
- **Balance Ratio**: 0.047
- **Total**: 67 conversations, 2548 utterances

### pairwise_09_expresseddonatedonated_vs_unclearnodonate.csv
- **Category 1 (Label 0)**: Expressed intention and donated
  - Conversations: 64
  - Utterances: 2434
- **Category 2 (Label 1)**: Unclear but did not donate
  - Conversations: 22
  - Utterances: 682
- **Balance Ratio**: 0.280
- **Total**: 86 conversations, 3116 utterances

### pairwise_10_noexpressdonated_vs_noexpressnodonate.csv
- **Category 1 (Label 0)**: Did not express intention but donated
  - Conversations: 4
  - Utterances: 164
- **Category 2 (Label 1)**: Did not express intention and did not donate
  - Conversations: 33
  - Utterances: 1171
- **Balance Ratio**: 0.140
- **Total**: 37 conversations, 1335 utterances

### pairwise_11_noexpressdonated_vs_uncleardonated.csv
- **Category 1 (Label 0)**: Did not express intention but donated
  - Conversations: 4
  - Utterances: 164
- **Category 2 (Label 1)**: Unclear but donated
  - Conversations: 3
  - Utterances: 114
- **Balance Ratio**: 0.695
- **Total**: 7 conversations, 278 utterances

### pairwise_12_noexpressdonated_vs_unclearnodonate.csv
- **Category 1 (Label 0)**: Did not express intention but donated
  - Conversations: 4
  - Utterances: 164
- **Category 2 (Label 1)**: Unclear but did not donate
  - Conversations: 22
  - Utterances: 682
- **Balance Ratio**: 0.240
- **Total**: 26 conversations, 846 utterances

### pairwise_13_noexpressnodonate_vs_uncleardonated.csv
- **Category 1 (Label 0)**: Did not express intention and did not donate
  - Conversations: 33
  - Utterances: 1171
- **Category 2 (Label 1)**: Unclear but donated
  - Conversations: 3
  - Utterances: 114
- **Balance Ratio**: 0.097
- **Total**: 36 conversations, 1285 utterances

### pairwise_14_noexpressnodonate_vs_unclearnodonate.csv
- **Category 1 (Label 0)**: Did not express intention and did not donate
  - Conversations: 33
  - Utterances: 1171
- **Category 2 (Label 1)**: Unclear but did not donate
  - Conversations: 22
  - Utterances: 682
- **Balance Ratio**: 0.582
- **Total**: 55 conversations, 1853 utterances

### pairwise_15_uncleardonated_vs_unclearnodonate.csv
- **Category 1 (Label 0)**: Unclear but donated
  - Conversations: 3
  - Utterances: 114
- **Category 2 (Label 1)**: Unclear but did not donate
  - Conversations: 22
  - Utterances: 682
- **Balance Ratio**: 0.167
- **Total**: 25 conversations, 796 utterances

## Usage

Each dataset can be used to train a binary classifier to distinguish between two specific persuasion behavior patterns. The binary_label column serves as the target variable.

## Files

- `pairwise_*.csv`: Individual pairwise datasets
- `pairwise_datasets_summary.csv`: Summary statistics for all datasets
- `README.md`: This documentation
