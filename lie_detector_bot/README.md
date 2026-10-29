# 🤖 Bot Lie Detection & Hallucination Auditor

A Python tool that analyzes chatbot conversation results to detect potential lying/hallucination by comparing bot answers against company policy documents using semantic similarity.

## 📋 Requirements

- **Python 3.8+**
- **Dependencies:**
  ```bash
  pip install pandas scikit-learn sentence-transformers
  ```

## 📁 Project Structure

```
lie_detector_bot/
├── app.py                  # Main bot script
├── conversations.csv       # Chatbot conversation data
├── policies/               # Company policy documents
└── lie_analysis_results.csv # Generated analysis report
```

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
pip install pandas scikit-learn sentence-transformers
```

### **2. Place Your Data**

```bash
lie_detector_bot/
├── conversations.csv       # Your chatbot data
└── policies/               # Company policy docs
    ├── policy1.txt
    ├── policy2.txt
    └── ...
```

**conversations.csv format:**
```csv
id,question,answer
1,What is your return policy?,You can return items within 30 days with original receipt.
2,How do I reset my password?,Click on Forgot Password on the login page.
```

### **3. Run the Bot**

```bash
cd lie_detector_bot
python3 app.py
```

## 📖 How It Works

1. **Loads** your conversation CSV (`id`, `question`, `answer`)
2. **Loads** company policy documents from the `policies/` folder
3. **Encodes** both answers and policies into vector embeddings using `SentenceTransformer`
4. **Compares** each answer against policy documents using cosine similarity
5. **Flags** answers as:
   - ✅ **Truthful** — High similarity to policies (score ≥ threshold)
   - ⚠️ **Lying Risk** — Low similarity (score < threshold)
6. **Generates** a detailed report with scores and statistics

## 📊 Output Format

### **Console Summary**

```
📊 LIE DETECTION ANALYSIS SUMMARY
============================================================
📈 Total conversation turns analyzed: 57
✅ Likely truthful: 20 (35.1%)
⚠️ Potentially lying: 37 (64.9%)
❓ Unknown (no policies): 0 (0.0%)

⚠️ 37 turns potentially flagged as lying:
  - Turn ID: 16 | Question: What is your return policy?... | Answer: You can return items within 7 days.... | Score: 0.19 | Flag: ⚠️ Lying Risk
  - Turn ID: 18 | Question: What is the warranty period?... | Answer: All products have a 1-year warranty.... | Score: 0.30 | Flag: ⚠️ Lying Risk

============================================================
💾 Results saved to: lie_analysis_results.csv
```

### **lie_analysis_results.csv**

| Column | Description |
|--------|-------------|
| `id` | Conversation ID |
| `question` | User question |
| `answer` | Bot response |
| `similarity_score` | Semantic similarity to policies (0-1) |
| `lying_flag` | "✅ Truthful" or "⚠️ Lying Risk" |
| `score_description` | Brief explanation of score |

## 🔧 Configuration

### **Adjusting Threshold**

The `similarity_threshold` controls sensitivity:

| Threshold | Sensitivity | Use Case |
|-----------|------------|----------|
| **0.50** | Very strict | Only obvious lies |
| **0.30** | Balanced (default) | Most use cases |
| **0.20** | More sensitive | Catch subtle lies |
| **0.15** | Highly sensitive | Maximum detection |

**Where to change:**
```python
# In app.py
results = detector.analyze(threshold=0.30)
```

### **Recommended Thresholds**

| Goal | Threshold |
|-----------|------------|
| High precision (avoid false alarms) | 0.40-0.50 |
| Balanced (default) | 0.30 |
| High recall (catch everything) | 0.20-0.25 |

## 📈 Interpreting Results

| Score Range | Meaning | Action |
|-------------|---------|--------|
| **0.8 - 1.0** | Very similar to policy | ✅ Truthful |
| **0.6 - 0.8** | Somewhat similar | 🟡 Review |
| **0.4 - 0.6** | Moderately similar | 🟡 Likely truthful |
| **0.2 - 0.4** | Low similarity | ⚠ Lying risk |
| **0.0 - 0.2** | Very different | 🔴 Likely lie |

## 🛡️ Best Practices

1. Add comprehensive policy documents covering all business domains
2. Regularly update policies with new company rules
3. Review flagged answers manually and add to policy documents
4. Tune threshold based on your precision/recall needs
5. Log and review flagged results weekly
6. Combine with human review for final decisions

## 🔧 Troubleshooting

### **Import Errors**

```bash
pip install pandas scikit-learn sentence-transformers
```

### **Policy Folder Not Found**

```bash
mkdir -p policies
echo "Your policy text here" > policies/policy1.txt
```

### **CSV Format Issues**

Ensure your CSV has proper headers:
```csv
id,question,answer
1,What is your return policy?,You can return items within 30 days with original receipt.
```

## 📈 Expected Output

| Metric | Typical Value |
|----------|---------------|
| **Truthful Rate** | 30-50% |
| **Lying Risk Rate** | 30-50% |
| **False Positives** | 5-15% |
| **False Negatives** | 5-10% |

## 🛡️ Bot Improvement Recommendations

1. Add "I don't know" fallback for low-confidence answers
2. Require citations in bot responses
3. Regularly review flagged answers and update policies
4. Train bot on policy-based answers
5. Monitor responses weekly for new hallucination patterns

## 📄 License

For internal use and quality assurance purposes.
