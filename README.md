# Turning Unstructured Data into Structured Data

---

### 👥 **Team Members**


| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Vy Dang    | @dtv9307 | Data preprocessing, Feature extraction, prompt design     |
| Charles Francisco   | @Charles-Francisco   | Prompt design, feature extraction logic, end-to-end pipeline refinement |
|Nishu Shrestha     | @nishu8343  | Prompt design, feature extraction          |
|Samay Bhojwani    | @Samay30 | Prompt engineering, feature extraction, Data preprocessing  |


---

## 🎯 **Project Highlights**

- Developed a machine learning model using `GPT-4o` to address structured data getting buried under those unstructured conversations.
 - Achieved >70% extraction accuracy on high-yield features.

---
## 👩🏽‍💻 **Setup and Installation**

* Clone Repository: git clone https://github.com/Samay30/ASAPP-Fall-AI-Studio-Project
* Required Dependencies:
Python 3.9+
Jupyter Notebook or VS Code + Jupyter extension
Pandas
* Open project
* Switch to ASAPP-1C-Presentation branch
* Run the code cells on ‘Generate_Structured_Data.ipynb’ in order to reproduce the results 

---

## 🏗️ **Project Overview**

**CORE CHALLENGE**
*Turn unstructured data (conversations between two humans) into structured data with LLMs.

**GOALS**
* Achieve >70% extraction accuracy on high-yield features.
* Build a robust, reproducible, and well-documented extraction pipeline.
* Optimize prompt design to maximize extraction accuracy. 

**BUSINESS IMPACT**
* Converts agent-customer conversations into structured, analyzable data.
* Identifies patterns, supports automation, and improves response quality
* Enables other teams to make data-driven decisions based on real customer data.
* Turns contact centers into a valuable source of ideas and insights. 

---

## 📊 **Data Exploration**

**Dataset Overview**

* 10K+ annotated customer support chats in JSON format.
* Covers returns, refunds, promo, account inquiries, policy enforcement.
* Includes both original and delexicalized (replaces sensitive info with placeholders) dialogues.
* Split into training, dev, test sets.
* Provided with guidelines and ontology files.

**Exploratory Data Analysis**
* Inspected columns and sample rows to understand feature types and available attributes.
* Built a clean dataframe combining customer info, product/order details, and scenario metadata.
* Explored key scenarios (e.g., product defect, returns).
* Flattened nested scenario and metadata fields (e.g., flow, memberlevel, payment method).
* Identified high-value features and sample conversation patterns.

---

## 🧠 **Model Development**

*This project used prompt engineering with the OpenAI API (GPT-4.5) rather than training a traditional model. The LLM was guided to perform the task through carefully designed prompts that structured inputs and constrained outputs.

*Feature selection was handled through prompt design by emphasizing different subsets of information and comparing their impact on performance. Prompt structure and clarity were iteratively refined to improve consistency and accuracy.

*The dataset was split into train, dev, test sets. Accuracy metrics were used for evaluation, with simpler prompts serving as the baseline.

---

## 📈 **Results & Key Findings**

*Most feature sets resulted in accuracy scores in the 60–70% range, indicating that the model captures meaningful patterns in the data but still leaves room for improvement. 

*Experiments with more advanced large language model–based features showed noticeably better accuracy 
---

## 🚀 **Next Steps**

* Try more advanced LLMs, try to get >80% accuracy for all features.
* Consider adding extra pre-processing for underperforming features.
* Try PII detection.

---

## 🙏 **Acknowledgements** 
* Big thanks to our Coach Aram Ramos for always supporting us and stepping in when we need help!
