# AI Hiring Model Bias Audit
### Evaluating Fairness and Reliability in an AI Attrition Prediction Model
**Tools:** Python · scikit-learn · SQL (SQLite) · Power BI · pandas · seaborn · matplotlib

---

##  Project Overview

AI-powered hiring and retention tools are increasingly common in HR departments. But how fair and reliable are they — really?

This project audits an AI attrition prediction model trained on IBM HR Analytics data, evaluating whether it produces **accurate and unbiased outputs** across gender, age, education level, and department.

The short answer: it doesn't.

The model achieves **85% overall accuracy** by predicting that most people stay. But it misses up to **97.1% of employees who actually leave** in some demographic groups, and systematically over-flags experienced, highly-qualified women as flight risks when they are among the most stable employees in the dataset.

This is not a model you would want making hiring or retention decisions.

---

##  The Business Question

> *Can an AI recruitment screening model be trusted to evaluate attrition risk fairly and reliably across gender, age, and educational background — and where does it fail?*

---

##  Dataset

- **Source:** [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)
- **Size:** 1,470 employees, 35 features
- **Target variable:** Attrition (Yes/No)
- **Overall attrition rate:** 16.1% (237 employees left)
- **Data quality:** Zero missing values, zero duplicates

**Key features used:**
- Demographics: Age, Gender, Education
- Tenure: YearsAtCompany, YearsInCurrentRole, TotalWorkingYears
- Compensation: MonthlyIncome, PercentSalaryHike
- Satisfaction: JobSatisfaction, WorkLifeBalance, EnvironmentSatisfaction
- Other: OverTime, DistanceFromHome, NumCompaniesWorked

---

##  Methodology

The project runs in four sessions:

```
Session 1          Session 2          Session 3          Session 4
──────────         ──────────         ──────────         ──────────
Data Loading   →   Model Training →   Bias Audit     →   Dashboard +
Cleaning           & Evaluation       Demographic        Executive
EDA                                   Intersectional     Brief
SQL Setup                             Analysis
```

### Session 1 — Data Loading, Cleaning & EDA
- Loaded and audited dataset for quality issues
- Dropped constant-value columns (EmployeeCount, Over18, StandardHours)
- Engineered demographic features: Age_Group bins, Education_Label mapping, Gender_Binary
- Stored clean dataset in SQLite database
- Ran SQL queries to surface attrition patterns by department and gender

### Session 2 — Model Training & Evaluation
- Selected 16 features representing the kind of data a real AI hiring tool might use
- Trained two models: Logistic Regression and Random Forest (sklearn)
- Evaluated on 80/20 train/test split, stratified by target
- Generated individual AI risk scores and binary flags for all 1,470 employees
- Stored scored dataset back to SQLite for bias audit

### Session 3 — Bias Audit
- Built a reusable bias_audit() function calculating false positive rate, false negative rate, and average AI risk score per demographic group
- Ran audit across four dimensions: Gender, Age Group, Education Level, Department
- Conducted intersectional analysis combining Gender × Age × Education
- Calculated Bias Gap (AI score minus actual attrition rate) to identify over and under-penalised groups

### Session 4 — Output
- Exported 7 structured CSV files for Power BI
- Built 5-page Power BI dashboard
- Documented AI workflow (where it helped, where I overrode it)
- Produced executive brief with 6 governance recommendations

---

## 📊 Key Findings

### Finding 1 — The Model Fails Where It Matters Most

| Age Group | Actual Attrition | AI Risk Score | False Negative Rate |
|---|---|---|---|
| 18–25 | 34.8% | 31.6% | 67.5% |
| 26–35 | 19.1% | 19.6% | 72.4% |
| 36–45 | 9.2% | 13.2% | 83.7% |
| **46–60** | **12.5%** | **6.5%** | **97.1%** |

The AI misses **97.1% of employees aged 46–60 who actually leave**. For senior employees, the model provides virtually no warning before they resign. An HR team trusting this tool would be operating blind for their most experienced workforce segment.

---

### Finding 2 — Compounded Age Bias

Age carries **10.8% of direct model weight** — but age-correlated proxy features amplify this significantly:

| Feature | Model Weight |
|---|---|
| Monthly Income | 13.0% |
| **Age** | **10.8%** |
| Total Working Years | 9.0% |
| Distance From Home | 8.0% |
| Years at Company | 7.0% |

Total Working Years and Years at Company are both **proxies for age**. Combined with the direct Age feature, age-related signals drive **26.8% of all model decisions** — constituting compounded age discrimination risk under **EU AI Act Article 10** requirements on training data fairness.

---

### Finding 3 — The PhD Paradox

> *The most qualified, experienced women in the dataset are being systematically over-flagged as flight risks.*

| Group | Actual Attrition | AI Risk Score | Bias Gap |
|---|---|---|---|
| Female · 36–45 · Doctor | **0.0%** | **15.6%** | **+15.6 pts** |
| Female · 36–45 · Master | 6.0% | 13.5% | +7.5 pts |
| Female · 36–45 · Below College | 0.0% | 7.1% | +7.1 pts |

Female employees aged 36–45 with doctoral qualifications have **zero actual attrition** yet receive the highest over-penalisation of any group in the dataset. If deployed in a hiring context, this model would systematically disadvantage experienced, highly-qualified women in mid-career — the exact demographic most organisations claim to want to retain.

---

### Finding 4 — The Young Female Graduate Blind Spot

> *The AI most severely underestimates risk for the highest-risk group.*

| Group | Actual Attrition | AI Risk Score | Bias Gap |
|---|---|---|---|
| Female · 18–25 · Bachelor | **50.0%** | **33.1%** | **-16.9 pts** |
| Male · 46–60 · College | 26.9% | 10.3% | -16.6 pts |
| Female · 18–25 · Below College | 46.2% | 31.6% | -14.6 pts |

Half of young female Bachelor graduates in this dataset left the company. The AI scored them at only 33.1% — a 16.9-point underestimation. These employees are simultaneously among the highest actual flight risks and the least accurately flagged by the model.

---

### Finding 5 — The Department Blind Spot

| Department | Actual Attrition | AI Risk Score | Gap |
|---|---|---|---|
| Sales | **20.6%** | **15.5%** | -5.1 pts |
| Human Resources | 19.0% | 16.2% | -2.8 pts |
| Research & Development | 13.8% | 16.4% | +2.6 pts |

The AI assigns its **lowest risk score to the highest-attrition department**. Sales employees are leaving at the highest rate yet receive the least warning from the model. R&D employees — the most stable department — receive the highest risk scores.

---

##  Where AI Helped — And Where It Didn't

### Where Claude/ChatGPT Helped
- Suggested initial structure for the `bias_audit()` function — accepted and adapted
- Accelerated boilerplate code for visualisation grids — accepted
- Helped debug a SQLite GROUP BY error in Session 1 — accepted

### Where I Overrode AI Suggestions
1. **Model selection for the audit:** AI recommended Random Forest as the primary model due to higher raw accuracy. I chose Logistic Regression because **interpretability matters more than accuracy in a bias audit**. A model you can explain is more trustworthy than one you can't — especially for HR governance.

2. **Framing of 85% accuracy:** AI initially described this as "strong model performance." I rejected this framing. 85% accuracy with a 77–91% false negative rate on the minority class is not strong performance — it is a model that **fails precisely where it matters most**.

3. **Compounded age proxy finding:** AI missed the connection between TotalWorkingYears, YearsAtCompany, and Age as compounding proxies. I identified and documented this independently after reviewing feature importance scores.

4. **Feature exclusions:** AI suggested including MaritalStatus as a feature. I excluded it as a **protected characteristic** that should not feed a hiring or retention model regardless of its predictive value.

> AI accelerated the technical build by approximately 40%. The analytical judgements — what the findings mean, which model to trust, where the real bias lies, and what the business should do — required human reasoning that the AI could not supply independently. This is precisely the skill an AI/ML Analyst provides.

---

##  Recommendations

Based on the audit findings, the following governance interventions are recommended before any deployment of this model in hiring or retention contexts:

1. **Do not deploy in current form** — the model's false negative rates make it unreliable for HR decision-making across all age groups

2. **Remove age-proxy features** — retrain the model after removing TotalWorkingYears and YearsAtCompany to reduce compounded age bias

3. **Add fairness constraints** — implement equalised odds constraints across age groups during model training

4. **Audit quarterly** — track false negative rates by demographic group after every model update, not just overall accuracy

5. **Require human override** — no automated screening or retention decision should be made without HR manager review of AI outputs

6. **Consider department-specific models** — a single model across all departments performs poorly for Sales (highest attrition, lowest AI score); role-specific models would outperform the one-size-fits-all approach

---

##  Repository Structure

```
ai-hiring-bias-audit/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Source dataset (Kaggle)
│
├── notebooks/
│   ├── session1_eda.ipynb                         # Data loading, cleaning, EDA, SQL
│   ├── session2_model.ipynb                       # Model training & evaluation
│   ├── session3_bias_audit.ipynb                  # Demographic & intersectional audit
│   └── session4_output.ipynb                      # Export & documentation
│
├── outputs/
│   ├── powerbi_employees.csv                      # Individual AI scores
│   ├── powerbi_gender_audit.csv                   # Gender bias summary
│   ├── powerbi_age_audit.csv                      # Age bias summary
│   ├── powerbi_education_audit.csv                # Education bias summary
│   ├── powerbi_department_audit.csv               # Department bias summary
│   ├── powerbi_intersectional.csv                 # Intersectional analysis
│   └── powerbi_bias_gaps.csv                      # Over/under penalised groups
│
├── charts/
│   ├── eda_who_is_leaving.png                     # Session 1 EDA charts
│   ├── feature_importance.png                     # Session 2 model features
│   └── bias_audit_charts.png                      # Session 3 bias visualisations
│
├── dashboard/
│   └── AI_Bias_Audit_Dashboard.pbix               # Power BI dashboard file
│
├── hr_audit.db                                    # SQLite database
└── README.md
```

---

##  How to Run This Project

**Prerequisites:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Steps:**
1. Download the IBM HR dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `/data` folder
3. Run notebooks in order: session1 → session2 → session3 → session4
4. Open `AI_Bias_Audit_Dashboard.pbix` in Power BI Desktop and refresh data sources

---

## 📬 Contact

**Natnael Amenu**
- LinkedIn: [linkedin.com/in/natnael-berhanu-amenu](https://linkedin.com/in/natnael-berhanu-amenu)
- Portfolio: [datascienceportfol.io/natnaelamenu328](https://datascienceportfol.io/natnaelamenu328)
- Email: natnaelamenu328@gmail.com

---

*This project was completed as part of an independent data analytics portfolio. The dataset is publicly available on Kaggle and is used here for educational and portfolio purposes.*

