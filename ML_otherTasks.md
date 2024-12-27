20241227 Updated

---

| **Performance Type** | **Physiological Measures**           | **Indication for Sensory Comfort**                                                                                                                                       | **Related Cognitive Performances**                                |
|-----------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| **EEG**              | - Alpha waves (8-12 Hz)<br>- Beta waves (13-30 Hz)<br>- Theta waves (4-8 Hz) | - **Increased alpha waves**: Indicates relaxation, associated with **visual comfort** or reduced sensory stress.<br>- **Beta suppression**: May reflect sensory overload or discomfort.<br>- **Theta waves**: Linked to thermal discomfort or fatigue. | - **Attention and Concentration**: Higher beta activity indicates better sustained attention.<br>- **Working Memory**: Increased theta during memory tasks.<br>- **Executive Function**: High frontal theta associated with decision-making. |
| **Heart Rate (HR)**  | - Heart Rate Variability (HRV)<br>- Heart Rate (HR)                      | - **Higher HRV**: Indicates lower stress and **thermal comfort**.<br>- **Elevated HR**: Signals auditory or visual discomfort, particularly due to sudden stimuli.                                         | - **Processing Speed**: Faster responses may correlate with higher HRV.<br>- **Executive Function**: Higher HRV suggests better adaptability and control. |
| **Skin Conductance Level (SCL)** | - GSR (Galvanic Skin Response)<br>- Skin Conductance Level (SCL) | - **Increased SCL**: Indicates sensory discomfort or stress, often in response to auditory or visual changes.<br>- **Stable SCL**: Suggests **visual and auditory comfort**.                             | - **Attention and Concentration**: Peaks in SCL indicate response to challenging tasks.<br>- **Processing Speed**: Lower SCL fluctuations during simple tasks. |

---

| **Aspect**                 | **With LSL Synchronisation**                                    | **Without LSL Synchronisation**                             |
|----------------------------|---------------------------------------------------------------|------------------------------------------------------------|
| **Experimental Preparation** | - Easy integration of multiple devices.<br>- Time-synchronisation setup ensures aligned data streams.<br>- Requires basic LSL framework setup. | - Manual setup for each device.<br>- Higher risk of misaligned data.<br>- Increased preparation complexity for multi-device coordination. |
| **Data Collection Convenience** | - Automatic time-stamping ensures accurate synchronisation.<br>- Reduces risk of data loss or mismatched streams.<br>- Supports continuous, real-time monitoring. | - Requires manual effort to align timestamps.<br>- Risk of data gaps or overlaps.<br>- Real-time monitoring limited to individual device streams. |
| **Data Analysis Efficiency**  | - Aligned data ready for immediate analysis.<br>- Compatible with time-sensitive methods (e.g., LSTM).<br>- Minimises preprocessing effort. | - Requires extensive preprocessing to align data.<br>- Errors in synchronisation may affect temporal analysis.<br>- Time-consuming for large datasets. |
| **Reliability of Results**    | - High temporal accuracy for event-response analysis.<br>- Facilitates robust multimodal correlations. | - Lower temporal accuracy.<br>- Difficult to draw precise event-response relationships.<br>- Increased potential for analytical errors. |

---
.

| **Prediction Stage**           | **Data Elements**                   | **AI/ML Methods**                       | **Purpose**                                              | **Expected Outputs**                                              | **Expected Design Optimisation**                                  |
|--------------------------------|------------------------------------|----------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| **1. Baseline Analysis**       | - Comfort Ratings (subjective)<br>- Task Performance (accuracy) | **GLM (Generalised Linear Model)**      | Identify **linear incremental effects** of sensory adjustments relative to visual-only (VA) baseline. | - Incremental effects (p-values, R²)<br>- Effect comparison of VAA, VTA, MSA vs. VA baseline. | - Define baseline sensory conditions for comfort and efficiency.<br>- Identify if single sensory adjustments (e.g., auditory or thermal) are sufficient or if combinations are needed. |
| **2. Feature Importance**      | - Physiological Data:<br>  - EEG, HR, GSR<br>- Sensory Adjustments | **Random Forest (RF)**                  | Determine **most important features** influencing comfort and task performance. | - Feature importance plot showing key predictors (EEG, HR, GSR). | - Prioritise adjustments that significantly improve comfort |
| **3. Complex Non-Linear Analysis** | - Physiological Data<br>- Subjective Feedback<br>- Sensory Adjustments | **RF, SVM, ANN (regression)**           | Predict comfort and task efficiency under **non-linear relationships** between data. | - Predicted comfort/task scores.<br>- Prediction errors (MSE/MAE). | - Optimise **material selections** (e.g., matte or textured finishes) to minimise visual distractions.<br>- Implement dynamic soundscaping systems for improved auditory focus.<br>- Individual thermal controllers for personalised thermal comfort zones. |
| **4. Temporal Prediction**     | - Time-Series Data:<br>  - EEG, HR, GSR over time<br>  - Sequential Sensory Adjustments | **LSTM (Long Short-Term Memory)**       | Predict **dynamic changes** in comfort and performance over time based on physiological data. | - Time-series comfort/task performance predictions.<br>- Accuracy metrics (MSE/MAE). | - Develop real-time **adaptive systems** that adjust lighting intensity, soundscaping, or thermal conditions based on user comfort and task needs over time. |
| **5. Model Integration [Optimised Prediction Systems]**       | - Outputs of LSTM, RF, ANN, SVM    | **Weighted Averaging (Ensemble)**       | Combine predictions for a robust **final comfort/task score**. | - Final weighted prediction for comfort/task performance. | - Implement a **holistic adaptive system** integrating visual, auditory, and thermal adjustments for continuous comfort optimisation.

---

## **AI/ML for Data Analysis**

### **Data Types and Collection**
#### **1. Subjective Feedback**
- **Comfort scores** for sensory adjustments (visual, auditory, and thermal comfort).

#### **2. Objective Data**
- **Physiological Performance**
  - EEG data
  - PPG: Heart rate (HR)
  - Galvanic skin response (GSR)
- **Physical Movement**
  - Head movement
- **Task Performance**
  - Accuracy of cognition test

---

### **Machine Learning Approaches**

#### 1. Quantifying Additional Effects
##### **Use Case**:
Generalised Linear Models (GLM) can be used to identify ==linear incremental effects== (p-values and R² Coefficient of Determination) compared to a baseline. 

- **Objective**: determine how different combinations of sensory adjustments contribute to the outcomes, such as physiological responses, task performance, or comfort levels, in comparison to the baseline.

##### **Model**:
  - **GLM** 
  - **Input Features**: Physiological data (EEG, HR, GSR), sensory conditions (VA/VAA/MSA), sensory interactions
  - **Target**: Comfort Ratings, Task Performance Categories.

##### **Expected Output**:
- show whether adding auditory adjustments (VAA) or adding thermal adjustments (VTA) significantly improves or reduces comfort compared to just visual adjustments (VA).
- reveal how the full multisensory experience (MSA) impacts
  
#### **2. RF/SVM Regression/ANN non-linear relationship prediction**

##### **Use Case**:
Handling complex, ==**non-linear relationships**== between **physiological data** (EEG, HR, GSR) and **subjective feedback** (comfort, task performance).

- **Objective**: Predict participant comfort or task performance based on physiological data and multiple sensory adjustments.

##### **Model**:
- **RF/SVM/ANN** predict the relationship between the independent variables (EEG, HR, GSR, sensory conditions) and the dependent variable (comfort or task efficiency).
  - **Input Features**: Physiological data (EEG, HR, GSR), sensory conditions (VA/VAA/MSA).
  - **Target**: Comfort scores, task performance.

##### **Expected Output**:
- **(RF)Feature importance plot**: Showing which physiological metrics are most important for predicting comfort or performance.
- **(RF, SVM and ANN) Prediction error**: Measuring how accurately the model can predict comfort or performance from physiological data.

#### **3. LSTM Time-Series Prediction for Physiological and Sensory Data**

##### **Use Case**:
LSTM (Long Short-Term Memory) is ideal for handling **time-series data** and capturing the **temporal dependencies** between physiological signals (EEG, HR, GSR) and sensory adjustments (visual, auditory, thermal), along with predicting **subjective feedback** like comfort or task performance over time.

- **Objective**: Predict the **evolution of comfort** or **task performance** over time based on physiological data and changing sensory adjustments during different time periods.

##### **Model**:
- **LSTM** is designed to process time-series data by maintaining memory of past information, allowing it to predict how physiological and sensory adjustments influence comfort or task performance in the future.
  - **Input Features**: Sequential physiological data (EEG, HR, GSR), sensory adjustments over time (visual, auditory, thermal).
  - **Target**: Time-based comfort ratings, task performance scores.

##### **Expected Output**:
- **Time-series predictions**: LSTM provides dynamic predictions of comfort and task performance based on evolving physiological and sensory data across time steps.
- **Prediction error**: Measures how accurately LSTM predicts comfort or performance over time, using metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE).

#### 4. Combine LSTM, ANN, RF, and SVM using Weighted Averaging

1. **Evaluate model performance**:
   - For each model, calculate performance metrics such as **Mean Squared Error (MSE)** or **Accuracy** (for classification tasks). The model with a lower error or higher accuracy will receive a higher weight.
   
2. **Determine the weights for each model**:
   - Assign weights based on the inverse of the error for each model. For example, models with lower errors (better performance) will get higher weights.
   - **Weight calculation**:
     \[
     w_{\text{LSTM}} = \frac{1 / \text{MSE}_{\text{LSTM}}}{(1 / \text{MSE}_{\text{LSTM}}) + (1 / \text{MSE}_{\text{ANN}}) + (1 / \text{MSE}_{\text{RF}}) + (1 / \text{MSE}_{\text{SVM}})}
     \]
     Repeat the same for ANN, RF, and SVM to get \( w_{\text{ANN}} \), \( w_{\text{RF}} \), and \( w_{\text{SVM}} \).

3. **Calculate the ==final weighted average prediction==**:

   - **Weighted average formula**:
     \[
     \text{Final Prediction} = w_{\text{LSTM}} \times \text{LSTM Prediction} + w_{\text{ANN}} \times \text{ANN Prediction} + w_{\text{RF}} \times \text{RF Prediction} + w_{\text{SVM}} \times \text{SVM Prediction}
     \]

#### 5. Applications of the Final Prediction
The **final prediction** from the weighted averaging of **LSTM**, **ANN**, **RF**, and **SVM** can be used in **Unity VR interactions** or in **real-world customised/adaptive design**:
##### Adding a ==implication system== in VR interactions:
The system displays real-time feedback in the form of comfort prediction and cognitive performance implications based on the input physiological performance:
- Comfort Prediction: A bar or gauge that shows the predicted comfort level on a scale (e.g., “Comfort: 80% optimal”).
- Cognitive Performance Implication: A brief message or visual indicator (e.g., “Focus: Improved by 15%” or “Task efficiency: Likely to decrease”). 


---

### **Statistical Analysis Approaches**

#### **1. Linear Regression Analysis**

##### **Use Case**:
Linear regression can be used to explore the relationship between **physiological data** (EEG, HR, GSR) and **subjective feedback** (e.g., comfort scores). 

- **Objective**: To determine if, for example, thermal adjustments significantly affect comfort scores or if EEG data (attention levels) predicts task efficiency.

##### **Model**:
- **Linear regression** is used to examine the impact of one or more independent variables (e.g., physiological metrics, sensory conditions) on a dependent variable (e.g., comfort or task efficiency).
  - **Independent Variables (X)**: Physiological data (EEG, HR, GSR), sensory adjustments (visual, auditory, thermal).
  - **Dependent Variables (Y)**: Subjective feedback (comfort scores, task performance).

##### **Purpose**:
The goal is to estimate how sensory adjustments influence participant comfort and productivity, and to see which physiological measures are correlated with changes in subjective ratings.

##### **Expected Output**:
- **Regression coefficients**: Indicating the strength and direction of the relationship between sensory adjustments or physiological responses and comfort scores.
- **Residuals plot**: Evaluating the model’s accuracy by analysing residuals (differences between predicted and actual values).
- **P-values**: To assess the statistical significance of the predictors.

##### **Advantages**:
- **Simplicity**: Easy to implement and interpret for understanding linear relationships.
- **Efficiency**: Well-suited to small datasets, with fast model training times.

##### **Disadvantages**:
- **Limited to linear relationships**: It cannot capture complex, non-linear interactions between variables.
- **Sensitive to outliers**: Outliers can skew the results.

- **Best for data size**: Works best for **small datasets** or moderately sized datasets with clearly defined linear relationships.

---

#### **2. Analysis of Variance (ANOVA)**

##### **Use Case**:
ANOVA is appropriate for comparing **different sensory conditions** and their effects on subjective feedback, such as comparing comfort or task efficiency across multiple sensory adjustments (e.g., light, noise, temperature).

- **Objective**: To determine if the comfort scores significantly differ under different sensory conditions.

##### **Model**:
- **One-Way or Two-Way ANOVA** can be used to examine how different sensory conditions affect the dependent variable (e.g., comfort scores).
  - **Factors**: Sensory adjustments (e.g., visual, auditory, thermal).
  - **Dependent Variables**: Comfort scores or task performance.

##### **Purpose**:
This analysis can determine whether different sensory adjustments (individually or in combination) significantly affect subjective comfort and task performance.

##### **Expected Output**:
- **F-value and P-value**: Indicating if there is a statistically significant difference in comfort or performance across sensory conditions.
- **Box plots**: Visualising the distribution of comfort or task performance under different conditions.

##### **Advantages**:
- **Effectively compares groups**: Ideal for experiments with multiple sensory conditions.
- **Simple interpretation**: Clearly shows if there are significant differences between conditions.

##### **Disadvantages**:
- **Only tests mean differences**: ANOVA doesn’t provide insights into complex interactions or trends over time.
  
- **Best for data size**: Suitable for **medium-sized datasets** where several groups or conditions are compared.

---


---

### **Summary**

#### **Statistical Analysis**:
- **Linear regression** is ideal for exploring simple relationships in smaller datasets, providing interpretable results but limited to linear associations.
- **ANOVA** effectively compares groups and sensory conditions, revealing significant differences but does not capture complex interactions.

#### **Machine Learning**:
- **Random forest** excels with complex, large datasets, identifying non-linear patterns and feature importance, but lacks interpretability and is computationally expensive.
- **SVM** is suitable for small datasets, providing powerful classification, though it requires significant tuning and is less efficient with larger datasets.

For smaller datasets, **statistical methods** like linear regression and ANOVA are efficient and interpretable. For larger, complex datasets, **machine learning methods** like random forest and SVM offer deeper insights into relationships but come at the cost of increased complexity and computational requirements.
---

机器学习的大多数应用确实是关注于找出输入和输出之间的关系，这是监督学习的核心。然而，也存在不直接依赖于输入-输出关系的机器学习方法，主要体现在无监督学习和增强学习中：

1. **无监督学习**：在无监督学习中，我们没有明确的输出标签或结果来指导模型的学习。目的是让模型自己发现数据中的结构和模式。例如：
   - **聚类**：将数据分组成几个集合或类别，使得同一类别中的数据项彼此相似，而不同类别的数据项不相似。
   - **关联规则学习**：在大数据集中寻找变量之间的有趣关系，比如市场篮子分析，旨在发现商品之间的共同购买模式。
   - **特征提取**：如主成分分析（PCA），用于发现数据中的主要特征和降低数据维度。

2. **增强学习**：在增强学习中，模型通过与环境的互动来学习行为策略，以最大化累积奖励。这里的学习目标是发现哪些行为会在长期内带来最大的奖励，而不是预测某个确定的输出。
   - 代理（模型）在没有明确输出指导的情况下进行尝试和错误，通过奖励和惩罚来学习最佳行为策略。

这些机器学习方法与监督学习的主要区别在于，它们不依赖于预先定义的输出标签来训练模型。无监督学习侧重于发现数据本身的结构和模式，而增强学习侧重于通过试错过程在特定环境中学习策略。