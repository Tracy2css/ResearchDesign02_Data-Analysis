20250215 Updated
**Pre-Test Feedback:**
1. Interview feedback
   - colleague existance or interactions in the virtual environment
     - virtual people will added in Prototype A
   - lack of working tasks in the adjusted environment
     - virtual meeting with specific info match tasks will be added
   - fatigue and time management difficulty due to repetitive cognition tests
     - Only include Fruite Stroop 
   - flickers (Both) and light leak (PICO 4) in the headset
     - VRPC with long cable connections and screen streaming
       -  better graphic qualities
       -  better compatibility with the headset from different brands
       -  better save user's options (Debug.log)
   - sickness possibly caused by movement before finding the workstation
     - no movement to select the workstation before interaction
     - only with body rotation and head orientation
   - Emotiv Flex works but not stable, adding get is too time-consuming
     - wet hair and gel can improve the collection quality, but hard to clean the cap for the next participant
     - other experimenters spend at least AUD 100 in each participant recruitment
   - Emotiv MN8 works well

**Modifications of the final experiment:**

1. **VR Scenarios & Sensory Adjustments:**  
    Before entering the virtual meeting, users adjust sensory settings to achieve a comfortable state. 
    Participants experience four VR conditions: 
   - visual only; 
   - visual + auditory; 
   - visual + thermal; 
   - visual + auditory + thermal. 

2. **Meeting Phase without Further Adjustments (alliesthesia):**  
   Once the virtual meeting begins, participants will remember some info on slides, the sensory settings remain fixed. However, the increased task load and pressure of the meeting may trigger a change in user demands, e.g.
   - cooler
   - quieter
   - brighter

3. **Post-Meeting Assessments:**  
   In additin to the previous survey (sensory pleasantness, mood change, user engagement...), after the meeting, users can
   - (in the real world) match info collected from the meeting (select correct options from some confused answers)
   - (in the real world) answer questionnaires about their new adjustment demands (new sensory preferences) in the meeting performance, such as 1-7 quiter to louder, cooler to warmer 
   - complete brief cognitive tests of Fruite Stroop (measuring concentration and response speed).

---

**From No-Meeting to Meeting: Changes in User Performances and Demands**

- **Changes in User Performance:**  
  Different sensory adjustment combinations may buffer these demand shifts. For instance, even if users develop new needs during the meeting, those in a multi-sensory condition (e.g., visual + auditory + thermal) might show more stable physiological indicators compared to those with a single-sensory adjustment, like 
  - stable heart rate 
  - lower skin conductance changes 
  - maintain better cognitive performance (e.g., faster response times, higher accuracy).
   
  Statistical analysis can be used to identify 
  - not only which sensory adjustment groups are more effective in *achieveng comfort states* (compared to "visual only" group) *before the meeting phase*
  - but also which sensory adjustment groups are more effective in *maintaining performance during the meeting phase (task-loaded phase).*
  - reponding to **SRQ 02:** What are the key sensory interactions and spatial design elements that significantly influence perceived comfort and cognitive performance in open workspaces?

- **Changes in Sensory Adjustment Demands:**  
   - which algorithms and feature (factors) can be used to better predict the demand changes.
     - compare peformances of multiple ML algorithms (e.g., linear regression, decision trees, random forests...) 
     - specify the most prediction contribution features (SHAP)
     - responding to **SRQ03**: How can perceptual changes and adaptive behaviours be predicted based on indoor multisensory variations to optimise configurations in open workspaces?
  - Understand whether the key features have a direct **causal effect** on the ***predicted demand change*** and ***cognitive performance outcomes***.
    - Causal Regression Model for Cognitive Performance (*or other methods?*)
    - responding to **SRQ03**: How can perceptual changes and adaptive behaviours be predicted based on indoor multisensory variations to optimise configurations in open workspace
  - Develop adaptive, data-driven systems that could automatically response to user needs in different task-load conditions.
    - responding to **the main research question:** How can VR technology and ML prediction be effectively employed to analyse and optimise indoor experience for enhanced user comfort and productivity in open workspaces?
  
  


| **Input Features**                                                                                         | **Output Goals**                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| - Sensory adjustment grups and pre-meeting adjustments                               | - Predicted Demand Change Score                   |
| - Subjective pleasantness ratings and mood change (1–7 scale) in the pre-meeting phase                                                                    | - Predicted Cognitive Performance metrics (e.g., reaction time in ms or accuracy on cognitive tests)                                                                     |
| - Physiological data                                         | - Feature importance insights indicating which factors most strongly drive the predicted demand changes and cognitive performance changes                 |
| - Demand Change Score/Cognition Performance Metrics                                        | - Group-level predictions for comparing different sensory configuration groups (to be further analyzed via statistical tests like ANOVA)                                 |

---
20250110 Updated
### Step 1: Machine Learning Prediction Phase

**Data Collection and Input Variables**

For each participant, we collect data in the pre-meeting phase and during the meeting, including:

- **Pre-Meeting Sensory Settings:**  
  - Temperature (°C)
  - Noise Level (dB)
  - Brightness (Lumens)
- **Subjective Feedback:**  
  - Comfort Rating (1–7 scale)
- **Physiological Data:**  
  - ΔHR (change in heart rate, beats per minute)
  - ΔEEG (change in EEG composite index, e.g., power in certain bands)
- **Baseline Cognitive Performance:**  
  - Pre-meeting cognitive score (e.g., reaction time or accuracy)

Our ML model is trained to predict two outcomes:
1. **Demand Change Score (1–7):** How much the participant’s sensory demand changes in the meeting phase (e.g., higher score indicates a stronger need for a cooler or quieter environment).
2. **Cognitive Performance:** Measures such as reaction time or accuracy on a short cognitive test conducted after the meeting.

**Hypothetical Data Example:**

| Participant | Temperature (°C) | Noise (dB) | Brightness (Lumens) | Comfort Rating (1–7) | ΔHR (bpm) | ΔEEG (μV²) | Predicted Demand Change (1–7) | Predicted Cognitive Performance (e.g., Reaction Time in ms) |
|-------------|------------------|------------|---------------------|----------------------|-----------|------------|------------------------------|------------------------------------------------------------|
| 1           | 23               | 55         | 400                 | 6                    | +5        | +0.8       | 4.8                          | 450                                                        |
| 2           | 24               | 60         | 450                 | 5                    | +10       | +1.2       | 5.5                          | 480                                                        |
| 3           | 22               | 50         | 380                 | 7                    | +3        | +0.5       | 4.2                          | 430                                                        |
| 4           | 23               | 58         | 410                 | 6                    | +8        | +1.0       | 5.0                          | 460                                                        |
| 5           | 24               | 62         | 420                 | 5                    | +12       | +1.5       | 5.8                          | 500                                                        |

*Explanation:*  
- ΔHR: Positive values indicate an increase in heart rate (more stress).
- ΔEEG: Higher values indicate increased brain activity related to cognitive load.
- The ML model integrates these inputs to output both a refined demand change score and a predicted cognitive performance metric (e.g., reaction time).

**Feature Importance Analysis (e.g., using SHAP):**

| Feature           | Average SHAP Value for Demand Change | Average SHAP Value for Cognitive Performance |
|-------------------|--------------------------------------|----------------------------------------------|
| ΔHR               | 0.45                                 | 0.40                                         |
| ΔEEG              | 0.35                                 | 0.30                                         |
| Comfort Rating    | 0.15                                 | 0.20                                         |
| Temperature       | 0.05                                 | 0.05                                         |
| Noise Level       | 0.05                                 | 0.05                                         |

*Interpretation:*  
The ML model suggests that both ΔHR and ΔEEG are the most influential predictors for both demand change and cognitive performance. A greater increase in HR and EEG power is associated with higher demand change scores (i.e., a stronger need for environmental adjustments) and slower reaction times.

---

### Step 2: Causal Analysis Phase

**Objective:**  
To determine whether the key features (ΔHR and ΔEEG) have a direct causal effect on the predicted demand change and cognitive performance outcomes.

**Method:**  
We build a multivariable causal regression model controlling for other factors.

**Causal Regression Model for Demand Change:**

\[
\text{DemandChange} = \beta_0 + \beta_1 \times \Delta \text{HR} + \beta_2 \times \Delta \text{EEG} + \beta_3 \times \text{Comfort Rating} + \beta_4 \times \text{Temperature} + \beta_5 \times \text{Noise Level} + \epsilon
\]

**Hypothetical Regression Results for Demand Change:**

| Variable          | Coefficient (\(\beta\)) | Std. Error | p-value | Interpretation                                                                 |
|-------------------|-------------------------|------------|---------|--------------------------------------------------------------------------------|
| Intercept         | 2.0                     | 0.8        | 0.04    | Base demand change score.                                                      |
| ΔHR               | +0.04                   | 0.01       | 0.002   | Each 1 bpm increase in HR corresponds to an increase of 0.04 in demand score.   |
| ΔEEG              | +1.5                    | 0.5        | 0.005   | Each 1 unit increase in EEG index increases the demand score by 1.5.            |
| Comfort Rating    | -0.3                    | 0.2        | 0.08    | Lower comfort is associated with higher demand change, marginally significant.  |
| Temperature       | +0.1                    | 0.1        | 0.15    | Not statistically significant.                                               |
| Noise Level       | +0.02                   | 0.04       | 0.60    | Little effect on demand change.                                               |

**Causal Regression Model for Cognitive Performance:**

Similarly, we can build a model for cognitive performance (e.g., reaction time):

\[
\text{ReactionTime} = \alpha_0 + \alpha_1 \times \Delta \text{HR} + \alpha_2 \times \Delta \text{EEG} + \alpha_3 \times \text{Comfort Rating} + \alpha_4 \times \text{Temperature} + \alpha_5 \times \text{Noise Level} + \epsilon
\]

**Hypothetical Regression Results for Reaction Time:**

| Variable          | Coefficient (\(\alpha\)) | Std. Error | p-value | Interpretation                                                              |
|-------------------|--------------------------|------------|---------|-----------------------------------------------------------------------------|
| Intercept         | 400                      | 20         | 0.01    | Baseline reaction time in ms.                                               |
| ΔHR               | +2.0                     | 0.5        | 0.001   | Each 1 bpm increase in HR increases reaction time by 2 ms (slower performance).|
| ΔEEG              | +20                      | 5          | 0.003   | Each 1 unit increase in EEG index increases reaction time by 20 ms.           |
| Comfort Rating    | -5                       | 3          | 0.05    | Higher comfort is associated with faster reaction times.                    |
| Temperature       | +1.0                     | 0.8        | 0.20    | Not significant.                                                            |
| Noise Level       | +0.5                     | 0.4        | 0.30    | Not significant.                                                            |

*Interpretation:*  
- Both ΔHR and ΔEEG have significant positive coefficients for both outcomes. A higher ΔHR or ΔEEG causally contributes to a higher demand change score and slower reaction times, indicating increased stress and cognitive load.
- These results confirm that changes in HR and EEG are not just correlated with the outcomes but have a direct causal influence.

---

### Step 3: Integrating Predictions and Causal Analysis for Sensory Adjustment Strategy

**Using the Predictions:**  
- The ML model provides personalized demand change scores and predicted cognitive performance. For example, if a participant’s predicted demand change score is high (e.g., 5.8 on a 1–7 scale), it indicates a strong need for environmental adjustment.

**Incorporating Causal Insights:**  
- The causal analysis confirms that significant drivers like ΔHR and ΔEEG have a direct causal effect on these demand scores and cognitive performance.
- With these insights, we can prioritize interventions: for instance, if a drop in HRV (increase in ΔHR) is the main driver, then the system should primarily adjust temperature settings (e.g., make the environment cooler) when such physiological changes are detected.

**Linking to Sensory Adjustment Configurations:**  
- By comparing groups with different sensory adjustment configurations (e.g., visual only vs. visual + auditory vs. visual + thermal vs. multi-sensory), we can determine which configuration best buffers the adverse effects.
- If the multi-sensory (visual + auditory + thermal) group shows smaller increases in demand change scores and maintains better cognitive performance despite ΔHR and ΔEEG changes, this indicates a synergistic benefit.
- The system can then be designed to use adaptive algorithms that not only predict individual demand changes but also recommend specific sensory adjustments (e.g., cooling, noise reduction) tailored to the configuration that minimizes stress effects.

---

### Summary Table: Input Features vs. Output Goals

| **Input Features**                                                                                              | **Output Goals**                                                                                                                       |
|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| - Pre-meeting sensory settings (temperature, noise, brightness)                                               | - Predicted Demand Change Score (1–7 scale: e.g., higher score indicates a stronger need for cooler, quieter, etc.)                     |
| - Subjective comfort ratings (1–7)                                                                              | - Predicted Cognitive Performance (e.g., reaction time in ms or accuracy rate)                                                         |
| - Physiological data (ΔHR in bpm, ΔEEG in μV²)                                                                    | - Feature importance analysis revealing which factors (e.g., ΔHR, ΔEEG) most strongly drive these predictions                         |
| - Contextual indicator (pre-meeting vs. meeting state)                                                          | - Group-level differences in predicted demand changes and cognitive performance (to be compared using ANOVA)                             |

---

### Final Analysis

1. **ML Prediction Stage:**  
   The ML model integrates multi-modal data to output a refined demand change score and cognitive performance metric for each individual. Feature importance analysis indicates that changes in HR and EEG are the most critical predictors.

2. **Causal Analysis Stage:**  
   Using causal regression (and possibly other methods like propensity score matching or causal forests), we validate that ΔHR and ΔEEG have direct causal effects on the demand change score and cognitive performance. For example, each 1 bpm increase in HR leads to a 0.04-point increase in the demand change score, while each 1 unit increase in the EEG index increases the demand score by 1.5 points.

3. **Integrating Insights for Sensory Adjustment:**  
   These results not only help predict which sensory demand changes are most significant but also guide the design of adaptive systems. By understanding the causal relationships, we can configure the sensory environment (e.g., automatically lowering temperature or reducing noise) in a way that addresses the most impactful drivers of demand change and cognitive decline.
   - For instance, if the causal analysis shows that HR changes are the strongest driver, then in scenarios where HR increases sharply, the system could prioritize cooling adjustments.
   - Moreover, comparing different sensory configuration groups using statistical methods (ANOVA) on the predicted scores can reveal which configuration offers a "multi-sensory synergy" effect, maintaining better physiological and cognitive stability.

---

### Conclusion

By first using ML to generate individualized, refined predictions of demand changes and cognitive performance and then applying causal analysis to validate the influence of key factors (like ΔHR and ΔEEG), we obtain actionable insights into which sensory adjustments are most critical. These insights support the development of adaptive, data-driven systems that automatically adjust the workspace environment, optimizing both user comfort and cognitive performance across different sensory configurations.

20241228 Updated

## 1. Explanation

### \(1\) Data to be collected (including subjective and physiological indicators)

- **Relation to SRQ01**:  
  SRQ01 focuses on “how VR technology, physiological measurements, and cognitive assessments can jointly evaluate the indoor multisensory experience.”  
  - Specifying **what** data (subjective ratings, physiological signals like EEG/HRV, cognitive task performance) ensures a **holistic approach** to assessing multisensory experiences in open workspaces.  
  - This step clarifies the integrated methodology—i.e., combining VR scenarios, user feedback, and sensor data.

- **Relation to SRQ02**:  
  SRQ02 aims at identifying the “key sensory interactions and spatial design elements” influencing comfort and performance.  
  - By collecting multi-sensory and spatially contextual data (lighting levels, noise intensities, temperature, user location), researchers can **pinpoint** which elements have the greatest impact.

- **Relation to SRQ03**:  
  SRQ03 is about “predicting perceptual changes and adaptive behaviors based on indoor multisensory variations.”  
  - The **breadth and depth** of the collected data (including real-time changes in environment and user states) form the foundation for building predictive models about user adaptation and reconfiguration needs.

---
以下内容将基于您先前提到的“多感官体验 + 机器学习”思路，**结合当前（四工位 × 两阶段）实验设计**和 **Alliesthesia** 的概念，系统说明：  
1) **应收集哪些数据**（包括主客观与生理指标），  
2) 这些数据**如何作为机器学习（ML）的输入与输出**，  
3) **ML 要完成什么任务**（分类、回归或其他），  
4) **预测结果如何用于开放式办公空间的多感官优化（design for optimizing multisensory experience）**，  
5) 如果与 **Alliesthesia** 现象相关联，如何利用预测结果加深对状态依赖感官需求的理解。  

---

## 一、数据收集与实验对应

为了配合先前设计的实验（在每个工位 A/B/C/D 进行无会议和有会议两个阶段，并记录用户的调节行为、认知测试成绩等），我们建议在 **同一时间段**、**同一工位** 下收集以下数据：

1. **主观评价（Subjective Ratings）**  
   - **多感官舒适度打分**：如视觉舒适度、听觉舒适度、温度舒适度，以及综合舒适度（0-10 或 1-7 等量表）。  
   - **情绪/状态量表**：例如 SAM（Self-Assessment Manikin）或自定义的心情/焦虑/疲劳度打分。  
   - **Alliesthesia 相关反馈**：在“前后状态”切换（无会议→会议）后，用户是否感到同一环境刺激（灯光、噪音、温度）发生了“更佳/更差”的主观变化，可用简单调查题或短答。

2. **客观生理数据（Physiological Data）**  
   - **EEG（脑电波）**：Alpha、Beta、Theta等频段功率，用于衡量专注度、放松度、疲劳度等。  
   - **心率 HR 与心率变异性 HRV**：反映用户紧张或放松程度。  
   - **皮肤电反应（EDA/GSR）**：记录用户在不同刺激和状态下的唤醒水平。  
   - （可选）**体表温度/红外热像**：若重点研究 Thermal Alliesthesia，可监测用户是否出现体温上升或手部温度变化。

3. **任务绩效/认知测试结果（Task Performance）**  
   - **准确率、反应时**：在阶段 1、阶段 2 各进行简短的认知/工作任务，以评估专注度、工作效率等。  
   - **错误率 / 记录得分**：若有更复杂的任务（如记忆或推理测试），则记录用户的正确率或得分变化。

4. **环境与调节行为记录（Environment & Adjustment Behavior）**  
   - **实际感官参数**：如照度（Lux值）、音量/噪声级别（dB）、空调设定温度/风量等。  
   - **用户调节轨迹**：在本工位里，用户对温度/噪音/灯光做了哪些微调？（调节时间点、幅度、频次）  
   - **阶段顺序与空间位置**：记录该受试者在第几个工位、第几阶段、具体座位位置等，以辅助 Spatial/Temporal Alliesthesia 分析。

---

## 二、机器学习：输入（Features）与输出（Targets）

### 1. ML Inputs（特征）

在模型中，可将下列数据整合为多维特征（Feature Vector）：

1. **主观打分**  
   - 舒适度评分（视觉/听觉/温度/整体），情绪状态评分，Alliesthesia 问卷反馈（如“同样声音在会议时更吵”）。  
2. **生理指标**  
   - EEG 不同频段功率、心率 HR、HRV、EDA 等。  
3. **任务绩效**  
   - 认知/工作任务的准确率、反应时间、错误率等。  
4. **环境与调节行为**  
   - 实际噪音分贝（dB）、灯光亮度（Lux）、温度设定（°C）、用户调节次数、调节幅度；  
   - 位置信息（是否靠近噪音源/空调出口）等用于 Spatial 分析。

### 2. ML Outputs（预测目标）

可设置以下预测目标（可选一或多种）：

1. **舒适度或满意度**（分类 / 回归）  
   - **分类**：高/中/低舒适度；  
   - **回归**：0-10 分制的预测。  
2. **认知表现/效率**（回归）  
   - 预测用户在后续任务中的准确率、反应时、工作完成度等，帮助评估多感官配置对工作效率的影响。  
3. **Alliesthesia 效应度量**（可自定义）  
   - 若想量化“同一刺激在前后状态下感知差异”的程度，也可将其定义为模型的输出之一。例如：“差异指数” = (会议状态下的舒适评分) – (无会议状态下的舒适评分)。

---

## 三、ML 要完成的主要任务

1. **回归任务（Regression）**  
   - 预测“给定生理数据、主观偏好、环境参数”时，用户的**舒适度评分**或**工作表现**。  
   - 例：“根据 EEG + HRV + 上一次调节行为 + 环境噪音水平，预测用户此时的整体舒适度 0~10”。  

2. **分类任务（Classification）**  
   - 判断用户在当前多感官设置下是处于“高舒适 vs. 中舒适 vs. 低舒适”三档，或“可接受 vs. 不可接受”的二元分类。  
   - 亦可预测用户是否会再次进行大幅度调节行为（Yes/No）。

3. **时间序列 / 状态跟踪（Temporal Analysis）**  
   - 若想捕捉 **Temporal Alliesthesia**，可使用 LSTM 或 RNN 处理时间序列数据（如 EEG/EDA 随时间变化），预测何时用户会对同一刺激产生“疲劳”或“忍受极限”的状态。

4. **个性化 / 群体聚类（Clustering）**  
   - 通过无监督方法（如 K-Means）将用户分为不同“偏好群体”：如极度怕热群体、对噪音极其敏感群体等。

---

## 四、使用什么模型？为什么？

| **模型**                             | **适用场景**                                           | **理由**                                                                                                       |
|--------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **Random Forest (RF)**               | 回归或分类 (舒适度/效率预测)                          | 对多维度数据表现稳健，能提供特征重要性，易于解释——哪项生理/环境特征对预测贡献大。                               |
| **Support Vector Machine (SVM)**     | 回归或分类，小数据集                                   | 对小规模实验数据能较好地构建高维超平面，分类/回归精度较高。                                                    |
| **Long Short-Term Memory (LSTM)**    | 处理时间序列数据，侧重捕捉状态随时间的变化（Temporal） | 适用于捕捉生理指标（EEG、心率随时间变化）如何影响舒适度或认知表现；能更好体现 Alliesthesia 的“状态演化”。     |
| **Neural Network (ANN/MLP)**         | 回归/分类通用，多感官复杂交互                         | 能处理高度非线性数据，适合发掘视觉-听觉-温度等多感官之间潜在的交互。                                           |
| **K-Means (或其他聚类算法)**          | 分析用户群体偏好                                       | 用于探索无监督模式下的偏好差异，发现不同 Alliesthesia 敏感人群。                                               |

---

## 五、预测结果如何指导多感官设计优化 + Alliesthesia

1. **个性化建议（Personalization）**  
   - 模型预测出“在会议高紧张状态下，某用户最适宜的灯光+温度+噪音水平组合”→ 在开放式办公环境中，可自动或半自动为其切换到对应模式。  
   - 这呼应 Alliesthesia：当用户**状态**（紧张/疲劳）发生改变，系统可**预测**并提供恰当感官环境以维持舒适与效率。

2. **优先级资源投放**  
   - 若模型显示“大部分人”在会议场景中对“噪音”最敏感，而对“温度”第二敏感，那么设计团队可在会议区优先改进隔音、在普通办公区注重温度可控等。  
   - 这能帮助在有限预算内**最大化舒适度与效率提升**。

3. **自动监测与动态调节**  
   - LSTM 等时间序列模型可根据用户实时 EEG/EDA 波动→ 预测即将出现的不适或注意力下降→ 提前自动调节光线/风量。  
   - 在 Alliesthesia 理论下，随着“内在状态”随时间转变（Temporal维度），**外在环境**也能跟进提供“差异化刺激”。

4. **群体/个人差异与 Alliesthesia**  
   - 如果某些人特别显著地对同一声音在前后状态下反馈不一致（高 Alliesthesia 敏感度），模型可能标记其为“噪音敏感型”群体。  
   - 设计者可以针对这些群体提供更加局部或更强的感官调节（如个人降噪头戴），**尊重状态依赖**的需求差异。

5. **验证设计迭代**  
   - 在不同迭代的办公环境设计中，继续收集数据→ 让模型评估舒适度和工作效率→ 判断新设计是否更贴合用户多感官需求、是否更好地应对 Alliesthesia 现象。  
   - 数据驱动的结果能客观衡量改进成效。

---

## 六、与 Alliesthesia 的关系总结

- **Alliesthesia 核心**：用户在**不同生理或心理状态**下，对“同一感官刺激”有不同感知。  
- **ML 预测**：结合生理数据（EEG/HRV/EDA）+ 主观打分+ 环境参数→ 可以**预测**当用户状态变化时，对环境刺激的容忍度/偏好。  
- **设计落地**：在开放式办公中，通过预测结果建立“自适应”或“个性化”多感官调节方案，更好地匹配用户工作流程（无会议 vs. 有会议）和个体差异，**最大化舒适度与工作效率**。这既是对 Alliesthesia 原理的实践，也能在环境设计上体现“动态因人制宜”。

---

### \(2\) How these data serve as ML inputs and outputs

- **Relation to SRQ01**:  
  - Explains the **holistic evaluation pipeline**: physiological data, subjective scores, and spatial parameters become **model inputs**, while predicted comfort or performance levels are **model outputs**.  
  - Demonstrates how VR-based scenarios feed into ML models for a **comprehensive** evaluation.

- **Relation to SRQ02**:  
  - Identifying which sensory or spatial features get included as ML inputs addresses **which design elements** are influential.  
  - Model outputs (e.g., predicted comfort scores) reveal **which interactions** among lights, noise, temperature, or layout matter most.

- **Relation to SRQ03**:  
  - If ML outputs include “predicted adaptive behavior” or “comfort shifts,” then **temporal or real-time** data usage directly ties to how we foresee user changes under different configurations.  
  - This aligns with **optimizing** workspace setups based on predicted user states.

---

### \(3\) ML tasks (classification, regression, etc.)

- **Relation to SRQ01**:  
  - Choosing an ML task (e.g., regression to estimate comfort on a continuous scale, classification for comfort categories) clarifies how multiple data sources are fused to provide a **holistic measure** of user experience.

- **Relation to SRQ02**:  
  - By running **feature importance** or analyzing model performance under different tasks (classification vs. regression), researchers can deduce which **sensory interactions** or design factors most strongly drive outcomes like perceived comfort or cognitive efficiency.

- **Relation to SRQ03**:  
  - ML tasks that focus on **predicting future states** (time-series modeling, classification of user adaptation) reveal how users’ perceptual changes evolve.  
  - In real practice, these models enable **proactive** reconfiguration or design recommendations, addressing how to “optimise configurations in open workspaces.”

---

### \(4\) How prediction results help design optimization (multisensory experience)

- **Relation to SRQ01**:  
  - Demonstrates the **practical application** of a holistic evaluation: from predicted results, designers see which VR-based prototypes or sensory combos best satisfy occupant needs.  
  - Closes the loop between measurement and actionable design changes.

- **Relation to SRQ02**:  
  - Once the key influences (e.g., noise vs. temperature) are identified by the model, designers **prioritize** resources or implement certain spatial design elements (e.g., acoustic partitioning, localized cooling).  
  - This addresses **which** interactions are crucial for perceived comfort and performance.

- **Relation to SRQ03**:  
  - By predicting how occupant behaviors or comfort levels will shift over time or across scenarios, the **optimized** workspace configuration can adapt to dynamic conditions (e.g., adjusting airflows or lighting automatically).  
  - Directly answers “how to reconfigure” based on predicted changes.

---

### \(5\) Alliesthesia linkage: using predictions to enhance understanding of state-dependent needs

- **Relation to SRQ01**:  
  - Alliesthesia (the idea that the same stimuli may be perceived differently as one’s internal or contextual state changes) underscores why VR + physiological measures are needed.  
  - The predictive modeling can **capture** these state shifts to evaluate experiences more thoroughly.

- **Relation to SRQ02**:  
  - Identifying **which** sensory elements vary drastically as users’ states change clarifies how designs might incorporate dynamic or flexible solutions (e.g., adjustable acoustic settings or temperature zones).  
  - Reinforces “key sensory interactions” that might exhibit strong Alliesthesia effects.

- **Relation to SRQ03**:  
  - If ML models predict **perceptual changes** given certain stress or fatigue states, then the workspace can “know” how occupant comfort evolves, adjusting accordingly.  
  - Ties directly to the “adaptive behaviors” question—Alliesthesia is the rationale behind **why** occupant preferences shift, and the model predictions help to **anticipate** those shifts for optimization.

---

## 2. Correspondence Table

Below is a concise table mapping the **five focal points** to **SRQ01, SRQ02, SRQ03**:

| **Focal Point**                                                         | **SRQ01**                                                                                                          | **SRQ02**                                                                                                                  | **SRQ03**                                                                                                                                          |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **(1) Data to be collected**<br>(subjective, physiological, etc.)      | Shows how VR + physiological + cognitive data are combined for a **holistic** multisensory evaluation in open workspaces.                    | Identifies **key sensory** and spatial variables to measure (e.g., lighting, noise, temperature) that might affect comfort/performance.   | Provides the **raw inputs** for predicting adaptive behaviors and state changes under various **indoor multisensory** conditions.         |
| **(2) ML inputs/outputs**                                              | Illustrates how integrated data streams (user feedback, sensor readings) feed into a **complete** ML pipeline for evaluation.                  | By defining which features (sensory/spatial) go in and which target (comfort/performance) is predicted, clarifies **which interactions matter**. | If ML outputs include predictions of behavioral shifts, that informs how design can be **dynamically tuned** to occupant needs.                  |
| **(3) ML tasks**<br>(classification, regression, etc.)                 | Clarifies the approach (e.g. regression for continuous comfort, classification for states) used to **synthesize** multi-modal info.            | Pinpoints **which** tasks (e.g. regression vs. classification) better reveal the **key influence** of certain design elements on user response. | Enables **anticipation** of occupant changes: e.g. time-series or classification triggers reconfiguration, addressing the “how to optimize” aspect. |
| **(4) Design optimization**<br>(multisensory experience)               | Connects the **evaluation** results to actionable design improvements; completes the loop from measurement to practice.                        | Discerns **important design factors** (acoustics, layout, etc.) and suggests **how** to implement solutions for maximizing comfort & performance. | Uses the **predicted** occupant states or comfort levels to automatically or proactively reconfigure the environment in real-time.                   |
| **(5) Alliesthesia**<br>(state-dependent preferences)                   | Explains **why** physiological + VR + cognitive data are critical: they capture how the same stimuli differ under changing user states.        | Emphasizes **which** sensory elements exhibit strong Alliesthesia, guiding flexible or dynamic design elements.                                  | Embeds the rationale behind **adaptive behavior** predictions—enables **configurations** that adapt to occupant states to maintain comfort/efficiency. |

---
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