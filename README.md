# clinical-psychometric-analysis-part1
A statistical analysis (partially manually coded math) on a dataset containing non-standardized psychometrics of various patients (https://shorturl.at/ZHiiK). 

**Note:** In the code, "upper-bound/upper events" are referred to as "rare-events."

---

## Analysis Workflow & Log

1. **Weighted Score Calculation:**  
   - Created weighted scores for each patient as the sum of "transformed" psychometrics.

2. **Stratification by Diagnosis:**  
   - Stratified weighted scores by the diagnosis of each patient's psychiatric disorder.  
   - *Note: Diagnosis was not included in the score calculation.*

3. **Distribution Approximation:**  
   - Approximated normal distributions (n ≈ 30 per group) for weighted scores within each disorder stratum.
   - Created a pooled distribution by sampling equal numbers of weighted score data points from each disorder's weighted score distribution.

4. **Upper-Threshold Analysis:**  
   - Investigated the upper-threshold of weighted scores for each disorder's distribution.
   - Higher weighted scores correspond to more "extreme"/high psychometric values, potentially indicating more severe symptom presentations.

5. **Threshold Optimization:**  
   - Tested various upper thresholds; determined that 1.5 standard deviations above the pooled distribution mean best captured higher weighted scores while maintaining score variability.
   - Higher thresholds produced overly clustered data; lower thresholds led to excessive variability and reduced interpretability.

6. **Gamma Distribution Fitting:**  
   - Fit a gamma distribution to the upper-event data using a manual MLE implementation (gradient ascent to estimate alpha and beta, series approximation for digamma).
   - Gamma distribution was chosen due to the skewed nature of the upper-bound distributions for each disorder.

7. **Comparing Tailness (Kurtosis) and Skew:**  
   - **Goal:** Assess whether the skewness and kurtosis of weighted score distributions differ statistically across disorders.

   **Method 1:**  
   - Sampled 10,000 weighted scores from one disorder distribution and 10,000 from another.
   - Calculated skew/kurtosis for each sample and constructed a confidence interval (CI) for the difference.
   - *Result: Almost no statistical significance between disorders.*

   **Method 2:**  
   - For each disorder, repeatedly sampled 30 data points, calculated sample skew/kurtosis, and built a sampling distribution.
   - Compared two disorders by creating a CI for the difference in means of these sampling distributions.
   - *Result: Again, almost no statistical significance between disorders. However, power analysis revealed low statistical power (power improved with larger sample size), indicating a high chance of Type II error. Thus, it's difficult to conclude whether true differences exist.*

---

## Next Steps & Part 2

- **Limitation Identified:**  
  Creating weighted scores may have obscured differences by hiding distinct psychometric patterns into similar overall scores across disorders.  
  *Example:* A person with BPD and another with depression may both have a weighted score of 5, but the underlying non-standardized psychometric profiles could differ substantially => Planning to analyze in the part 2 project using copulas.

- **Future Directions:**  
  - Develop a new analysis using copulas to directly model the joint distribution of psychometric traits within each disorder.
  - Further analyses to explore additional research questions and possibly revisit the original approach with refined methods.

---
