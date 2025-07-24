# clinical-psychometric-analysis-part1
A statistical analysis (partially manually coded math) on a dataset containing non-standardized psychometrics of various patients (https://shorturl.at/ZHiiK). 

**Note:** In the code, "upper-bound/upper events" are referred to as "rare-events."

---

## Analysis Workflow & Log

1) Created weighted scores for each patient => sum of "transformed" psychometrics
   
2) Stratified weighted scores by diagnosis of psych disorder of patients => diagnosis not included in score

3) Approximated normal distributions (n approx around 30) for each weighted score observations per disorder strata => created a pooled distribution by sampling equal amounts of weighted score data points from each disorder weighted distribution.

4) Wanted to study the upper-threshold of the weighted scores for each disorder distribution => higher weighted scores meant the patients experienced more "extreme"/high psychometrics, potentially indicating a more severe presentation of symptoms

5) Tested various upper-thresholds and found that 1.5 above the stdv of the pooled distribution was the best to observe the higher upper weighted scores yet mantain the variability of scores => a higher threshold resulted in too clustered data vs a lower threshold resulting in too variable and less meaningful data. 

6) Fit a gamma distribution over the upper-event using a manual implementation of mle (gradient ascent to estimate alpha and beta and used series approxiamation to estimate digamma for calculations). Chose gamma since the upper bound distirbutions for each disorder were skewed and the data seemed more appropriate for a gamma dist. 

7) Wanted to compare the upper score distributions for taildness(kurtosis) and skew for each disorder => Is the skew and taildness of each weighted scores distribution statistically different for each disorder? :

Method 1: Sampled 10,000 data points (weighted scores) from one istribution for one disorder and sampled 10,0000 data points from another dist for another disorder. Calculated the skew/kurtosis for each large sample and created a CI for the difference in skew/kurtosis => Almost no statistical significance between disorders  

Method 2: From one weighted score dist for a disorder, sampled 30 data points, found the skew/kurtosis of the sample, created a sampling distribution of skew/kurtosis. Did this for two disorders, so that I had a sampling distribution of skew/kurtosis for both disorders that I was comparing. Created a CI for the difference in means between the two sampling dist => Almost no statistical significance between disorders. HOWEVER => Did power analysis and realized the power was very low ( as expected power did increase by increasing sample size ). Indicates high type 2 error chances => Difficult to tell if the skew/kurtosis is truly different in samples 

## Next Steps & Part 2
I think that creating weighted scores possibly impacted my results by removing the effects of the indiviual psychometric distributions => Eg. A person with bpd can have the weighted score of 5 and a person with depression can have a weighted score of 5, but the non-standardized psychometrics that caused them to get that weighted score can be completely different. I am currently coding a project where I will analyze the disorders by creating copulas of traits and doing an analysis based on that. Furthermore to continue this project I may do some continued analysis to answer some more questions. 


