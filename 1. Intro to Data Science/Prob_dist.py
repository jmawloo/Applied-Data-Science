import pandas as pd
import numpy as np

"""DISTRIBUTIONS: Set of all possible random values
e.g. coin flipping heads + tails.
    -Binomial distribution (2 outcomes)
    -discrete (categories of heads/tails, no real numbers)
    -evenly weighted (heads just as likely as tails)
e.g. Tornado events in Ann Arbor:
    -Binomial Dist., discrete, evenly weighted (tornadoes are rare events).
"""

print(np.random.binomial(1, 0.5)) # run coin-flipping simulation once with 50% chance of landing heads (0).
print(np.random.binomial(1000, 0.5) / 1000) # See a number close to 0.5
# This is how to calculate the number of times the combined outcome of 20 flips is greater than 15 (15 heads).
x = np.random.binomial(20, .5, 10000) #10000 simulations of 20 coin flips, result gets stored in list.
print((x>=15).mean()) # Then we take only the values greater than/ equal to 15

tornadochance = 0.01/100
print(np.random.binomial(100000, tornadochance))

tornadochance = 0.01
tornadoevents = np.random.binomial(1,tornadochance,1000000)
consecutive = 0
for i in range(1, len(tornadoevents)-1): # exclude the first AND last day.
    if tornadoevents[i] == 1 and tornadoevents[i-1] == 1:
        consecutive += 1

print('{} tornadoes back to back in {} years'.format(consecutive,1000000/365), '\n')

#MORE DISTRIBUTIONS
"""
Uniform Distribution: Constant probability over observation time, with continuous plots. (flat horizontal line).
Normal (Gaussian) Distribution: Highest probability in middle of obs value, curving down on the sides (Bell curve).
    -Mean (central tendency) is zero, Std. Dev (measure of variability): How bacly variables are spread out from mean.
    -Expected Value: Mean value we'd expect to get if we performed an infinite number of trials.
    
PROPERTIES:
    -Central Tendency: Mean, median, mode
    -Variability: Standard Deviation, Interquartile range.
    -Kurtosis: Sharpness of peak of freq-dist. curve/ (-/+ respectively mean more flat/sharp)
    -Degrees of Freedom: Number of independent values/quantities that can be assigned to statistical dist.
    -Modality: number of peaks in dist. (one = unimodal, two = bimodal).    
"""
print(np.random.uniform(0,1)) #Any number between these values can be generated.
print(np.random.normal(0.75)) #0.75 is the mean value

dist = np.random.normal(0.75, size=1000)
print(np.sqrt(np.sum((np.mean(dist)-dist)**2)/len(dist))) # Formula for std. Dev
print(np.std(dist)) # also does the trick

import scipy.stats as stats

print(stats.kurtosis(dist)) # negative means more flat, positive more curved.
#Note we're measuring kurtosis of 1000 trials and not a single curve.
""" 
Skewed Normal distributions are called Chi Squared dist. (argument is "degrees of freedom").
    -Degrees of Freedom closely related to # of samples taken from normal population (significance testing).
    -As degrees of freedom increase, dist is more CENTRED>   
    
"""
print(stats.skew(dist))
chi_squared_df2 = np.random.chisquare(2, size=10000) #2 is degrees of freedom.
print(stats.skew(chi_squared_df2)) # skew of nearly 2 is quite large.
chi_squared_df2 = np.random.chisquare(5, size=10000)# resampling for D.o.F. to be 5
print(stats.skew(chi_squared_df2),'\n') # less skewed.

"""
Bimodal Distribution: A dist. that has two high points.(Gaussian Mixture Models).
    -Happen regularly in Data Mining.
    -Can be modelled with 2 normal dist., with diff parameters. (useful for clustering data).
    
"""

"""HYPOTHESIS TESTING
- Core Data analysis behind experimentation.
A/B Testing: The comparison of two similar conditions to see which one provides the better outcome.
Hypothesis: Testable Statement,
    -Alternative Hyp.: our idea (e.g. difference between groups). <- Always more confident about our alt. Hypothesis.
    -Null Hyp.: Alt. of our idea (e.g. no difference b/w groups).
e.g. whether (alt) or not (null) students who sign up faster for course materials perform better than their peers who sign up later.
"""
df = pd.read_csv('grades.csv')
print(df.head())
print(len(df))

early = df[df['assignment1_submission'] <= '2015-12-31']
late = df[df['assignment1_submission'] > '2015-12-31']
print(early.mean()) # Date time values ignored; pandas knows it's not a number.
print(late.mean()) #Those who submit late are almost equally as often as those submitting early.

"""
Critical Value alpha (a)
    -Threshold as to how much chance one willing to accept.
    -Typical values in social sciences = .1, .05, .01. Depends on what you're doing and the amount of noise in your data.
    - Physics labs have much lower tolerance for alpha values.
    - Lower-cost versus high cost: low interventions (e.g. email reminder) are convenient, but higher interventions
    (e.g. calling the student) is more of a burden on both the student and institution (higher burden of proof, lower critical value alpha.

T-test: One way to compare averages of 2 different populations.
    -Result includes a statistic and a p-value.
**Most statistical tests require data to conform to certain shape. Therefore check data 1st before applying any test.

"""
p_valstat = stats.ttest_ind(early['assignment1_grade'],late['assignment1_grade'])
print(p_valstat)
# since p_val is larger than 0.05 critical value, cannot reject null hypothesis (2 populations are the same). I.e. no statistically significant difference between THESE 2 samples (not all samples).
print(stats.ttest_ind(early['assignment2_grade'],late['assignment2_grade'])) #p_val still too large
print(stats.ttest_ind(early['assignment3_grade'],late['assignment3_grade'])) #Close, but still far beyond value.
"""
Generally the more t-tests run, the more likely we'll obtain a positive result.
    -Called p-hacking/dredoing when u do many tests untol u find a statistically significant one; serious methodological issue.
    -At a confidence level of 0.05, expect to find one positive result in 1 time out of 20 tests.
    -Remedies:
        -Bonferroni correction : tighten alpha value. (Threshold based on number of tests being run).
        -Hold-out sets (i.e. is data generalizable? Form specific hypotheses based on isolated data sets, then run experiments based on more limiting hypothesis.
        Used in Machine learning to build predictive models (cross-fold validation)
        -Investigation pre-registration. Outline what to expect to find + why, and describe tests that would backup positive proof of this.
        Register with 3rd party in academic circles (e.g. journal/board determining whether it's reasonable or not.) Then run
        study and report results regardless of it being positive or not.
        Experience larger burden in connecting to existing theory since must convince board of feasibility of testing hypothesis.  
"""
