import csv
import math
import numbers
import os
import random
import shutil
import statistics


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from scipy.odr import quadratic
import sympy as sp
from sympy import *




from sympy import Eq
from sympy import diff, integrals, integrate
import pandas as pd

from joblib import Parallel, delayed



alpha,beta,x_i = symbols('alpha beta x_i',real=true)

alpha_all_coeff=[]
beta_all_coeff=[]

#mapping the qualitative observations to a quantitative measure (discrete random variable)
def return_value_map(filter):
    v_map={}

    if filter == "u_s_s_m":
        v_map={
            "Usually":3,
            "Seldom":1,
            "Sometimes":2,
            "Most-Often":4
        }
    elif filter =="binary":
        v_map={
            "NO":0,
            "YES":1
        }
    elif filter=="out_ten":
        v_map={
            "1 From 10":1,
            "2 From 10":2,
            "3 From 10":3,
            "4 From 10":4,
            "5 From 10":5,
            "6 From 10":6,
            "7 From 10":7,
            "8 From 10":8,
            "9 From 10":9,
            "10 From 10":10
        }
    return v_map

#returning all unique values and the total count per unique value in the list
def return_unique_freq_dict(param_list):
    unique_dict ={}

    for x in param_list:
        if not x in unique_dict:
            unique_dict[x] = param_list.count(x)

    return unique_dict

#estimating digamma via series approximation
def calculate_digamma(alpha_coeff):

    digamma_equation = log(alpha)

    for coeff,power  in zip([-1/2,-1/12,-1/252,1/240,-5/660,691/32760,-1/12],
                            [1,2,4,6,8,10,12,14]):
        digamma_equation += coeff*(alpha**(-power))

    digamma_estimate = digamma_equation.subs({alpha: alpha_coeff}).evalf()

    return digamma_equation


#solving for gradient diff term
def gradient_plug_in(coeff_alpha,coeff_beta,data_list,gradient_matrix):
    subs_coeff = {
        alpha: coeff_alpha,
        beta: coeff_beta
    }

    print(f"gradient_diff_alpha_inital: {gradient_matrix[0]}")
    print(f"gradient_diff_beta_inital: {gradient_matrix[1]}")

    gradient_matrix_calc = gradient_matrix.subs({polygamma(0, alpha): calculate_digamma(coeff_alpha)})

    #print(gradient_matrix_calc[0])

    gradient_matrix_calc = gradient_matrix_calc.subs(subs_coeff)

    print(f"gradient_diff_alpha_inital: {gradient_matrix_calc[0]}")
    print(f"gradient_diff_beta_inital: {gradient_matrix_calc[1]}")

    grad_matrix_alpha = lambdify(x_i,gradient_matrix_calc[0],modules='numpy')
    grad_matrix_beta = lambdify(x_i, gradient_matrix_calc[1], modules='numpy')

    diff_matrix_alpha = grad_matrix_alpha(np.array(data_list))
    diff_matrix_beta = grad_matrix_beta(np.array(data_list))

    print(f"diff_matrix_alpha: \n{diff_matrix_alpha}")
    print(f"\n\ndiff_matrix_beta: \n{diff_matrix_beta}")

    diff_matrix_alpha = sum(diff_matrix_alpha)
    diff_matrix_beta = sum(diff_matrix_beta)

    print(f"diff_matrix_alpha: \n{diff_matrix_alpha}")
    print(f"\n\ndiff_matrix_beta: \n{diff_matrix_beta}")

    return np.array([diff_matrix_alpha,diff_matrix_beta])

    # gradient_matrix_np = lambdify(x_i,gradient_matrix_calc,modules='numpy')
    #
    # print(f"gradient_matrix_np: {gradient_matrix_np[0]}\n{gradient_matrix_np[1]}")
    #
    # diff_matrix = np.array([f(np.array(data_list)) for f in gradient_matrix_np])  # shape (2, N)
    # diff_sum = diff_matrix.sum(axis=1)  # shape (2,)
    #
    # print(f"Differential_SUM :{diff_sum}")

    # return diff_sum


def initial_gamma_guess(np_data_list):
    alpha_guess = math.pow(np_data_list.mean(),2)/np_data_list.var()
    beta_guess = np_data_list.mean()/np_data_list.var()

    return [alpha_guess,beta_guess]

#checking for coeff convergence
def is_converging():

    if len(alpha_all_coeff)>1 and len(beta_all_coeff)>1:



        #theta_convergence = [math.pow((alpha_all_coeff[i]-alpha_all_coeff[i-1]),2) + math.pow((beta_all_coeff[i]-beta_all_coeff[i-1]),2)
        #                     for i in range(1,len(alpha_all_coeff))]

        #theta_convg = sqrt(np.array(theta_convergence).sum())

        theta_convg = sqrt(
            math.pow((
                alpha_all_coeff[len(alpha_all_coeff)-1] - alpha_all_coeff[len(alpha_all_coeff)-2]
            ),2) + math.pow((
                beta_all_coeff[len(beta_all_coeff)-1] - beta_all_coeff[len(beta_all_coeff)-2]
            ),2)
        )

        epsilon = 10**-4

        print(f"theta_convg: {theta_convg}")

        if theta_convg < epsilon:
            return True

    return False

#gradient ascent to estimate the parameters for a gamma distribution
def gradient_ascent_gamma(gradient_matrix,alpha_guess,beta_guess,data_list):
    n_learning_curve = 0.01

    initial_guess = np.array([alpha_guess,beta_guess])
    alpha_all_coeff.append(initial_guess[0])
    beta_all_coeff.append(initial_guess[1])

    np.asarray(alpha_all_coeff)
    np.asarray(beta_all_coeff)

    print(f"{initial_guess}")

    while not is_converging():

        # coeffs [alpha_i / beta_i] =  prev_coeff [alpha_(i-1) / beta_(i-1) ] + learning_curve n [n / n] * differential_likelihood_plugged [ diff_alpha / diff_beta ]
        initial_guess = (np.array([alpha_all_coeff[len(alpha_all_coeff)-1],beta_all_coeff[len(beta_all_coeff)-1]]) + n_learning_curve *
                         gradient_plug_in(alpha_all_coeff[len(alpha_all_coeff)-1],beta_all_coeff[len(beta_all_coeff)-1],data_list, gradient_matrix))

        print(f"previous coeff: {np.array([alpha_all_coeff[len(alpha_all_coeff)-1],beta_all_coeff[len(beta_all_coeff)-1]])}")

        alpha_all_coeff.append(initial_guess[0])
        beta_all_coeff.append(initial_guess[1])

        print(f"Next coeff: {initial_guess}")

    return initial_guess



def gamma_mle(data_list):
    np_data_list = np.asarray(data_list)
    ll_gama_inner_sigma = alpha*log(beta) - log(gamma(alpha)) + (alpha-1)*log(x_i) - beta*x_i

    partials = [diff(ll_gama_inner_sigma,partial_diff_param) for partial_diff_param in [alpha,beta]]

    gradient_matrix = Matrix(partials)

    alpha_guess,beta_guess = initial_gamma_guess(np_data_list)

    alpha_estimate,beta_estimate = gradient_ascent_gamma(gradient_matrix, alpha_guess, beta_guess, data_list)


    plt.figure()

    gamma_dist_obj = stats.gamma(a=alpha_estimate,scale=1/beta_estimate)

    x_cont = np.linspace(min(data_list), max(data_list), 1000)

    x_discrete = np.asarray(data_list)

    gamma_pdf_list =  gamma_dist_obj.pdf(x_discrete)



    plt.plot(x_cont, gamma_dist_obj.pdf(x_cont), 'r-', lw=2, label="Fitted Gamma PDF")
    plt.bar(data_list,gamma_pdf_list,color="red",alpha=0.7)
    plt.hist(data_list,color="orange",bins=25,density=true)

    plt.show()

    return stats.gamma(a=alpha_estimate,scale=1/beta_estimate)

def skew_kurtosis_lsample_diff_ci(dist_obj_1,dist_obj_2):

    n_sample =10000

    large_sample_dist_1 = dist_obj_1.rvs(size=n_sample)
    for i in range(0,len(large_sample_dist_1)): large_sample_dist_1[i]=int(large_sample_dist_1[i])
    large_sample_dist_2 = dist_obj_2.rvs(size=n_sample)
    for i in range(0, len(large_sample_dist_2)): large_sample_dist_2[i]=int(large_sample_dist_2[i])

    plt.figure()

    plt.hist(large_sample_dist_1,color="blue",bins=30)
    plt.hist(large_sample_dist_2, color="red", bins=30)

    plt.show()

    diff_skew_l1_l2 = stats.skew(large_sample_dist_1) - stats.skew(large_sample_dist_2)
    diff_kurtosis_l1_l2 = stats.kurtosis(large_sample_dist_1) - stats.kurtosis(large_sample_dist_2)

    s_error_skew = sqrt(6/n_sample)
    s_error_kurt = sqrt(24/n_sample)

    me_skew = stats.norm.ppf(0.95)*sqrt(math.pow(s_error_skew,2)+math.pow(s_error_skew,2))
    me_kurt = stats.norm.ppf(0.95)*sqrt(math.pow(s_error_kurt,2)+math.pow(s_error_kurt,2))

    return [[diff_skew_l1_l2 - me_skew , diff_skew_l1_l2 + me_skew],
            [diff_kurtosis_l1_l2 - me_kurt , diff_kurtosis_l1_l2 + me_kurt]]


# finds the ci for the diff of mean of the skew/kurtosis sampling dist
def skew_kurtosis_sampling_mean_diff_ci(dist_obj_1,dist_obj_2,param_number_samples,param_n_size):

    number_success_skew = 0
    number_success_kurtosis =0

    number_samples = param_number_samples
    n_sample_size = param_n_size

    large_skew_sample_list_1 = []
    large_skew_sample_list_2 = []

    large_kurt_sample_list_1 = []
    large_kurt_sample_list_2 = []

    #creating a distribution of individual sample statistics for both disorders given number_sample and sample_size (n_size) parameters

    for i in range(1,number_samples):

        #creating individual samples for two individual disorders

        indiv_samp_dist_1 = dist_obj_1.rvs(size=n_sample_size)
        for i in range(0, len(indiv_samp_dist_1)): indiv_samp_dist_1[i] = int(indiv_samp_dist_1[i])
        indiv_samp_dist_2 = dist_obj_2.rvs(size=n_sample_size)
        for i in range(0, len(indiv_samp_dist_2)): indiv_samp_dist_2[i] = int(indiv_samp_dist_2[i])

        indiv_samp_skew_dist_1 , indiv_samp_skew_dist_2 = stats.skew(indiv_samp_dist_1) , stats.skew(indiv_samp_dist_2)
        indiv_samp_kurt_dist_1, indiv_samp_kurt_dist_2 = stats.kurtosis(indiv_samp_dist_1), stats.kurtosis(indiv_samp_dist_2)

        diff_indiv_samp_skew_1_2 = indiv_samp_skew_dist_1-indiv_samp_skew_dist_2
        diff_indiv_samp_kurt_1_2 = indiv_samp_kurt_dist_1-indiv_samp_kurt_dist_2

        s_error_skew = sqrt(6 / n_sample_size)
        s_error_kurt = sqrt(24 / n_sample_size)

        me_skew = stats.norm.ppf(0.90) * sqrt(math.pow(s_error_skew, 2) + math.pow(s_error_skew, 2))
        me_kurt = stats.norm.ppf(0.90) * sqrt(math.pow(s_error_kurt, 2) + math.pow(s_error_kurt, 2))

        ci_indiv_samp_dskew = [diff_indiv_samp_skew_1_2-me_skew,diff_indiv_samp_skew_1_2+me_skew]
        ci_indiv_samp_dkurt = [diff_indiv_samp_kurt_1_2-me_kurt,diff_indiv_samp_kurt_1_2+me_kurt]

        # checking if individual ci for skew/kurtosis of individual sample diff fails to reject/rejects (contains 0 or not)
        # since a lot of sample noise => calculating power

        if not (ci_indiv_samp_dskew[0] < 0 < ci_indiv_samp_dskew[1]) :
            number_success_skew+=1
            #print(f"Success CI Skew Diff: {ci_indiv_samp_dskew}")
        if not (ci_indiv_samp_dkurt[0] < 0 < ci_indiv_samp_dkurt[1]):
            number_success_kurtosis += 1
            #print(f"Success CI Kurt Diff: {ci_indiv_samp_dkurt}")


        #creating the full sampling distribution

        large_skew_sample_list_1.append(indiv_samp_skew_dist_1)
        large_skew_sample_list_2.append(indiv_samp_skew_dist_2)
        large_kurt_sample_list_1.append(indiv_samp_kurt_dist_1)
        large_kurt_sample_list_2.append(indiv_samp_kurt_dist_2)


    #fitting a normal distribution over the sampling distribution

    mu_samp_skew_dist_1 , stdv_samp_skew_dist_1 = stats.norm.fit(large_skew_sample_list_1)
    mu_samp_skew_dist_2, stdv_samp_skew_dist_2  = stats.norm.fit(large_skew_sample_list_2)

    large_skew_samp_dist_1 = stats.norm(mu_samp_skew_dist_1,stdv_samp_skew_dist_1)
    large_skew_samp_dist_2 = stats.norm(mu_samp_skew_dist_2, stdv_samp_skew_dist_2)

    mu_samp_kurt_dist_1, stdv_samp_kurt_dist_1  = stats.norm.fit(large_kurt_sample_list_1)
    mu_samp_kurt_dist_2, stdv_samp_kurt_dist_2  = stats.norm.fit(large_kurt_sample_list_2)

    large_kurt_samp_dist_1 = stats.norm(mu_samp_kurt_dist_1,stdv_samp_kurt_dist_1)
    large_kurt_samp_dist_2 = stats.norm(mu_samp_kurt_dist_2,stdv_samp_kurt_dist_2)

    t_skew = large_skew_sample_list_1 + large_skew_sample_list_2
    t_kurt = large_kurt_sample_list_1 + large_kurt_sample_list_2

    x_bounds_skew = np.linspace(min(t_skew)-2,max(t_skew)+2)
    x_bounds_kurt = np.linspace(min(t_kurt)-2,max(t_kurt)+2)

    plt.figure()
    print("skew")
    plt.plot(x_bounds_skew,large_skew_samp_dist_1.pdf(x_bounds_skew),"red",label="large_skew_samp_dist_1")
    plt.hist(large_skew_sample_list_1,color="red",density=true)
    plt.plot(x_bounds_skew, large_skew_samp_dist_2.pdf(x_bounds_skew), "blue", label="large_skew_samp_dist_1")
    plt.hist(large_skew_sample_list_2, color="blue", density=true)

    plt.axvline(mu_samp_skew_dist_1,color="red")
    plt.axvline(mu_samp_skew_dist_2, color="blue")

    plt.show()

    plt.figure()
    print("kurtosis")
    plt.plot(x_bounds_kurt, large_kurt_samp_dist_1.pdf(x_bounds_kurt), "red", label="large_skew_samp_dist_1")
    plt.hist(large_kurt_sample_list_1, color="red", density=true)
    plt.plot(x_bounds_kurt, large_kurt_samp_dist_2.pdf(x_bounds_kurt), "blue", label="large_skew_samp_dist_1")
    plt.hist(large_kurt_sample_list_2, color="blue", density=true)

    plt.axvline(mu_samp_kurt_dist_1, color="red")
    plt.axvline(mu_samp_kurt_dist_2, color="blue")

    plt.show()

    mean_diff_skew_l1_l2 = mu_samp_skew_dist_1 - mu_samp_skew_dist_2
    mean_diff_kurt_l1_l2 = mu_samp_kurt_dist_1 - mu_samp_kurt_dist_2

    me_skew = stats.norm.ppf(0.90)*sqrt(
        (math.pow(stdv_samp_skew_dist_1,2)/sqrt(n_sample_size)) + (math.pow(stdv_samp_skew_dist_2,2)/sqrt(n_sample_size))
    )

    me_kurt = stats.norm.ppf(0.90) * sqrt(
        (math.pow(stdv_samp_kurt_dist_1, 2) / sqrt(n_sample_size)) + (math.pow(stdv_samp_kurt_dist_2, 2) / sqrt(n_sample_size))
    )

    ci_diff_skew = [mean_diff_skew_l1_l2-me_skew,mean_diff_kurt_l1_l2+me_skew]
    ci_diff_kurt = [mean_diff_kurt_l1_l2-me_kurt,mean_diff_kurt_l1_l2+me_kurt]

    power_skew_diff = number_success_skew/number_samples
    power_kurt_diff = number_success_kurtosis/number_samples


    # returning the cis as well as the powers for skew/kurt

    return [ci_diff_skew,ci_diff_kurt,power_skew_diff,power_kurt_diff]












weighted_scored_list =[]


og_csv_file = "Dataset-Mental-Disorders.csv"
df = pd.read_csv(og_csv_file)
filter=""
index=0

#converting qualitative psychometric observations to quantitative discreet observations
# => creating weighted scores for each individual patient

for col in df.columns:
    if not col=="Patient Number" and not col =="Expert Diagnose":

        if index<5: filter="u_s_s_m"
        elif index>14 : filter="out_ten"
        else: filter="binary"

        df[f"{col}_Score"] = df[col].map(return_value_map(filter))
        value =  df[f"{col}_Score"]
        #print(f"{col}_Score:\n  {value}")
    index+=1


score_index=[]
for i in range(19,36): score_index.append(i)

for diagnosis in ["Bipolar Type-1","Bipolar Type-2","Depression","Normal"]:
    #print( df.iloc[:,score_index])
    #print( df.loc[:,score_index])
    #print( df[:,score_index])

    df[f"{diagnosis}_Score"] = df.loc[df["Expert Diagnose"]==diagnosis, df.columns[score_index]].sum(axis=1)  #getting each individual patient weighted scores

bpd_1_score=[]
bpd_2_score=[]
depression_score=[]
normal_score=[]
pooled_scores=[]


for diagnosis_score,diagnosis_score_list in zip(["Bipolar Type-1_Score","Bipolar Type-2_Score","Depression_Score","Normal_Score"],
                                          [bpd_1_score,bpd_2_score,depression_score,normal_score]):
    for score in df[diagnosis_score]:
        if not math.isnan(score):
            diagnosis_score_list.append(score)


pooled_scores = bpd_1_score + bpd_2_score + depression_score + normal_score

x_bounds = np.linspace(min(pooled_scores)-2, max(pooled_scores)+2, 1200)

for score_list in [bpd_1_score,bpd_2_score,depression_score,normal_score]:
    score_list = [int(x) for x in score_list]



#fitting approx normal distributions over each weighted scores distribution => stratified by diagnosis

mu_bpd_1, stdv_bpd_1 = stats.norm.fit(bpd_1_score)
bpd_1_norm = stats.norm(mu_bpd_1,stdv_bpd_1)
#pdf_norm_bpd_1 = bpd_1_norm.pdf(x_bounds)

mu_bpd_2, stdv_bpd_2 = stats.norm.fit(bpd_2_score)
bpd_2_norm = stats.norm(mu_bpd_2,stdv_bpd_2)
#pdf_norm_bpd_2 = bpd_2_norm.pdf(x_bounds)

mu_depression, stdv_depression = stats.norm.fit(depression_score)
depression_norm = stats.norm(mu_depression,stdv_depression)
#pdf_norm_depression = depression_norm.pdf(x_bounds)

mu_normal, stdv_normal = stats.norm.fit(normal_score)
normal_norm = stats.norm(mu_normal,stdv_normal)
#pdf_norm_normal = normal_norm.pdf(x_bounds)

mu_pooled, stdv_pooled = stats.norm.fit(pooled_scores)
pooled_norm = stats.norm(mu_pooled,stdv_pooled)
#pdf_norm_pooled = pooled_norm.pdf(x_bounds)

pdf_norm_bpd_1,pdf_norm_bpd_2,pdf_norm_depression,pdf_norm_normal,pdf_norm_pooled =\
    [
     frozen_dist_obj.pdf(x_bounds)
     for frozen_dist_obj in
     [bpd_1_norm,bpd_2_norm,depression_norm,normal_norm,pooled_norm]
     ]


# for pdf_dist,data,label,color,width in zip(
#                                 [pdf_norm_bpd_1,pdf_norm_bpd_2,pdf_norm_depression,pdf_norm_normal,
#                                 pdf_norm_pooled],
#                                 [bpd_1_score,bpd_2_score,depression_score,normal_score,
#                                 pooled_scores],
#                                 ["pdf_norm_bpd_1","pdf_norm_bpd_2","pdf_norm_depression","pdf_norm_normal",
#                                 "pdf_norm_pooled"],
#                                 ['blue','green','cyan','magenta','black'],
#                                 [1.5,1.5,1.5,1.5,4]
#                                ):

#     plt.plot(x_bounds,pdf_dist,color=color,lw=width,label=label)
#     plt.hist(data, bins=30, density=True, alpha=0.5, color=color)
#
#
# plt.axvline(statistics.mean(pooled_scores),color='purple')
# #plt.show()

#creating a stratified pooled sample of weighted scores => random sampling from each diagnosis distribution of weighted scores

strata_pooled_sample = []

for frozen_dist_obj in [bpd_1_norm,bpd_2_norm,depression_norm,normal_norm,pooled_norm]:
    r_disorder_sample = list(frozen_dist_obj.rvs(size=30))
    for i in range(0,len(r_disorder_sample)): r_disorder_sample[i]=int(r_disorder_sample[i])

    strata_pooled_sample+=r_disorder_sample

#plt.figure()
dot_plot_scatter = return_unique_freq_dict(strata_pooled_sample)
print(f"strata_pool: {strata_pooled_sample}")
print(f"unique_count: {dot_plot_scatter}")
#plt.scatter(list(dot_plot_scatter),list(dot_plot_scatter.values()),color='red')
#plt.hist(list(strata_pooled_sample),bins=19,color='orange',density=true)
#plt.axvline(statistics.mean(strata_pooled_sample),color='blue')
x_strata_pooled_bounds=np.linspace(
    statistics.mean(strata_pooled_sample)-2*statistics.stdev(strata_pooled_sample),
    statistics.mean(strata_pooled_sample)+2*statistics.stdev(strata_pooled_sample)
)
strata_pooled_mu,strata_pooled_stdv = stats.norm.fit(strata_pooled_sample)
strata_pooled_norm = stats.norm(strata_pooled_mu,strata_pooled_stdv)
pdf_strata_pooled = strata_pooled_norm.pdf(x_strata_pooled_bounds)
#plt.plot(x_strata_pooled_bounds,pdf_strata_pooled,color='magenta',lw=1)

#plt.show()

#testing diff thresholds => trying to observe the upper bound scores 

ineq_v_1 = strata_pooled_mu + 2*statistics.stdev(strata_pooled_sample)
ineq_v_2 = strata_pooled_mu + (stats.norm.ppf(0.8))*statistics.stdev(strata_pooled_sample)
ineq_v_3 = strata_pooled_mu + (stats.norm.ppf(0.9))*statistics.stdev(strata_pooled_sample)
ineq_v_main = strata_pooled_mu + 1.5*statistics.stdev(strata_pooled_sample)  # ---> USING THIS FOR RARE MAIN
ineq_v_5 = strata_pooled_mu + 1*statistics.stdev(strata_pooled_sample)

# for ineq_threshold,ineq_label in zip([ineq_v_1,ineq_v_2,ineq_v_3,ineq_v_4,ineq_v_5],
#                                      ["2_stdv above","80th percentile(left)","90th percentile(left)","1.5_stdv above","1_stdv above"]
#                                      ):

rare_events_strata_pool = []
rare_events_bpd_1 = []
rare_events_bpd_2 = []
rare_events_depression = []
rare_events_normal = []

#sampling from upper threshold to study rare events

for frozen_dist_obj,rare_event_list in zip(
                                [bpd_1_norm,bpd_2_norm,depression_norm,normal_norm,strata_pooled_norm],
                                [rare_events_bpd_1,rare_events_bpd_2,rare_events_depression,rare_events_normal,
                                 rare_events_strata_pool]
                                ):
    while len(rare_event_list)<120:
        x_sample_i = frozen_dist_obj.rvs(size=1)
        x_sample_i = x_sample_i.item()
        if x_sample_i > ineq_v_main : rare_event_list.append(int(x_sample_i))


#print(f"\n\n\n{ineq_v_main}: \n")

# for  rare_event_list,label in zip([rare_events_bpd_1,rare_events_bpd_2,rare_events_depression,rare_events_normal,
#                                    rare_events_strata_pool],
#                                    ["bpd_1","bpd_2","depression","normal","strata_pool"]
#                                   ):
    # print(f"{label}")
    #dot_plot_i = return_unique_freq_dict(list(rare_events_strata_pool))
    #plt.scatter(list(dot_plot_i), list(dot_plot_i.values()), color='red')
    #plt.hist(rare_event_list, color='orange', bins=25, density=true)
    #plt.show()


# dot_plot_i = return_unique_freq_dict(list(rare_events_strata_pool))
# plt.scatter(list(dot_plot_i), list(dot_plot_i.values()), color='red')
# plt.hist(rare_events_strata_pool,color='orange',bins=25,density=true)
# plt.show()
#
# plt.figure()
# for rare_events_data,color,label in zip([rare_events_bpd_1,rare_events_bpd_2,rare_events_depression,rare_events_normal,
#                                  rare_events_strata_pool],
#                                         ['blue','magenta','red','yellow','orange'],
#                                         ["bpd_1","bpd_2","depression","normal","strata_pool"]
#                                         ):
#     dot_plot_i = return_unique_freq_dict(list(rare_events_data))
#     plt.scatter(list(dot_plot_i),list(dot_plot_i.values()),color=color,label=label)
#     plt.show()


#creating frozen distribution objects to represent each rare event dist per strata diagnosis => doing manual mle to estimate the
#rare score distribution for each diagnosis

r_events_spool_gamma_dist_obj, r_events_bpd1_gamma_dist_obj, r_events_bpd2_gamma_dist_obj,r_events_depression_gamma_dist_obj, r_events_normal_gamma_dist_obj =\
    [
    gamma_mle(data_list)
        for data_list in [rare_events_bpd_1,rare_events_bpd_2,rare_events_depression,rare_events_normal,
                                  rare_events_strata_pool]
]


# diff in skew/kurt using large sample and ci
dist_obj = [r_events_bpd1_gamma_dist_obj, r_events_bpd2_gamma_dist_obj,r_events_depression_gamma_dist_obj, r_events_normal_gamma_dist_obj]
dist_names = ["rare events bpd1 (large sample)","rare events bpd2 (large sample)","rare events depression (large sample)","rare events normal (large sample)"]

for i in range(0,len(dist_obj)-1):

    for j in range(i+1,len(dist_obj)):

        print(f"\n\ndifference in {dist_names[i]}-{dist_names[j]}: \n")
        ci_diff = skew_kurtosis_lsample_diff_ci(dist_obj[i],dist_obj[j])
        print(f"skew: {ci_diff[0]}\nkurtosis: {ci_diff[1]}")



# diff in skew/kurt using sampling dist method and diff in sampling dist mean ci
# dist_obj = [r_events_spool_gamma_dist_obj,r_events_bpd1_gamma_dist_obj, r_events_bpd2_gamma_dist_obj,r_events_depression_gamma_dist_obj, r_events_normal_gamma_dist_obj]
dist_names = ["rare events bpd1 (sampling dist)","rare events bpd2 (sampling dist)","rare events depression (sampling dist)","rare events normal (sampling dist)"]

for i in range(0,len(dist_obj)-1):

    for j in range(i+1,len(dist_obj)):

        power_analysis_skew  = []
        power_analysis_kurt = []

        sample_list = [30,120,240,360,480,600]

        #epsilon = 0.0001

        for sample_size in sample_list:

            # diff_skew = power_analysis_skew[len(power_analysis_skew)-1] - power_analysis_skew[len(power_analysis_skew)-2]
            # diff_kurt = power_analysis_skew[len(power_analysis_kurt)-1] - power_analysis_skew[len(power_analysis_kurt)-2]
            #
            # if max([diff_skew,diff_kurt]) > epsilon :

            print(f"\n\ndifference in {dist_names[i]}-{dist_names[j]}: \nSample size: {sample_size}\n")
            ci_diff = skew_kurtosis_sampling_mean_diff_ci(dist_obj[i], dist_obj[j],param_number_samples=5000,param_n_size=sample_size)
            print(f"skew: {ci_diff[0]}\nkurtosis: {ci_diff[1]}\nPower_diff_skew: {ci_diff[2]}\nPower_diff_kurt: {ci_diff[3]}")
            power_analysis_skew.append(ci_diff[2])
            power_analysis_kurt.append(ci_diff[3])

            #else: break

        sample_list_np = np.array(sample_list)
        x_reg_linespace = np.linspace(30, 600)
        p_skew_np = np.array(power_analysis_skew)
        p_kurt_np = np.array(power_analysis_kurt)


        print(f"\np_skew: {p_skew_np}\np_kurt: {p_kurt_np}")

        lin_reg_skew, lin_reg_kurt = stats.linregress(sample_list_np,p_skew_np),stats.linregress(sample_list_np,p_kurt_np)



        expo_lin_reg_skew, expo_lin_reg_kurt = (stats.linregress(sample_list_np, np.log(p_skew_np)),
                                                  stats.linregress(sample_list_np, np.log(p_kurt_np))
                                                  )
        power_lin_reg_skew, power_lin_reg_kurt = (stats.linregress(np.log(sample_list_np), np.log(p_skew_np)),
                                                stats.linregress(np.log(sample_list_np), np.log(p_kurt_np))
                                                )

        # graphing power curve (power vs sample size)

        for data,lin_reg,expo_lin_reg,power_lin_reg in zip(
            [p_skew_np,p_kurt_np],
            [lin_reg_skew,lin_reg_kurt],
            [expo_lin_reg_skew,expo_lin_reg_kurt],
            [power_lin_reg_skew,power_lin_reg_kurt],
        ) :
            plt.figure()
            plt.plot(sample_list_np,data,'o',color='orange')
            #plt.plot(x_reg_linespace, lin_reg.slope*x_reg_linespace + lin_reg.intercept,color='red')
            #plt.plot(x_reg_linespace, expo_lin_reg.slope*x_reg_linespace + expo_lin_reg.intercept,color='blue')
            #plt.plot(x_reg_linespace, power_lin_reg.slope*x_reg_linespace + power_lin_reg.intercept,color='green')
            plt.show()







