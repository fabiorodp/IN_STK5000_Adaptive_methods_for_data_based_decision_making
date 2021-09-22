# **Project 1 - IN-STK5000/9000 - Autumn21**

## **Students:**
Fábio Rodrigues Pereira : fabior@uio.no

Nicholas Walker : ...@uio.no

Aurora Poggi: ...@uio.no


## **Data source:**
[observation_features.csv](https://raw.githubusercontent.com/fabiorodp/IN_STK5000_Adaptive_methods_for_data_based_decision_making/main/project1/data/observation_features.csv)

[treatment_features.csv](https://raw.githubusercontent.com/fabiorodp/IN_STK5000_Adaptive_methods_for_data_based_decision_making/main/project1/data/treatment_features.csv)

[treatment_action.csv](https://raw.githubusercontent.com/fabiorodp/IN_STK5000_Adaptive_methods_for_data_based_decision_making/main/project1/data/treatment_actions.csv)

[treatment_outcome.csv](https://raw.githubusercontent.com/fabiorodp/IN_STK5000_Adaptive_methods_for_data_based_decision_making/main/project1/data/treatment_outcomes.csv)

## **Description of the features:**
Symptoms (10 bits): Covid-Recovered, Covid-Positive, No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death

Age (integer)

Gender (binary)

Genome (128 bits)

Comorbidities (6 bits): Asthma, Obesity, Smoking, Diabetes, Heart disease, Hypertension

Vaccination status (3 bits): 0 for unvaccinated, 1 for receiving a specific vaccine

## **Description of the actions and outcomes:**
Treatment (k bits): Multiple simultaneous treatments are possible.

Post-Treatment Symptoms (8 bits): No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death.


## **Tasks:**
1. Perform the following modelling tasks for the observational variables:

   (a) Predicting the effect of genes and/or age/comorbidities on symptoms.

   (b) Estimating the efficacy of vaccines.
   
   (c) Estimating the probability of vaccination side-effects.

2. Model the effect of treatments on alleviating symptoms (e.g. preventing death)

3. Although this data involves people, you are not required to do a formal study of fairness or privacy in this part of the project. However, you are encouraged to describe verbally what possible issues are.

The files are: 'observation_features.csv'

For each task, you need to select subsets of the data. Since this is only an observational study, you need to e.g. separate the vaccinated from the unvaccinated. It is crucial that your modelling study is reproducible, in the following senses:

1. You should first clearly define the problem you want to investigate . You must define the problem formally. This means that there should be a mathematical formula or computational procedure through which the analysis can be performed automatically. (Example. If you want to find which genes cause bad outcomes in covid patients, you should define an appropriate mathematical criterion for what the most important genes are, relating to the distribution of outcomes and genes. You should also precisely define an automatic method for selecting such genes (using an existing one is OK of course). Simply making an 'important features' plot and picking the top 5 is inadequate)
2. Somebody could reproduce the technical aspects of your work. So you must explain and justify why you are using a specific methodology. (In the above example, you should test the methodology for selecting genes with synthetic data where you know which features affect the outcomes.)
3. Somebody could replicate the study from scratch with a new dataset and most likely reach the same conclusion. So you must test your methodology in a pipeline to make sure it works as expected. (Beyond the simulation study, tuning your method on the actual data through bootstrapping or cross-validation is going to be important)

For that reason, you MUST communicate your uncertainties about your conclusions very clearly. It is also important to set up a robust pipeline for your analysis. In particular, you must take steps to ensure that you do not uncover spurious correlations. A simulation study will be great for achieving that, i.e. to make sure it works.

Submit a final report about your project, either as a standalone PDF or as a jupyter notebook. For this, you can imagine playing the role of an analyst who submits a possible decision rule to the authorities for a vaccine or treatment recommendation.

Note: You should consider the datasets ephemeral. Do not get too invested in their details, just make sure that you implement a good methodology so that you can get robust results even with new datasets (which may have different properties). For this particular data, I have checked that it makes sense at a high level, but perhaps something weird is there if you dig deeply enough. If so, let me know.