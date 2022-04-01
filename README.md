# SRL Challenge Set

This repository contains a challenge set implemented using CheckList (Ribeiro et al., 2020) for evaluating Semantic Role Labelling (SRL) systems. This set allows for a systematic analysis of model performance and targets six linguistic capabilities which lie at the core of SRL: 
* patient recognition
* intrument-patient distriction disambiguation
* agent recognition over small and large predicate distances
* manner recognition
* an invariance test for negation
* location-agent disambiguation

Each test contains 2 to 27 sentence sructures, for which are each created 100 test cases using the CheckList templates. The complete set of test cases can be found in the folder `JSON_test_and_predict_files`. 

In the current folder six jupyter notebooks can be found, in which I implement each of the tests. These notebooks were based on code provided by Pia Sommerauer and the CheckList tutorials. In these notebooks, I compare two SRL models as a case study : the LSTM based AllenNLP SRL model (Stanovsky et al., 2018) and the BERT based AllenNLP SRL BERT model (Shi and Lin, 2019). The model complete model outputs can also be found in the `JSON_test_and_predict_files` folder. 

In the case study, I am able to identify three major limitations: 
* both models are unable to disambiguate between instruments and patient descriptions, 
* both models fail to classify constituents that are positioned further away from the predicate 
* the LSTM based model has a lower performance for non-English names, which indicates unfairness

For a full description of the challenge set and the case study, please refer to the report: `SRL_Challenge_Set_Report.pdf`

This project was carried out for the course NLP Technologies at the Vrije Universiteit Amsterdam, thought by Pia Sommerauer, Antske Fokkens and Jose Angel Daza Arevalo. 