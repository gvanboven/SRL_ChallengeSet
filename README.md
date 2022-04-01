# SRL Challenge Set

This repository contains a challenge set implemented using CheckList (Ribeiro et al., 2020) for evaluating Semantic Role Labelling (SRL) systems. This set allows for a systematic analysis of model performance and targets six linguistic capabilities which lie at the core of SRL: 
* Patient recognition;
* Intrument-patient description disambiguation;
* Agent recognition over small and large predicate distances;
* Manner recognition;
* An invariance test for negation;
* Location-agent disambiguation.

Each test contains 2 to 27 sentence sructures, for each of which are created 100 test cases using the CheckList templates. The complete set of test cases can be found in the folder `JSON_test_and_predict_files`. 

In the current folder six jupyter notebooks can be found, in which I implement each of the tests. These notebooks were based on code provided by Pia Sommerauer and the CheckList tutorials. In these notebooks, I compare two SRL models as a case study : the LSTM based AllenNLP SRL model (Stanovsky et al., 2018) and the BERT based AllenNLP SRL BERT model (Shi and Lin, 2019). The complete model outputs can also be found in the `JSON_test_and_predict_files` folder. 

In the case study, I am able to identify three major limitations of the models: 
* Both models are unable to disambiguate between instruments and patient descriptions, 
* Both models fail to classify constituents that are positioned further away from the predicate,
* The LSTM based model has a lower performance for non-English names, which indicates unfairness.

For a full description of the challenge set and the case study, please refer to the report: `SRL_Challenge_Set_Report.pdf`

This project was carried out for the course NLP Technologies at the Vrije Universiteit Amsterdam, thought by Pia Sommerauer, Antske Fokkens and Jose Angel Daza Arevalo. 

References:      
Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with checklist. _arXiv preprint arXiv:2005.04118_.     
Peng Shi and Jimmy Lin. 2019. Simple bert models for relation extraction and semantic role labeling. _arXiv preprint arXiv:1904.05255_.      
Gabriel Stanovsky, Julian Michael, Luke Zettlemoyer, and Ido Dagan. 2018. Supervised open information extraction. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 885–895_.


