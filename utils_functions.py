import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
import json


###########################################
########## CORE  FUNCTIONALITIES ##########
###########################################

def get_arg(pred, verb_number, arg_target='ARG1'):
    """ 
    Helper function to extract target argument
    :param pred: the model prediction
    :param verm_number: the verb number of the predicate of interest
    :param arg_target: the name of the argument of interest
    :type pred: dict
    :type verb_number: int
    :type arg_target: string

    :returns arg_set: the set of tokens that are classified to be this argument
    :type arg_set: set

    """
    # we assume one predicate:
    predicate_arguments = pred['verbs'][verb_number]
    words = pred['words']
    tags = predicate_arguments['tags']
    
    arg_list = []
    for t, w in zip(tags, words):
        arg = t[2:]
        if arg == arg_target:
            arg_list.append(w)
    arg_set = set(arg_list)
    return arg_set


def get_model_results(model, n, sentence, capability, testcase_name):
    """
    Extracts the predictions of a given model for a given sentence
    And returns a dict with this info, which is to be stored in an output file

    :param model: a pretrained srl model
    :param n: the number of the current data sample
    :param sentence: the current data sample sentence
    :param capability: the name of the linguistic capability of interest
    :param testcase_name: the name of the current test
    :type model : allennlp_models.structured_prediction.predictors.srl
    :type n: int
    :type sentence: string
    :type capability: string
    :type testcase_name: string

    :returns results: dict with all relevant info to store in output file
    :type results: dict
    """
    results = {'sentence' : sentence,
               'capability' : capability,
               'testcase_name' : testcase_name,
               'preds' : model.results['preds'][n],
               'confs': model.results['confs'][n],
               'passed' : bool(model.results['passed'][n])}
    return results

def extract_data_and_predictions(t, capability, testcase_name, test_srl, test_srlbert, test_data, SRL_predictions, SRLBERT_predictions):
    """ 
    Function that extracts all relevant test data and predictions information that is to be stored in the output files

    :param t: dict containing all test case information
    :param capability: the name of the linguistic capability of interest
    :param testcase_name: the name of the current test
    :param test_srl: predictions for the first SRL model
    :param test_srlbert: predictions for the second SRL model
    :param test_data: output list in which all test cases should be stored
    :param SRL_predictions: output list in which all predictions of the first model should be stored
    :param SRLBERT_predictions: output list in which all predictions of the second model should be stored
    :type t: dict
    :type capability: string
    :type testcase_name: string
    :type test_srl: dict
    :type test_srlbert: dict
    :type test_data: list
    :type SRL_predictions: list
    :type SRLBERT_predictions: list

    :returns test_data: output list to which new test cases are added
    :returns SRL_predictions: output list to which new predictions of the first model are added
    :returns SRLBERT_predictions: output list to which new predictions of the second model are added
    """
    for n, sentence in enumerate(t['data']):
        #extract input sentence info
        input_item = {'sentence' : sentence,
                        'meta' : t['meta'][n],
                        'capability' : capability,
                        'testcase_name': testcase_name}
        #extract predictions info for the two models
        srl_prediction =  get_model_results(test_srl, n, sentence, capability, testcase_name)
        srlbert_prediction = get_model_results(test_srlbert, n, sentence, capability, testcase_name)
        #save extracted info
        test_data.append(input_item)
        SRL_predictions.append(srl_prediction)
        SRLBERT_predictions.append(srlbert_prediction)
    return test_data, SRL_predictions, SRLBERT_predictions


def predict_and_store(t, capability, testcase_name, expect, formattype, predict_srl, predict_srlbert, test_data, SRL_predictions, SRLBERT_predictions):
    """ 
    Function that creates test cases given a template, makes predictions for the given models and stores the test cases as well as the predictions

    :param t: dict containing all test case information
    :param capability: the name of the linguistic capability of interest
    :param testcase_name: the name of the current test
    :param expect: function that checks if the argument of interest is predicted as expected
    :param formattype: function that creates the correct formatting for the test
    :param predict_srl: first pretrained SRL model to be tested
    :param predict_srlbert: second pretrained SRL model to be tested
    :param test_data: output list in which all test cases should be stored
    :param SRL_predictions: output list in which all predictions of the first model should be stored
    :param SRLBERT_predictions: output list in which all predictions of the second model should be stored
    :type t: dict
    :type capability: string
    :type testcase_name: string
    :type expect: python function
    :type formattype: python function
    :type predict_srl: allennlp_models.structured_prediction.predictors.srl
    :type predict_srlbert: allennlp_models.structured_prediction.predictors.srl
    :type test_data: list
    :type SRL_predictions: list
    :type SRLBERT_predictions: list

    :returns test_data: output list to which new test cases are added
    :returns SRL_predictions: output list to which new predictions of the first model are added
    :returns SRLBERT_predictions: output list to which new predictions of the second model are added
    """
    #test the srl model
    print('SRL')
    test_srl = MFT(**t, expect=expect)
    test_srl.run(predict_srl)
    test_srl.summary(format_example_fn=formattype)

    #test the srl bert model
    print('SRL BERT')
    test_srlbert = MFT(**t, expect=expect)
    test_srlbert.run(predict_srlbert)
    test_srlbert.summary(format_example_fn=formattype)

    #store samples and predictions
    test_data, SRL_predictions, SRLBERT_predictions = extract_data_and_predictions(t, capability, testcase_name, test_srl, \
                                                                test_srlbert, test_data, SRL_predictions, SRLBERT_predictions)
    return test_data, SRL_predictions, SRLBERT_predictions

def store_data(path, data, new_file=True):
    """
    Function that saves a given list to a json file on the given path

    :param path: path to a .json file to store data in
    :param data: list containing information to be stored
    :param new_file: setting indicating whether previous information in the file should be deleted (True) or not (False)
    :type path: string
    :type data: list
    :type new_file: boolean
    """
    #if there already is content in the file, make sure we do not lose it. 
    if new_file == False:
        with open(path, "r") as file:
            old_data = json.load(file)

        old_data.append(data)
        data = old_data

    with open(path, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


###########################################
############### FORMATTING ################
###########################################


# Helper function to display failures for the first identified verb
def format_srl_verb0(x, pred, conf, label=None, meta=None):

    predicate_structure = pred['verbs'][0]['description']
        
    return predicate_structure

# Helper function to display failures for the second identified verb
def format_srl_verb1(x, pred, conf, label=None, meta=None):

    predicate_structure = pred['verbs'][1]['description']
        
    return predicate_structure

# Helper function to display failures for the third identified verb
def format_srl_verb2(x, pred, conf, label=None, meta=None):

    predicate_structure = pred['verbs'][2]['description']
        
    return predicate_structure

###########################################
########## ARGUMENT RECOGNITION ###########
###########################################

# The functions below are used to control whether the correct tokens are identified as the arguments of interest

######### PATIENT RECOGNITION #############

def found_arg1_people_verb0(x, pred, conf, label=None, meta=None):
    # people should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']
        
    people = [meta['first']]
    people.extend(lastname)
    people = set(people)
    
    arg_1 = get_arg(pred, 0, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg1_people_verb1(x, pred, conf, label=None, meta=None):
    # people should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']
        
    people = [meta['first']]
    people.extend(lastname)
    people = set(people)
    arg_1 = get_arg(pred, 1, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg1_people_verb2(x, pred, conf, label=None, meta=None):
    # people should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']
        
    people = [meta['first']]
    people.extend(lastname)
    people = set(people)
    arg_1 = get_arg(pred, 2, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg1_doctor_verb0(x, pred, conf, label=None, meta=None):
    
    # 'doctor' + name should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']

    doctor = ['Doctor', meta['first']]
    doctor.extend(lastname)
    doctor = set(doctor)
    
    arg_1 = get_arg(pred, 0, arg_target='ARG1')

    if arg_1 == doctor:
        pass_ = True
    else:
        #print(doctor, arg_0)
        pass_ = False
    return pass_

def found_arg1_doctor_verb1(x, pred, conf, label=None, meta=None):
    
    # 'doctor' + name should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']

    doctor = ['Doctor', meta['first']]
    doctor.extend(lastname)
    doctor = set(doctor)
    
    arg_1 = get_arg(pred, 1, arg_target='ARG1')

    if arg_1 == doctor:
        pass_ = True
    else:
        #print(doctor, arg_0)
        pass_ = False
    return pass_

def found_arg1_doctor_verb2(x, pred, conf, label=None, meta=None):
    
    # 'doctor' + name should be recognized as arg1
    
    #make sure we correctly deal with lastnames of longer than one word    
    if len(meta['last']) > 1:
        lastname = meta['last'].split()
    else:
        lastname = meta['last']

    doctor = ['Doctor', meta['first']]
    doctor.extend(lastname)
    doctor = set(doctor)
    
    arg_1 = get_arg(pred, 2, arg_target='ARG1')

    if arg_1 == doctor:
        pass_ = True
    else:
        #print(doctor, arg_0)
        pass_ = False
    return pass_

######### MANNER RECOGNITION #############

def found_arg_manner_verb0(x, pred, conf, label=None, meta=None):
    
    # the manner should be recognized as manner
    manner = set([meta['manner']])

    arg_mnr = get_arg(pred, 0, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg_mannerlong_verb0(x, pred, conf, label=None, meta=None):
    
    # 'the manner + {manner}' should be recognized as manner
        
    manner = set(['The', 'manner', meta['manner']])
    
    arg_mnr = get_arg(pred, 0, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg_inamanner_verb0(x, pred, conf, label=None, meta=None):
    # 'in a {manner} manner' should be recognized as manner
    manner = set(['in', 'a', 'manner', meta['manner']])
    
    arg_mnr = get_arg(pred, 0, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg_eversomanner_verb0(x, pred, conf, label=None, meta=None):
    #'Ever so {manner}' should be recognized as manner
    manner = set(['Ever', 'so', meta['manner']])

    arg_mnr = get_arg(pred, 0, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg_manner_verb1(x, pred, conf, label=None, meta=None):
    
    ## the manner should be recognized as manner 
    manner = set([meta['manner']])
    
    arg_mnr = get_arg(pred, 1, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg_eversomanner_verb1(x, pred, conf, label=None, meta=None):
    
    # 'Ever so {manner}' should be recognized as manner    
    manner = set(['Ever', 'so', meta['manner']])
    
    arg_mnr = get_arg(pred, 1, arg_target='ARGM-MNR')

    if arg_mnr == manner:
        pass_ = True
    else:
        pass_ = False
    return pass_

######### AGENT RECOGNITION #############

def found_arg0_verb0(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg0
        
    people = set([meta['first_name']])

    arg0 = get_arg(pred, 0, arg_target='ARG0')

    if arg0 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg0_verb1(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg0
        
    people = set([meta['first_name']])
    
    arg0 = get_arg(pred, 1, arg_target='ARG0')

    if arg0 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_byarg0_verb1(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg0
        
    people = set(['by', meta['first_name']])
    

    arg0 = get_arg(pred, 1, arg_target='ARG0')

    if arg0 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

########## NEGATION EXPERIMENTS ##########

def found_arg1_verb0(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg1
        
    people = set([meta['first']])

    arg0 = get_arg(pred, 0, arg_target='ARG1')

    if arg0 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg1_verb1(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg1
        
    people = set([meta['first']])

    arg0 = get_arg(pred, 1, arg_target='ARG1')

    if arg0 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg2_verb0(x, pred, conf, label=None, meta=None):
    
    # the instrument should be recognized as arg2
        
    instrument = set(['with', 'a', meta['instrument']])

    arg2 = get_arg(pred, 0, arg_target='ARG2')

    if arg2 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg2_verb1(x, pred, conf, label=None, meta=None):
    
    # the instrument should be recognized as arg2
        
    instrument = set(['with', 'a', meta['instrument']])

    arg2 = get_arg(pred, 1, arg_target='ARG2')

    if arg2 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_argloc_verb0(x, pred, conf, label=None, meta=None):
    
    # the location should be recognized as argm-loc
    loc = set(meta['location'].split())

    argloc = get_arg(pred, 0, arg_target='ARGM-LOC')

    if argloc == loc:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_argloc_verb1(x, pred, conf, label=None, meta=None):
    
    # the location should be recognized as argm-loc
    loc = set(meta['location'].split())

    argloc = get_arg(pred, 1, arg_target='ARGM-LOC')

    if argloc == loc:
        pass_ = True
    else:
        pass_ = False
    return pass_

######### LOCATION RECOGNITION

def found_location_arg0_verb0(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg0
    if 'country' in meta:

        if len(meta['country'])> 1:
            loc = meta['country'].split()
        else:
            loc = meta['country']
    elif 'city' in meta:
        if len(meta['city'])> 1:
            loc = meta['city'].split()
        else:
            loc = meta['city']
    loc = set(loc)
    

    arg0 = get_arg(pred, 0, arg_target='ARG0')

    if arg0 == loc:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_location_arg0_verb1(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg0
    if 'country' in meta:

        if len(meta['country'])> 1:
            loc = meta['country'].split()
        else:
            loc = meta['country']
    elif 'city' in meta:
        if len(meta['city'])> 1:
            loc = meta['city'].split()
        else:
            loc = meta['city']
    loc += ['by']
    loc = set(loc)
    

    arg0 = get_arg(pred, 1, arg_target='ARG0')

    if arg0 == loc:
        pass_ = True
    else:
        pass_ = False
    return pass_


def found_location_argloc_verb1(x, pred, conf, label=None, meta=None):
    
    # the location should be recognized as argm-loc
    if len(meta['country'])> 1:
        country = meta['country'].split()
    else:
        country = meta['country']
    country += ['in']
    loc = set(country)

    argloc = get_arg(pred, 1, arg_target='ARGM-LOC')
    
    if argloc == loc:
        pass_ = True
    else:
        pass_ = False
    return pass_

######### INSTRUMENT DISAMBIGUATION #########

def found_door_arg1_verb0(x, pred, conf, label=None, meta=None):
    
    # the instrument should be recognized as arg2
    patient = set(['the', 'door', 'with', meta['description']])

    arg1 = get_arg(pred, 0, arg_target='ARG1')

    if arg1 == patient:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_door_arg2_verb0(x, pred, conf, label=None, meta=None):
    
    # the instrument should be recognized as arg2
    instrument = set(['with', 'the', meta['instrument']])

    arg2 = get_arg(pred, 0, arg_target='ARG2')

    if arg2 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_door_arg2_verb1(x, pred, conf, label=None, meta=None):
    
    # the instrument should be recognized as arg2
    instrument = set(['with', 'the', meta['instrument']])

    arg2 = get_arg(pred, 1, arg_target='ARG2')

    if arg2 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_