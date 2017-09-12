#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math
import sys
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
POIs = [person for person in enron_data if enron_data[person]["poi"]==1]
for poi in POIs:
    print(poi, math.log(enron_data[poi]['total_payments'], 10))

data = featureFormat(['total_payments'], enron_data)
