import os
import math
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import optimize, signal
from scipy.signal import find_peaks_cwt

from lmfit import models


### IMPORT DATA ###

def addfiletolist (filename, sample):

    list_theta_sample = []
    list_counts_sample = []
    


    fobj = open(filename, "r", errors='ignore')

    np.seterr(divide='ignore', invalid='ignore')


    for number_of_line, line in enumerate(fobj):
        splitted_line = line.split()
        if number_of_line == 0 or line.startswith("#") or len(splitted_line) == 0:
           continue
        splitted_line = line.split()
        list_theta_sample.append(float(splitted_line[0].replace(",",".")))  
        list_counts_sample.append(float(splitted_line[1].replace(",",".")))    
          


    fobj.close()
    list_theta_sample, list_counts_sample = np.array(list_theta_sample), np.array(list_counts_sample) #, np.array(list_theta_model_sample) 
    return list_theta_sample, list_counts_sample 

list_theta_cf14, list_counts_cf14 = addfiletolist("/mnt/c/Users/Lenovo/desktop/studium/bachelorarbeit/XRD/MD.02.2.4.dat", 1)

### SPEC ###

spec = {
    'x': list_theta_cf14,
    'y': list_counts_cf14,
    'model': [
        {'type': 'VoigtModel'},
        # {'type': 'VoigtModel'},
        # {'type': 'VoigtModel'},
        # {'type': 'VoigtModel'},
        # {'type': 'GaussianModel'},
        # {'type': 'GaussianModel'},
        # {'type': 'GaussianModel'},
        # {'type': 'GaussianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'LorentzianModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
        # {'type': 'PseudoVoigtModel'},
    ]
}

### Generate Models ###

def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel', 'PseudoVoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

generate_model(spec)

with open ('Test/spec3.dat', 'a') as q:
            print(spec, file=q)



### Spec-Update via Peakanalyse ###

def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        print('ha')
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel', 'PseudoVoigtModel']:
            print('hu')
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
                print('hi')
            else:
                model['params'] = params
                print('ho')
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies

update_spec_from_peaks(spec, [0], peak_widths=(15,))

with open ('Test/spec3.dat', 'a') as q:
            print(spec, file=q)



### Fit mit den neuen Initial guesses ###

model, params = generate_model(spec)
output = model.fit(spec['y'], params, x=spec['x'])
fig, gridspec = output.plot(data_kws={'markersize':  1})
fig.savefig('Test/test1.png')
