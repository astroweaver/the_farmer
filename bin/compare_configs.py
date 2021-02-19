# Compare two Farmer config files to determine the differences 
# assumes the config files are either within the current working directory or within config/
# LMZ

import sys 
import os 
import importlib
import numpy as np 
sys.path.insert(0, os.path.join(os.getcwd(), 'src')) 
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))  

name_conf1 = sys.argv[1]
name_conf2 = sys.argv[2]

conf1 = importlib.import_module(name_conf1)
conf2 = importlib.import_module(name_conf2)

config1_keys = dir(conf1)
config2_keys = dir(conf2)

warnings = []

print()
print(f"............ Comparing configs ............")

print(f"{name_conf1} length: {len(config1_keys)}")
print(f"{name_conf2} length: {len(config2_keys)}")
print()

if len(config1_keys) != len(config2_keys):
    warnings.append("- Configs have different lengths \n") 

in1_notin2 = list(np.array(config1_keys)[~np.isin(config1_keys,config2_keys)])
in2_notin1 = list(np.array(config2_keys)[~np.isin(config2_keys,config1_keys)])

if len(in2_notin1) > 0:
    warnings.append(f"- {name_conf1} is missing the following params included in {name_conf2}: {in2_notin1} \n")

if len(in1_notin2) > 0:
    warnings.append(f"- {name_conf2} is missing the following params included in {name_conf1}: {in1_notin2} \n")

matching_params1 = list(np.array(config1_keys)[np.isin(config1_keys,config2_keys)])
matching_params2 = list(np.array(config2_keys)[np.isin(config2_keys,config1_keys)])
matching_params1.extend(matching_params2)

matching_params = list(set(matching_params1))

for m in matching_params:
    val1,val2 = conf1.__dict__[m], conf2.__dict__[m]
    if ("__" not in m) and ("WORKING" not in m):
        if val1 != val2:
            warnings.append(f"- Param value < {m} > differs: {name_conf1} has {m} = {val1}, {name_conf2} has {m} = {val2}")

if len(warnings) > 0:
    print("!!! WARNINGS !!! \n")
    [print(w) for w in warnings]

else:
    print("No warnings found")