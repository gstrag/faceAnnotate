import os
import pandas as pd
import json


img_folder = 'test/'
os.makedirs('results/', exist_ok=True)

filenames = os.listdir(img_folder)

additional_args = {
    'FairFace' : '',
    'DMUE': '',
    '3DDFA_V2': " --onnx --show_flag=false -o pose"
}

main_name = {
    'FairFace': 'main.py',
    'DMUE': 'inference.py',
    '3DDFA_V2': 'demo.py -f'
}

systems_used = list(additional_args.keys())

i = 0
subjects = {}
for filename in filenames:
    print("Processing {}. Completed %.2f%%".format(filename) % (100 * i / len(filenames)), end='\r')
    for system in systems_used:
        cmd = "python {}/{} ../".format(system, main_name[system]) + img_folder + filename + additional_args[system]
        os.system(cmd)
    df = pd.DataFrame()
    dfs = []
    for system in systems_used:
        dfs.append(pd.read_csv(system + '/test_outputs.csv'))
    dictio = {}
    for item in dfs:
        for column in item.columns:
            dictio[column] = item[column].values[0]
    subjects[filename] = dictio
    i = i + 1


with open('results/results.json', 'w') as fp:
    json.dump(subjects, fp, indent=4)
