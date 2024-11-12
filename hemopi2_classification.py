import argparse
import warnings
import os
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import re
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification
warnings.filterwarnings("ignore")

# Function to check the sequence
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>' + name)
        seq.append(sequence)
    if len(seqid) == 0:
        f = open(file, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq) + 1):
            seqid.append(">Seq_" + str(i))
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1, df2

# Function to check the length of sequences
def lenchk(file1, wd):
    cc = []
    df1 = file1
    df1.columns = ['seq']
    for i in range(len(df1)):
        if len(df1['seq'][i]) > 40:
            cc.append(df1['seq'][i][0:40])
        else:
            cc.append(df1['seq'][i])
    df2 = pd.DataFrame(cc)
    output_len_file = os.path.join(wd, 'out_len')
    df2.to_csv(output_len_file, index=None, header=None)
    df2.columns = ['Seq']
    return df2, output_len_file

# Function to read and implement the model
def ML_run(file1, out, threshold):
    a = []
    df = pd.read_csv(file1)  # Read the input CSV file into a DataFrame
    clf = pickle.load(open('./model/hemopi2_ml_clf.sav', 'rb'))  # Load the pre-trained model
    data_test = df  # Use all features in df for testing
    X_test = data_test
    y_p_score1 = clf.predict_proba(X_test)  # Make predictions
    y_p_s1 = y_p_score1.tolist()  # Convert predictions to a list
    a.extend(y_p_s1)  # Append predictions to the list
    df_predictions = pd.DataFrame(a)  # Create a DataFrame from the list of predictions
    df1 = df_predictions.iloc[:, 1].round(3)  # Round predictions to 3 decimal places
    df2 = pd.DataFrame(df1)  # Convert to DataFrame
    df2.columns = ['ML Score']  # Rename column to 'ML Score'
    # Assign Hemolytic or Non-Hemolytic based on the threshold
    df2['Prediction'] = ['Hemolytic' if score >= threshold else 'Non-Hemolytic' for score in df2['ML Score']]
    df2.to_csv(out, index=None)  # Save the final DataFrame to the specified output CSV file
    return df2  # Return the DataFrame containing the predictions

################################Functions to read and implement the esm model############################
# Define a function to process sequences
def process_sequences(df, df_2):
    df_2.rename(columns={0: 'SeqID'}, inplace=True)
    outputs = [(df_2.loc[index, 'SeqID'], row['seq']) for index, row in df.iterrows()]
    return outputs

# Function to prepare dataset for prediction
def prepare_dataset(sequences, tokenizer):
    seqs = [seq for _, seq in sequences]
    inputs = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    return inputs

# Function to write output to a file
def write_output(output_file, sequences, predictions, threshold):
    with open(output_file, 'w') as f:
        f.write("SeqID,Seq,ML Score,Prediction\n")
        for (seq_id, seq), pred in zip(sequences, predictions):
            final_pred = "Hemolytic" if pred >= threshold else "Non-Hemolytic"
            f.write(f"{seq_id},{seq},{pred:.4f},{final_pred}\n")

# Function to make predictions
def make_predictions(model, inputs, device):
    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs

# Main function for ESM model integration
def run_esm_model(dfseq , df_2, output_file, threshold):
    # Process sequences from the DataFrame
    sequences = process_sequences(dfseq, df_2)

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare inputs for the model
    inputs = prepare_dataset(sequences, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make predictions
    predictions = make_predictions(model, inputs, device)

    # Write the output to a file
    write_output(output_file, sequences, predictions, threshold)

#################################################################################################################

# Function for generating pattern of a given length (protein scanning)
def seq_pattern(file1, file2, num):
    df1 = file1
    df1.columns = ['Seq']
    df2 = file2
    df2.columns = ['Name']
    cc = []
    dd = []
    ee = []
    ff = []
    gg = []
    for i in range(len(df1)):
        for j in range(len(df1['Seq'][i])):
            xx = df1['Seq'][i][j:j+num]
            if len(xx) == num:
                cc.append(df2['Name'][i])
                dd.append('Pattern_' + str(j + 1))
                ee.append(xx)
                ff.append(j + 1)  # Start position (1-based index)
                gg.append(j + num)  # End position (1-based index)
    df3 = pd.concat([pd.DataFrame(cc), pd.DataFrame(dd), pd.DataFrame(ff), pd.DataFrame(gg), pd.DataFrame(ee)], axis=1)
    df3.columns = ['SeqID', 'Pattern ID', 'Start', 'End', 'Seq']
    return df3


def class_assignment(file1, thr, out, wd):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0, len(df1)):
        if df1['ML Score'][i] >= float(thr):
            cc.append('Hemolytic')
        else:
            cc.append('Non-Hemolytic')
    df1['Prediction'] = cc
    df1 = df1.round(3)
    output_file = os.path.join(wd, out)
    df1.to_csv(output_file, index=None)

def emb_process(file, wd):
    df = pd.read_csv(file)
    df.insert(0, 'seq_ID', ['seq_' + str(i) for i in range(1, len(df) + 1)])
    ss = df[['seq_ID']]
    df2 = df.drop(['seq_ID'], axis=1)
    colNumber = df2.shape[1]
    headerRow = []
    for i in range(colNumber):
        headerRow.append('prot' + str(i))
    df2.columns = headerRow
    df3 = pd.concat([ss, df2], axis=1)
    output_file = os.path.join(wd, 'emb_processed.csv')
    df3.to_csv(output_file, index=None)
    return df3

def generate_mutant(original_seq, residues, position):
    std = "ACDEFGHIKLMNPQRSTVWY"
    if all(residue.upper() in std for residue in residues):
        if len(residues) == 1:
            mutated_seq = original_seq[:position-1] + residues.upper() + original_seq[position:]
        elif len(residues) == 2:
            mutated_seq = original_seq[:position-1] + residues[0].upper() + residues[1].upper() + original_seq[position+1:]
        else:
            print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
            return None
    else:
        print("Invalid residues. Please enter one or two of the 20 essential amino acids.")
        return None
    return mutated_seq

def generate_mutants_from_dataframe(df, residues, position):
    mutants = []
    for index, row in df.iterrows():
        original_seq = row['Seq']
        mutant_seq = generate_mutant(original_seq, residues, position)
        if mutant_seq:
            mutants.append((original_seq, mutant_seq, position))
    return mutants

# Function for generating all possible mutants
def all_mutants(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = file2
    df2.columns = ['Name']
    df1 = file1
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    #dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j)
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['SeqID','Mutant_ID','Seq']
    return xx

####################################################### MERCI ##########################################

def MERCI_Processor_p(wd, merci_file, merci_processed, name):
    hh, jj, kk, qq = [], [], [], []
    filename = f"{wd}/{merci_file}"
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'

    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0:
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    
    if not hh:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('Non-Hemolytic')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x: x.strip(), l))[1:])
        df1.columns = ['SeqID']
        df1['SeqID'] = df1['SeqID'].str.strip('(')
        df1[['Seq', 'PHits']] = df1['SeqID'].str.split("(", expand=True)
        df2 = df1[['Seq', 'PHits']].copy()
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True, inplace=True)
        
        for j in oo:
            if j in df2['Seq'].values:
                jj.append(j)
                qq.append(df2.loc[df2['Seq'] == j, 'PHits'].values)
                kk.append('Hemolytic')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Non-Hemolytic')
    
    df3 = pd.concat([pd.DataFrame(jj), pd.DataFrame(qq), pd.DataFrame(kk)], axis=1)
    df3.columns = ['SeqID', 'PHits', 'Prediction']
    df3.to_csv(f"{wd}/{merci_processed}", index=None)

def Merci_after_processing_p(wd, merci_processed, final_merci_p):
    df5 = pd.read_csv(f"{wd}/{merci_processed}")
    df5 = df5[['SeqID', 'PHits']].copy()
    kk = []
    for i in range(len(df5)):
        if int(df5.at[i, 'PHits']) > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score Pos"] = kk
    df5 = df5[['SeqID', 'MERCI Score Pos']]
    df5.to_csv(f"{wd}/{final_merci_p}", index=None)

def MERCI_Processor_n(wd, merci_file, merci_processed, name):
    hh, jj, kk, qq = [], [], [], []
    filename = f"{wd}/{merci_file}"
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'

    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0:
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    
    if not hh:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['1']))
            kk.append('Hemolytic')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x: x.strip(), l))[1:])
        df1.columns = ['SeqID']
        df1['SeqID'] = df1['SeqID'].str.strip('(')
        df1[['Seq', 'NHits']] = df1['SeqID'].str.split("(", expand=True)
        df2 = df1[['Seq', 'NHits']].copy()
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True, inplace=True)
        
        for j in oo:
            if j in df2['Seq'].values:
                jj.append(j)
                qq.append(df2.loc[df2['Seq'] == j, 'NHits'].values)
                kk.append('Non-Hemolytic')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Hemolytic')
    
    df3 = pd.concat([pd.DataFrame(jj), pd.DataFrame(qq), pd.DataFrame(kk)], axis=1)
    df3.columns = ['SeqID', 'NHits', 'Prediction']
    df3.to_csv(f"{wd}/{merci_processed}", index=None)

def Merci_after_processing_n(wd, merci_processed, final_merci_n):
    df5 = pd.read_csv(f"{wd}/{merci_processed}")
    df5 = df5[['SeqID', 'NHits']].copy()
    kk = []
    for i in range(len(df5)):
        if int(df5.at[i, 'NHits']) > 0:
            kk.append(-0.5)
        else:
            kk.append(0)
    df5["MERCI Score Neg"] = kk
    df5 = df5[['SeqID', 'MERCI Score Neg']]
    df5.to_csv(f"{wd}/{final_merci_n}", index=None)

def hybrid(wd, df3, merci_hybrid_p_1, merci_hybrid_n_1, merci_hybrid_p_2, merci_hybrid_n_2, threshold, final_output):
    df6_2 = df3.copy()
    df6_2['SeqID'] = df6_2['SeqID'].str.replace('>', '', regex=False)
    df6 = df6_2[['SeqID', 'Seq', 'ML Score']]
    df4 = pd.read_csv(f"{wd}/{merci_hybrid_p_1}",names= ['SeqID', 'MERCI Score 1 Pos'], header=0, dtype={'Subject': object, 'MERCI Score Pos': np.float64})
    df5 = pd.read_csv(f"{wd}/{merci_hybrid_n_1}",names= ['SeqID', 'MERCI Score 1 Neg'], header=0, dtype={'Subject': object, 'MERCI Score Neg': np.float64})
    df7 = pd.read_csv(f"{wd}/{merci_hybrid_p_2}",names= ['SeqID', 'MERCI Score 2 Pos'], header=0, dtype={'Subject': object, 'MERCI Score Pos': np.float64})
    df8 = pd.read_csv(f"{wd}/{merci_hybrid_n_2}",names= ['SeqID', 'MERCI Score 2 Neg'], header=0, dtype={'Subject': object, 'MERCI Score Neg': np.float64})
    
    df9 = pd.merge(df6, df4, how='outer', on='SeqID')
    df10 = pd.merge(df9, df5, how='outer', on='SeqID')
    df11 = pd.merge(df10, df7, how='outer', on='SeqID')
    df12 = pd.merge(df11, df8, how='outer', on='SeqID')
    
    df12.fillna(0, inplace=True)
    df12['MERCI Score'] = df12[['MERCI Score 1 Pos', 'MERCI Score 1 Neg', 'MERCI Score 2 Pos', 'MERCI Score 2 Neg']].sum(axis=1)
    df12['Hybrid Score'] = df12[['ML Score', 'MERCI Score']].sum(axis=1)
    df12.drop(columns=['MERCI Score 1 Pos', 'MERCI Score 1 Neg', 'MERCI Score 2 Pos', 'MERCI Score 2 Neg'], inplace=True)
    df12 = df12.round(3)
    
    ee = []
    for i in range(len(df12)):
        if df12.at[i, 'Hybrid Score'] > float(threshold):
            ee.append('Hemolytic')
        else:
            ee.append('Non-Hemolytic')
    df12['Prediction'] = ee
    df12.to_csv(f"{wd}/{final_output}", index=None)

print('######################################################################################################')
print('# This program HemoPI2.0 is developed for predicting, desigining and scanning Hemolytic activity of peptides #')
print('##### Prediction of Hemolytic activity of  peptides, developed by Prof G. P. S. Raghava group. #######')
print('######## Please cite: HemoPI2.0; available at https://webs.iiitd.edu.in/raghava/hemopi2/  ############')
print('######################################################################################################')


# Initialize ArgumentParser
parser = argparse.ArgumentParser(description='Please provide following arguments')

# Define arguments
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output", type=str,
                    help="Output: File for saving results by default outfile.csv")
parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4, 5],
                    help="Job Type: 1: Predict, 2: Protein Scanning, 3: Design, 4: Design all possible mutants, 5: Motif Scanning, by default 1")
parser.add_argument("-m", "--model", type=int, choices=[1, 2, 3, 4],
                    help="Model: 1: Random Forest, 2: Hybrid1 (RF+MERCI), 3: ESM2-t6, 4: Hybrid2 (ESM+MERCI) by default 4")
parser.add_argument("-t", "--threshold", type=float,
                    help="Threshold: Value between 0 to 1 by default 0.46 (For RF and Hybrid1) and 0.55 (For ESM and Hybrid2)")
parser.add_argument("-p", "--Position", type=int,
                    help="Position of mutation (1-indexed)")
parser.add_argument("-r", "--Residues", type=str,
                    help="Mutated residues (one or two of the 20 essential amino acids in upper case)")
parser.add_argument("-w","--winleng", type=int, choices =range(8, 21), 
                    help="Window Length: 8 to 20 (scan mode only), by default 8")
parser.add_argument("-wd", "--working", type=str, required=True, help="Working Directory: Location for writing results")
parser.add_argument("-d", "--display", type=int, choices=[1, 2], default=2,
                    help="Display: 1: Hemolytic, 2: All peptides, by default 2")

# Parse arguments
args = parser.parse_args()

# Parameter initialization or assigning variable for command level arguments
Sequence = args.input  #

# Output file
if args.output is None:
    result_filename = "outfile.csv"
else:
    result_filename = args.output

# Job Type
if args.job is None:
    Job = 1
else:
    Job = args.job

# Model
if args.model is None:
    Model = 4
else:
    Model = args.model

if args.Position is None:
    position = 1
else:
    position = args.Position


if args.Residues is None:
    residues = "AA"
else:
    residues = args.Residues




###########Threshold#############
# Threshold
if args.threshold is None:
    if Model in [1, 2]:
        threshold = 0.46
    elif Model in [3, 4]:
        threshold = 0.55
else:
    threshold = args.threshold

# Window Length 
if args.winleng == None:
    Win_len = int(8)
else:
    Win_len = int(args.winleng)

wd = args.working

# Display
dplay = args.display

#------------------ Read input file ---------------------
f=open(Sequence,"r")
len1 = f.read().count('>')
f.close()

with open(Sequence) as f:
        records = f.read()
records = records.split('>')[1:]
seqid = []
seq = []
for fasta in records:
    array = fasta.split('\n')
    name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
    seqid.append(name)
    seq.append(sequence)
if len(seqid) == 0:
    f=open(Sequence,"r")
    data1 = f.readlines()
    for each in data1:
        seq.append(each.replace('\n',''))
    for i in range (1,len(seq)+1):
        seqid.append("Seq_"+str(i))

seqid_1 = list(map(">{}".format, seqid))
CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
CM.to_csv(f"{wd}/Sequence_1",header=False,index=None,sep="\n")
f.close()

# ==================#################===================== Prediction Module start from here ==============###############==============
if Job == 1:
    
    if Model == 1:
        print('\n======= Thanks for using Prediction module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Hemolytic Activity: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1 , output_len_file = lenchk(dfseq , wd)
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out4', threshold)
        df11 = pd.concat([df_2, df1, mlres], axis=1)
        df11 = round(df11, 3)
        df11.columns = ["SeqID", "Sequence", "ML Score", "Prediction"]
        df11['SeqID'] = df11['SeqID'].str.replace('>', '')
        if dplay == 1:
            df11 = df11.loc[df11['Prediction'] == "Hemolytic"]
            print(df11)
        elif dplay == 2:
            df11 = df11
            print(df11)
        df11.to_csv(f"{wd}/{result_filename}", index=None)
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out4')
        os.remove(f'{wd}/Sequence_1')

        

    elif Model == 2:
        print('\n======= Thanks for using Prediction module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Hemolytic Activity: Processing sequences please wait ...')
        
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out4', threshold)
        df3 = pd.concat([df_2, df1, mlres], axis=1)
        df3.rename(columns={0: 'SeqID'}, inplace=True)
        #### Adding MERCI
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', df_2)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', df_2)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', df_2)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', df_2)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df3, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df12 = pd.read_csv(f'{wd}/final_output')
        df12.loc[df12['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df12.loc[df12['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df12.columns = ['SeqID', 'Sequence', 'ML Score', "MERCI Score", "Hybrid Score", "Prediction"]

        if dplay == 1:
            df12 = df12.loc[df12.Prediction == "Hemolytic"]
            print(df12)
        elif dplay == 2:
            df12 = df12
            print(df12)

        df12 = round(df12, 3)
        df12.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 2
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out4')

    elif Model == 3:
        print('\n======= Thanks for using Prediction module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Hemolytic Activity: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1 = lenchk(dfseq, wd)
        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(dfseq, df_2, f"{wd}/{result_filename}", threshold)
        df13 = pd.read_csv(f"{wd}/{result_filename}")
        df13.rename(columns={"ML Score": "ESM Score"}, inplace=True)
        df13.columns = ['SeqID', 'Sequence', 'ESM Score', "Prediction"]
        df13['SeqID'] = df13['SeqID'].str.replace('>','')

        if dplay == 1:
            df13 = df13.loc[df13.Prediction == "Hemolytic"]
            print(df13)
        elif dplay == 2:
            print(df13)
        df13 = round(df13, 3)
        df13.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 3
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        
        

    elif Model == 4:
        print('\n======= Thanks for using Prediction module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Predicting Hemolytic Activity: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1 = lenchk(dfseq, wd)
        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(dfseq, df_2, f"{wd}/{result_filename}", threshold)
        df3 = pd.read_csv(f"{wd}/{result_filename}")
        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df3, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')
        df14 = pd.read_csv(f'{wd}/final_output')
        df14.loc[df14['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df14.loc[df14['Hybrid Score'] < 0, 'Hybrid Score'] = 0

        # Rename the column and ensure inplace=True works correctly
        df14.rename(columns={"ML Score": "ESM Score"}, inplace=True)
        df14.columns = ['SeqID', 'Sequence', 'ESM Score', "MERCI Score", "Hybrid Score", "Prediction"]
        if dplay == 1:
            df14 = df14.loc[df14.Prediction == "Hemolytic"]
            print(df14)
        elif dplay ==2:
            print(df14)
        df14 = round(df14, 3)
        df14.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 4
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        
#==================#################===================== Protein Scanning Module start from here ==============###############==============
if Job == 2:
    
    if Model == 1:  
        print('\n======= Thanks for using Protein Scan module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('================= Scanning : Processing sequences please wait ...')
        
        df_2,dfseq = readseq(Sequence)
        df_1 = seq_pattern(dfseq,df_2,Win_len)
        dfseq = pd.DataFrame(df_1["Seq"])
        df1, output_len_file = lenchk(dfseq, wd)
        df_1['SeqID'] = df_1['SeqID'].str.replace('>','')
        
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out4', threshold)
        df31 = pd.concat([df_1,  mlres], axis=1)
        df31 = round(df31, 3)
        df31.columns = ["SeqID", "Pattern ID", "Start", "End",  "Sequence", "ML Score", "Prediction"]

        if dplay == 1:
            df31 = df31.loc[df31.Prediction == "Hemolytic"]
            print(df31)
        elif dplay ==2:
            print(df31)
        df31 = round(df31, 3)
        df31.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 1
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out4')
        os.remove(f'{wd}/Sequence_1')

    elif Model == 2 :
        
        print('\n======= Thanks for using Protein Scan module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('================= Scanning : Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df_1 = seq_pattern(dfseq,df_2,Win_len)
        dfseq = pd.DataFrame(df_1["Seq"])
        df1, output_len_file = lenchk(dfseq, wd)
        df_1['SeqID'] = df_1['SeqID'].str.replace('>','')
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out4', threshold)
        df32 = pd.concat([df_1, mlres], axis=1)
        # Rename the column 0 to SeqID

        df32.rename(columns={0: 'SeqID'}, inplace=True)
        ####Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df32, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df32 = pd.read_csv(f'{wd}/final_output')
        df32.loc[df32['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df32.loc[df32['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df32 = pd.concat([df_1, df32.iloc[:, -4:]], axis=1)
        df32.columns = ['SeqID', 'Pattern ID',  'Start',  'End','Sequence',  'ML Score',  'MERCI Score' , 'Hybrid Score','Prediction' ]
      

        if dplay == 1:
            df32 = df32.loc[df32.Prediction == "Hemolytic"]
            print(df32)
        elif dplay == 2:
            df32 = df32
            print(df32)

        df32 = round(df32, 3)
        df32.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 4
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out4')
        os.remove(f'{wd}/out2')

    elif Model == 3 :
        print('\n======= Thanks for using Protein Scan module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('================= Scanning : Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df_1 = seq_pattern(dfseq,df_2,Win_len)
        dfseq = pd.DataFrame(df_1["Seq"])
        df1, output_len_file = lenchk(dfseq, wd)
        
        
        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(dfseq, df_1, f"{wd}/{result_filename}", threshold)
        df33 = pd.read_csv(f"{wd}/{result_filename}")
        # Rename the column and ensure inplace=True works correctly
        df33.rename(columns={"ML Score": "ESM Score"}, inplace=True)
        df33 = pd.concat([df_1, df33['ESM Score'],df33['Prediction']], axis=1)
        df33['SeqID'] = df33['SeqID'].str.replace('>','')
        df33.columns = ['SeqID', 'Pattern ID',  'Start',  'End','Sequence',  'ESM Score', 'Prediction' ]
      
        if dplay == 1:
            df33 = df33.loc[df33.Prediction == "Hemolytic"]
            print(df33)
        elif dplay == 2:
            print(df33)
        df33 = round(df33, 3)
        df33.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 3
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')

    elif Model == 4 :
        print('\n======= Thanks for using Protein Scan module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('================= Scanning : Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df_1 = seq_pattern(dfseq,df_2,Win_len)
        dfseq = pd.DataFrame(df_1["Seq"])
        df1, output_len_file = lenchk(dfseq, wd)
        
        
        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(dfseq, df_1, f"{wd}/{result_filename}", threshold)
        df33 = pd.read_csv(f"{wd}/{result_filename}")

        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df33, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df43 = pd.read_csv(f'{wd}/final_output')
        df43.loc[df43['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df43.loc[df43['Hybrid Score'] < 0, 'Hybrid Score'] = 0

        df43_part = df43.iloc[:, -4:]
        df43 = pd.concat([df_1, df43_part], axis=1)
        df43['SeqID'] = df43['SeqID'].str.replace('>','')
        df43.rename(columns={"ML Score": "ESM Score"}, inplace=True)
        df43.columns = ['SeqID', 'Pattern ID',  'Start',  'End','Sequence',  'ESM Score',  'MERCI Score' , 'Hybrid Score','Prediction' ]
      
        if dplay == 1:
            df43 = df43.loc[df43.Prediction == "Hemolytic"]
            print(df43)
        elif dplay ==2:
            print(df43)
        df43 = round(df43, 3)
        df43.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 4
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        
#======================###############=== Design Model Start from Here =========#########################=================
if Job == 3:
    
    if Model == 1:  
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        mutants = generate_mutants_from_dataframe(df1, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
        result_df['Mutant Sequence'].to_csv(f'{wd}/out_len_mut', index=None, header=None)
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        os.system(f'python3 ./Model/composition_calculate.py {wd}/out_len_mut {wd} {wd}/out3')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out22', threshold)
        mlres_m = ML_run(f'{wd}/out3', f'{wd}/out33', threshold)
        df21 = pd.concat([df_2, result_df['Original Sequence'], mlres, result_df[['Mutant Sequence', 'Position']], mlres_m], axis=1)
        
        # Ensure correct number of column names
        df21.columns = ['SeqID', 'Original Sequence', 'ML Score', 'Prediction', 'Mutant Sequence', 'Position', 'ML Score', 'Prediction']
        df21['SeqID'] = df21['SeqID'].str.replace('>','')
        df21 = round(df21, 3)
        if dplay == 1:
            df21 = df21.loc[df21['Mut_Prediction'] == "Hemolytic"]
            print(df3)
        elif dplay == 2 :
            df21 = df21
            print(df21)
        df21.to_csv(f"{wd}/{result_filename}", index=None)
        #Clean up temporary files used by Model 1
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out_len_mut')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out22')
        os.remove(f'{wd}/out3')
        os.remove(f'{wd}/out33')
        os.remove(f'{wd}/Sequence_1')
        
    elif Model == 2:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        mutants = generate_mutants_from_dataframe(df1, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
        result_df['Mutant Sequence'].to_csv(f'{wd}/out_len_mut', index=None, header=None)
        result_df['Seq'] = df_2
        mutant_fasta = result_df[['Seq','Mutant Sequence']].to_csv(f'{wd}/mutant.fasta', index=None, header=None, sep='\n')
        os.system(f'python3 ./Model/composition_calculate.py {output_len_file} {wd} {wd}/out2')
        os.system(f'python3 ./Model/composition_calculate.py {wd}/out_len_mut {wd} {wd}/out3')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out22', threshold)
        mlres_m = ML_run(f'{wd}/out3', f'{wd}/out33', threshold)

        df22 = pd.concat([df_2, result_df['Original Sequence'], mlres, result_df[['Mutant Sequence', 'Position']], mlres_m], axis=1)
        df22.rename(columns={0: 'SeqID'}, inplace=True)
        df22.rename(columns={"Original Sequence": 'Seq', 'Mutant Sequence': "Seq"}, inplace=True)
        df22['SeqID'] = df22['SeqID'].str.replace('>','')

        df221 = df22.iloc[:,:3]  #####original
        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', df_2)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', df_2)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', df_2)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', df_2)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df221, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df221 = pd.read_csv(f'{wd}/final_output')
        df221.loc[df221['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df221.loc[df221['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df221.rename(columns={'Seq': 'Original Sequence'}, inplace=True)

        df222 = df22.iloc[:, [0, 4, 6]]  ######mutants
        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', df_2)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', df_2)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', df_2)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', df_2)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df222, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df222 = pd.read_csv(f'{wd}/final_output')
        df222.loc[df222['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df222.loc[df222['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df222.rename(columns={'Seq': 'Mutant Sequence'}, inplace=True)
        
        df223 = pd.concat([df221, df222], axis=1)
        df223 = df223.iloc[:, [i for i in range(df223.shape[1]) if i != 6]]
        df223_part1 = df223.iloc[:, :7]
        df223_part2 = df223.iloc[:, 7:]
        df223 = pd.concat([df223_part1, result_df["Position"], df223_part2], axis=1)

        if dplay == 1:
            df223 = df223.loc[df223.Prediction == "Hemolytic"]
            print(df223)
        elif dplay ==2:
            print(df223)
        df223 = round(df223, 3)
        df223.to_csv(f"{wd}/{result_filename}", index=None)

        #Clean up temporary files used by Model 2
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/mutant.fasta')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out_len_mut')
        os.remove(f'{wd}/out22')
        os.remove(f'{wd}/out33')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out3')
        os.remove(f'{wd}/mutant.fasta')

    elif Model == 3:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)

        mutants = generate_mutants_from_dataframe(df1, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'seq', 'Position'])
        out_len_mut = pd.DataFrame(result_df['seq'])

        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        
        run_esm_model(dfseq, df_2, f"{wd}/out_ori", threshold)
        run_esm_model(out_len_mut, df_2 ,f'{wd}/out_m', threshold)
        # Load the CSV files into DataFrames
        out_ori = pd.read_csv(f"{wd}/out_ori")
        out_m = pd.read_csv(f"{wd}/out_m")
        out_m = out_m.drop(columns=['SeqID'])
        # Concatenate the DataFrames
        df23 = pd.concat([out_ori, out_m], axis=1)
        df23 = pd.concat([df_2, result_df['Original Sequence'], out_ori['ML Score'],out_ori['Prediction'] , result_df['seq'], result_df['Position'], out_m['ML Score'],out_m['Prediction'] ], axis=1)
        # Ensure correct number of column names
        df23.columns = ['SeqID', 'Original Sequence', 'ESM Score', 'Prediction', 'Mutant Sequence', 'Position', 'ESM Score', 'Prediction']
        df23['SeqID'] = df23['SeqID'].str.replace('>','')
        df23 = round(df23, 3)
        if dplay == 1:
            df23 = df23.loc[df23['Mut_Prediction'] == "Hemolytic"]
            print(df23)
        elif dplay == 2 :
            df23 = df23
            print(df23)
        df23.to_csv(f"{wd}/{result_filename}", index=None)
        # Clean up temporary files used by Model 3
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out_ori')
        os.remove(f'{wd}/out_m')
        os.remove(f'{wd}/Sequence_1')

    elif Model == 4:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        
        mutants = generate_mutants_from_dataframe(df1, residues, position)
        result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
        out_len_mut = result_df[['Mutant Sequence']].rename(columns={'Mutant Sequence': 'seq'})
        result_df['Seq'] = df_2

        mutant_fasta = result_df[['Seq','Mutant Sequence']].to_csv(f'{wd}/mutant.fasta', index=None, header=None, sep='\n')

        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(dfseq, df_2, f"{wd}/out_ori", threshold)
        run_esm_model(out_len_mut, df_2 ,f'{wd}/out_m', threshold)
        # Load the CSV files into DataFrames
        out_ori = pd.read_csv(f"{wd}/out_ori")
        out_m = pd.read_csv(f"{wd}/out_m")
        out_m = out_m.drop(columns=['SeqID'])
        
        # Concatenate the DataFrames
        df24 = pd.concat([out_ori, out_m], axis=1)
        df241 = df24.iloc[:,:3]  #####original

        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df241, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df241 = pd.read_csv(f'{wd}/final_output')
        df241.loc[df241['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df241.loc[df241['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df241.rename(columns={'Seq': 'Original Sequence'}, inplace=True)

        df242 = df24.iloc[:, [0, 4, 5]] ######mutants
        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df242, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')

        df242 = pd.read_csv(f'{wd}/final_output')
        df242.loc[df242['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df242.loc[df242['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df242.rename(columns={'Seq': 'Mutant Sequence'}, inplace=True)
        df242 = df242.drop(['SeqID'], axis=1)

        df243 = pd.concat([df241, df242], axis=1)

        # Split df243 into two parts: up to the 8th column and from the 9th column onwards
        df243_part1 = df243.iloc[:, :7]
        df243_part2 = df243.iloc[:, 7:]

        # Concatenate the parts with the Position column from result_df
        df243 = pd.concat([df243_part1, result_df["Position"], df243_part2], axis=1)

        # Ensure correct number of column names
        df243.columns = ['SeqID', 'Original Sequence', 'ESM Score', "MERCI Score", "Hybrid Score", 'Prediction', 'Mutant Sequence', 'Position', 'ESM Score', "MERCI Score", "Hybrid Score", 'Prediction']
      
        
        if dplay == 1:
            df243 = df243.loc[df243.Prediction == "Hemolytic"]
            print(df243)
        elif dplay ==2:
            print(df243)
        df243 = round(df243, 3)
        df243.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 4
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/mutant.fasta')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out_m')
        os.remove(f'{wd}/out_ori')
    
#======================###############=== Design all possible Mutants Start from Here =========#########################=================
if Job == 4:
    
    if Model == 1:  
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        muts = all_mutants(dfseq, df_2)
        muts.to_csv(os.path.join(wd, 'muts.csv'), index=None, header=None)
        mut_seq = muts['Seq']
        mut_seq = os.path.join(wd, 'mut_seq')
        muts['Seq'].to_csv(mut_seq, index=None, header=None)
        os.system(f'python3 ./Model/composition_calculate.py {wd}/mut_seq {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out22', threshold)
        df21 = pd.concat([muts, mlres], axis=1)
        df21.columns = ['SeqID', 'Mutant_ID', 'Seq', 'ML Score', 'Prediction']
        df21['SeqID'] = df21['SeqID'].str.replace('>','')
        df21 = round(df21, 3)
        df21.columns = ['SeqID','Mutant_ID','Sequences','ML Score', 'Prediction']
        if dplay == 1:
            df21 = df21.loc[df21['Mut_Prediction'] == "Hemolytic"]
            print(df3)
        elif dplay == 2 :
            df21 = df21
            print(df21)
        df21.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 2
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/mut_seq')
        os.remove(f'{wd}muts.csv')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/out22')
        os.remove(f'{wd}/Sequence_1')
    
    elif Model == 2:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        muts = all_mutants(dfseq, df_2)
        muts.to_csv(os.path.join(wd, 'muts.csv'), index=None, header=None)
        mut_seq = muts['Seq']
        mut_seq = os.path.join(wd, 'mut_seq')
        muts['Seq'].to_csv(mut_seq, index=None, header=None)
        os.system(f'python3 ./Model/composition_calculate.py {wd}/mut_seq {wd} {wd}/out2')
        mlres = ML_run(f'{wd}/out2', f'{wd}/out22', threshold)
        mutant_fasta = muts.copy()

        mutant_fasta['header'] = mutant_fasta['SeqID'].astype(str).str.cat(mutant_fasta['Mutant_ID'].astype(str), sep='_')
        mutant_fasta[['header', 'Seq']].to_csv(f'{wd}/mutant.fasta', index=None, header=None, sep='\n')
        df22 = pd.concat([muts, mlres], axis=1)
        df22.columns = ['SeqID', 'Mutant_ID', 'Seq', 'ML Score', 'Prediction']
        df22['SeqID'] = df22['SeqID'].str.replace('>','')
        df23 = df22.copy()
        df23['SeqID'] = df22['SeqID'] + '_' + df22['Mutant_ID']
        mut_seqid = list(df22['SeqID'] + '_' + df22['Mutant_ID'])
        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', mut_seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', mut_seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', mut_seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', mut_seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df23, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')
        df222 = pd.read_csv(f'{wd}/final_output')
        df222.loc[df222['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df222.loc[df222['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df222.rename(columns={'Seq': 'Mutant Sequence'}, inplace=True)
        
        df223 = pd.concat([df22.iloc[:, :3], df222.iloc[:, -4:]], axis=1)
        df223.columns = ['SeqID','Mutant_ID','Sequences','ML Score','MERCI Score','Hybrid Score', 'Prediction']
        if dplay == 1:
            df223 = df223.loc[df223.Prediction == "Hemolytic"]
            print(df223)
        elif dplay ==2:
            print(df223)
        df223 = round(df223, 3)
        df223.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 2
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/mutant.fasta')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out22')
        os.remove(f'{wd}/out2')
        os.remove(f'{wd}/mut_seq')
        os.remove(f'{wd}/muts.csv')

    elif Model == 3:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        muts = all_mutants(dfseq, df_2)
        muts.to_csv(os.path.join(wd, 'muts.csv'), index=None, header=None)
        mut_seq = pd.DataFrame(muts['Seq'])
        mut_seq.columns = ['seq']
        df_2 = pd.DataFrame(muts.iloc[:, 1])
        df_2.columns = ['SeqID'] 

        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(mut_seq, df_2, f"{wd}/out_ori", threshold)
        out_ori = pd.read_csv(f"{wd}/out_ori")
        df23 = pd.concat([muts, out_ori.iloc[:, -2:]], axis=1)
        df23.columns = ['SeqID','Mutant_ID','Sequences','ESM Score', 'Prediction']
        df23['SeqID'] = df23['SeqID'].str.replace('>','')
        
        df23 = round(df23, 3)
        if dplay == 1:
            df23 = df23.loc[df23['Mut_Prediction'] == "Hemolytic"]
            print(df23)
        elif dplay == 2 :
            df23 = df23
            print(df23)
        df23.to_csv(f"{wd}/{result_filename}", index=None)
        # Clean up temporary files used by Model 3
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/out_ori')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/muts.csv')

    elif Model == 4:
        print('\n======= Thanks for using Design module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
        print('==== Designing Peptides: Processing sequences please wait ...')
        df_2, dfseq = readseq(Sequence)
        df1, output_len_file = lenchk(dfseq, wd)
        
        muts = all_mutants(dfseq, df_2)
        muts.to_csv(os.path.join(wd, 'muts.csv'), index=None, header=None)
        mut_seq = pd.DataFrame(muts['Seq'])
        mut_seq.columns = ['seq']
        df_2 = pd.DataFrame(muts.iloc[:, 1])
        df_2.columns = ['SeqID']
        mutant_fasta = muts.copy()

        mutant_fasta['header'] = mutant_fasta['SeqID'].astype(str).str.cat(mutant_fasta['Mutant_ID'].astype(str), sep='_')
        mutant_fasta[['header', 'Seq']].to_csv(f'{wd}/mutant.fasta', index=None, header=None, sep='\n') 
        # Load the tokenizer and model
        model_save_path = "model/"
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = EsmForSequenceClassification.from_pretrained(model_save_path)
        model.eval()
        run_esm_model(mut_seq, df_2, f"{wd}/out_ori", threshold)
        out_ori = pd.read_csv(f"{wd}/out_ori")
        df23 = pd.concat([muts, out_ori.iloc[:, -2:]], axis=1)
        df24 = df23.copy()
        df24['SeqID'] = df23['SeqID'] + '_' + df23['Mutant_ID']
        mut_seqid = list(df23['SeqID'] + '_' + df23['Mutant_ID'])

        ###Adding merci
        merci = './merci/MERCI_motif_locator.pl'
        motifs_p_1 = './motif/pos_motif_1.txt'
        motifs_n_1 = './motif/neg_motif_1.txt'
        motifs_p_2 = './motif/pos_motif_2.txt'
        motifs_n_2 = './motif/neg_motif_2.txt'
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_1} -o {wd}/merci_p_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_1} -o {wd}/merci_n_1.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_p_2} -c KOOLMAN-ROHM -o {wd}/merci_p_2.txt")
        os.system(f"perl {merci} -p {wd}/mutant.fasta -i {motifs_n_2} -c KOOLMAN-ROHM -o {wd}/merci_n_2.txt")
        MERCI_Processor_p(wd, 'merci_p_1.txt', '/merci_output_p_1.csv', mut_seqid)
        MERCI_Processor_p(wd, 'merci_p_2.txt', '/merci_output_p_2.csv', mut_seqid)
        Merci_after_processing_p(wd, '/merci_output_p_1.csv', '/merci_hybrid_p_1.csv')
        Merci_after_processing_p(wd, '/merci_output_p_2.csv', '/merci_hybrid_p_2.csv')
        MERCI_Processor_n(wd, '/merci_n_1.txt', '/merci_output_n_1.csv', mut_seqid)
        MERCI_Processor_n(wd, '/merci_n_2.txt', '/merci_output_n_2.csv', mut_seqid)
        Merci_after_processing_n(wd, '/merci_output_n_1.csv', '/merci_hybrid_n_1.csv')
        Merci_after_processing_n(wd, '/merci_output_n_2.csv', '/merci_hybrid_n_2.csv')

        hybrid(wd, df24, '/merci_hybrid_p_1.csv', '/merci_hybrid_n_1.csv', '/merci_hybrid_p_2.csv', '/merci_hybrid_n_2.csv', threshold, '/final_output')
        df241 = pd.read_csv(f'{wd}/final_output')
        df241.loc[df241['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df241.loc[df241['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        df243 = pd.concat([muts, df241.iloc[:, -4:]], axis=1)
        df243.columns = ['SeqID','Mutant_ID','Sequences','ESM Score','MERCI Score','Hybrid Score', 'Prediction']
        df243['SeqID'] = df243['SeqID'].str.replace('>','')
        if dplay == 1:
            df243 = df243.loc[df243.Prediction == "Hemolytic"]
            print(df243)
        elif dplay ==2:
            print(df243)
        df243 = round(df243, 3)
        df243.to_csv(f"{wd}/{result_filename}", index=None)

        # Clean up temporary files used by Model 4
        os.remove(f'{wd}/final_output')
        os.remove(f'{wd}/merci_hybrid_p_1.csv')
        os.remove(f'{wd}/merci_hybrid_n_1.csv')
        os.remove(f'{wd}/merci_output_p_1.csv')
        os.remove(f'{wd}/merci_output_n_1.csv')
        os.remove(f'{wd}/merci_hybrid_p_2.csv')
        os.remove(f'{wd}/merci_hybrid_n_2.csv')
        os.remove(f'{wd}/merci_output_p_2.csv')
        os.remove(f'{wd}/merci_output_n_2.csv')
        os.remove(f'{wd}/merci_p_1.txt')
        os.remove(f'{wd}/merci_n_1.txt')
        os.remove(f'{wd}/merci_p_2.txt')
        os.remove(f'{wd}/merci_n_2.txt')
        os.remove(f'{wd}/mutant.fasta')
        os.remove(f'{wd}/Sequence_1')
        os.remove(f'{wd}/out_len')
        os.remove(f'{wd}/muts.csv')
        os.remove(f'{wd}/out_ori')


#==================#################===================== Motif Scanning Module start from here ==============###############==============
if Job == 5:
     
    print('\n======= Thanks for using Motif Scan module of HemoPI2.0. Your results will be stored in file :'f"{wd}/{result_filename}"' =====\n')
    print('================= Scanning : Processing sequences please wait ...')

    df_2, dfseq = readseq(Sequence)
    df1, output_len_file = lenchk(dfseq, wd)
    merci = './merci/MERCI_motif_locator.pl'
    motifs_p = './motif/all_pos.txt'
    motifs_n = './motif/all_neg.txt'

    os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_p} -o {wd}/merci_p.txt")
    os.system(f"perl {merci} -p {wd}/Sequence_1 -i {motifs_n} -o {wd}/merci_n.txt")

    MERCI_Processor_p(wd, 'merci_p.txt', '/merci_output_p.csv', seqid)
    Merci_after_processing_p(wd, '/merci_output_p.csv', '/merci_hybrid_p.csv')
    MERCI_Processor_n(wd, '/merci_n.txt', '/merci_output_n.csv', seqid)
    Merci_after_processing_n(wd, '/merci_output_n.csv', '/merci_hybrid_n.csv')

    # Read each CSV file into a separate DataFrame
    df_p = pd.read_csv(f'{wd}/merci_output_p.csv')
    df_n = pd.read_csv(f'{wd}/merci_output_n.csv')
    df_hp = pd.read_csv(f'{wd}/merci_hybrid_p.csv')
    df_hn = pd.read_csv(f'{wd}/merci_hybrid_n.csv')

    # Merge the DataFrames on 'SeqID'
    df_merged = df_p.merge(df_n, on='SeqID', how='outer')

    df4_selected = df_merged.iloc[:, [0, 1, 3]]

    # Define the function to determine the prediction based on PHits and NHits
    def determine_prediction(row):
        if row['PHits'] == 0 and row['NHits'] == 0:
            return 'Non-Hemolytic'
        elif row['PHits'] > row['NHits']:
            return 'Hemolytic'
        elif row['PHits'] < row['NHits']:
            return 'Non-Hemolytic'
        elif row['PHits'] == row['NHits']:
            return 'Hemolytic'
        else:
            return 'NA'

    # Make an explicit copy of the DataFrame
    df4_selected_copy = df4_selected.copy()

    df4_selected_copy['Prediction'] = df4_selected_copy.apply(determine_prediction, axis=1)

    df4_selected_copy.columns = ["SeqID", 'Positive Hits', 'Negative Hits', "Prediction"]

    if dplay == 1:
        df4_selected_copy = df4_selected_copy.loc[df4_selected_copy.Prediction == "Hemolytic"]
        print(df4_selected_copy)
    elif dplay == 2:
        print(df4_selected_copy)

    df4_selected_copy.to_csv(f"{wd}/{result_filename}", index=None)

    # Clean up temporary files used by Model 2
    os.remove(f'{wd}/merci_hybrid_p.csv')
    os.remove(f'{wd}/merci_hybrid_n.csv')
    os.remove(f'{wd}/merci_output_p.csv')
    os.remove(f'{wd}/merci_output_n.csv')
    os.remove(f'{wd}/merci_p.txt')
    os.remove(f'{wd}/merci_n.txt')
    os.remove(f'{wd}/Sequence_1')
    os.remove(f'{wd}/out_len')

    
print("\n=========Process Completed. Have an awesome day ahead.=============\n")
print('\n======= Thanks for using HemoPI2.0. Your results are stored in file :'f"{wd}{result_filename}"' =====\n\n')
print('Please cite: HemoPI2.0\n')
