import argparse
import warnings
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import re
from time import sleep
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
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
        seqid.append('>'+name)
        seq.append(sequence)
    if len(seqid) == 0:
        f = open(file, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq)+1):
            seqid.append(">Seq_"+str(i))
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
def ML_run(file1, out):
    a = []
    df = pd.read_csv(file1)
    clf = pickle.load(open('./Model/HemoPI2_reg.sav', 'rb'))
    data_test = df
    X_test = data_test
    y_p_score1 = clf.predict(X_test)
    y_p_score1 = np.exp(-y_p_score1)
    y_p_s1 = y_p_score1.tolist()
    a.extend(y_p_s1)
    df_predictions = pd.DataFrame(a)
    df1 = df_predictions.iloc[:, -1].round(3)
    df2 = pd.DataFrame(df1)
    df2.columns = ['HC50']
    dd = pd.concat([df, df2], axis=1)
    dd.to_csv(out, index=None)
    return df2

def emb_process(file):
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


def predict_hemolysis(df):
    df['Prediction'] = df['HC50'].apply(lambda x: 'Hemolytic' if x < 100 else 'Non-Hemolytic')
    return df

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
    xx.columns = ['Seq_ID','Mutant_ID','Seq']
    return xx

print('############################################################################################')
print('# This program HemoPI2.0 is developed for predicting, designing and scanning HC50 of peptides #')
print('# Prediction of Hemolytic activity of peptides, developed by Prof G. P. S. Raghava group. #')
print('# Please cite: HemoPI2.0; available at https://webs.iiitd.edu.in/raghava/hemopi2/  ###########')
print('############################################################################################')

parser = argparse.ArgumentParser(description='Please provide following arguments')
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output", type=str, help="Output: File for saving results by default outfile.csv")
parser.add_argument("-j", "--job", type=int, choices=[1, 2, 3, 4], help="Job Type: 1: Predict, 2: Protein Scanning, 3: Design, 4: Design all possible mutants,by default 1")
parser.add_argument("-p", '--Position', type=int, help='Position of mutation (1-indexed)')
parser.add_argument("-r", '--Residues', type=str, help='Mutated residues (one or two of the 20 essential amino acids in upper case)')
parser.add_argument("-w", "--winleng", type=int, choices=range(8, 21), help="Window Length: 8 to 20 (scan mode only), by default 8")
parser.add_argument("-d", "--display", type=int, choices=[1, 2], default=2, help="Display: 1: Hemolytic, 2: All peptides, by default 2")
parser.add_argument("-wd", "--working", type=str, required=True, help="Working Directory: Location for writing results")

args = parser.parse_args()

Sequence = args.input

if args.output is None:
    result_filename = "outfile.csv"
else:
    result_filename = args.output

if args.job is None:
    Job = int(1)
else:
    Job = int(args.job)

position = args.Position
residues = args.Residues

dplay = args.display

if args.winleng is None:
    Win_len = int(8)
else:
    Win_len = int(args.winleng)

wd = args.working

if not os.path.exists(wd):
    os.makedirs(wd)

if Job == 1:
    df_2, dfseq = readseq(Sequence)
    df1, output_len_file = lenchk(dfseq, wd)
    os.system(f'python3 Model/composition_calculate_hemopi2_2.py {output_len_file} {wd} {wd}/out2')
    mlres = ML_run(f'{wd}/out2', f'{wd}/out4')
    df3 = pd.concat([df_2, df1, mlres], axis=1)
    df3 = predict_hemolysis(df3)
    df3.columns = ['SeqID', 'Sequence', 'HC50(μM)', 'Prediction']
    df3['SeqID'] = df3['SeqID'].str.replace('>', '')
    if dplay == 1:
        df3 = df3.loc[df3['Prediction'] == "Hemolytic"]
        print(df3)
    elif dplay == 2:
        df3 = df3
        print(df3)
    result_filename = os.path.join(wd, 'final_output.csv')
    df3.to_csv(result_filename, index=None)

    #os.remove(output_len_file)
    #os.remove(f'{wd}/out2')
    #os.remove(f'{wd}/out4')

if Job == 2:

    df_2, dfseq = readseq(Sequence)
    df1, output_len_file = lenchk(dfseq, wd)
    df2 = seq_pattern(df1, df_2, Win_len)
    df2.columns = ['SeqID', 'Pattern ID', 'Start', 'End', 'Seq']
    df2.to_csv(f'{wd}/temp_scan.csv', index=None)
    dfseq = pd.DataFrame(df2['Seq'])
    df22 = lenchk(dfseq, wd)
    os.system(f'python3 ./Model/composition_calculate_hemopi2_2.py  {wd}/out_len {wd} {wd}/out2')
    mlres = ML_run(f'{wd}/out2', f'{wd}/out4')
    df3 = pd.concat([df2, mlres], axis=1)
    df3 = predict_hemolysis(df3)
    df3.columns = ['SeqID', 'Pattern ID', 'Start', 'End','Sequence', 'HC50(μM)', 'Prediction']
    df3['SeqID'] = df3['SeqID'].str.replace('>','')

    if dplay == 1:
        df3 = df3.loc[df3['Prediction'] == "Hemolytic"]
        print(df3)
    elif dplay == 2 :
        df3 = df3
        print(df3)

    result_filename = os.path.join(wd, 'final_output.csv')
    df3.to_csv(result_filename, index=None)

    

    #os.remove(output_len_file)
    #os.remove(f'{wd}/out2')
    #os.remove(f'{wd}/out4')
    #os.remove(f'{wd}/temp_scan.csv')

if Job == 3:
    df_2, dfseq = readseq(Sequence)
    df1 = lenchk(dfseq, wd)
    if 'seq' in dfseq.columns:
        dfseq.rename(columns={'seq': 'Seq'}, inplace=True)
    mutants = generate_mutants_from_dataframe(dfseq, residues, position )
    result_df = pd.DataFrame(mutants, columns=['Original Sequence', 'Mutant Sequence', 'Position'])
    out_len_mut_file = os.path.join(wd, 'out_len_mut')
    result_df['Mutant Sequence'].to_csv(out_len_mut_file, index=None, header=None)
    os.system(f'python3 Model/composition_calculate_hemopi2_2.py {wd}/out_len {wd} {wd}/out2')
    os.system(f'python3 Model/composition_calculate_hemopi2_2.py {wd}/out_len_mut {wd} {wd}/out3')
    mlres = ML_run(f'{wd}/out2', f'{wd}/out22')
    mlres_m = ML_run(f'{wd}/out3', f'{wd}/out33')
    mlres = predict_hemolysis(mlres)
    mlres_m = predict_hemolysis(mlres_m)
    df3 = pd.concat([df_2, result_df['Original Sequence'], mlres, result_df[['Mutant Sequence', 'Position']], mlres_m], axis=1)
    df3.columns = [['SeqID', 'Original Sequence', 'Ori_HC50(μM)', 'Prediction','Mutant Sequence', 'Position', 'Mut_HC50(μM)','Prediction']]
    df3['SeqID'] = df3['SeqID'].iloc[:,0].str.replace('>', '')
    if dplay == 1:
        df3 = df3.loc[df3['Prediction'] == "Hemolytic"]
        print(df3)
    elif dplay == 2 :
        df3 = df3
        print(df3)

    result_filename = os.path.join(wd, 'final_output.csv')
    df3.to_csv(result_filename, index=None)
    
    # Clean up temporary files
    #os.remove(os.path.join(wd, 'out_len'))
    #os.remove(os.path.join(wd, 'out_len_mut'))
    #os.remove(os.path.join(wd, 'out2'))
    #os.remove(os.path.join(wd, 'out22'))
    #os.remove(os.path.join(wd, 'out3'))
    #os.remove(os.path.join(wd, 'out33'))


if Job ==  4:
    df_2,dfseq = readseq(Sequence)
    df1 = lenchk(dfseq, wd)
    muts = all_mutants(dfseq,df_2)
    muts = all_mutants(dfseq, df_2)
    muts.to_csv(os.path.join(wd, 'muts.csv'), index=None, header=None)
    mut_seq = muts['Seq']
    mut_seq = os.path.join(wd, 'mut_seq')
    muts['Seq'].to_csv(mut_seq, index=None, header=None)
    os.system(f'python3 Model/composition_calculate_hemopi2_2.py {wd}/mut_seq {wd} {wd}/out2')
    mlres = ML_run(f'{wd}/out2', f'{wd}/out22')
    mlres = predict_hemolysis(mlres)
    df3 = pd.concat([muts, mlres], axis=1)
    df3.columns = [['SeqID', 'Mutant_ID', 'Sequence', 'HC50(μM)','Prediction']]
    df3['SeqID'] = df3['SeqID'].iloc[:,0].str.replace('>', '')

    if dplay == 1:
        df3 = df3.loc[df3['Prediction'] == "Hemolytic"]
        print(df3)
    elif dplay == 2 :
        df3 = df3
        print(df3)
    # Save concatenated data to CSV file
    result_filename = os.path.join(wd, 'final_output.csv')
    df3.to_csv(result_filename, index=None)


print("\n=========Process Completed. Have an awesome day ahead.=============\n")
print(f"\n======= Thanks for using HemoPI2.0. Your results are stored in file: {os.path.join(wd, result_filename)} =====\n\n")
print('Please cite: HemoPI2.0\n')