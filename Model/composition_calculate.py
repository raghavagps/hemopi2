from __future__ import print_function
import getopt
import sys
import os
import numpy as np
import pandas as pd
import math
import itertools
from itertools import repeat
import argparse
import csv
from collections import Counter
import re
import glob
import time
from time import sleep
from tqdm import tqdm
from argparse import RawTextHelpFormatter
import uuid
import warnings


# Standard amino acid codes
std = "ACDEFGHIKLMNPQRSTVWY"

############################Molecular_weight###################################

# Molecular weight table for amino acids
molecular_weight_table = {
    'A': 0.089, 'R': 0.174, 'N': 0.132, 'D': 0.133, 'C': 0.121,
    'E': 0.147, 'Q': 0.146, 'G': 0.075, 'H': 0.155, 'I': 0.131,
    'L': 0.131, 'K': 0.146, 'M': 0.150, 'F': 0.165, 'P': 0.115,
    'S': 0.105, 'T': 0.119, 'W': 0.204, 'Y': 0.181, 'V': 0.117
}

# Molecular weight of water (H2O)
water_weight = 0.018

def calculate_molecular_weight(protein_string):
    weight = sum(molecular_weight_table.get(aa, 0) for aa in protein_string)
    total_weight = weight - (water_weight * (len(protein_string) - 1))
    return total_weight

def molecular_weight_comp(file, out):
    filename, _ = os.path.splitext(file)
    f = open(out, 'w')
    sys.stdout = f
    df = pd.read_csv(file, header=None)
    zz = df.iloc[:, 0]
    print("Molecular Weight (kDa),")
    for j in zz:
        weight = calculate_molecular_weight(j)
        print("%.3f" % weight)
    f.truncate()

############################length###################################
def calculate_length(protein_string):
    return len(protein_string)


def length_comp(file, out):
    filename, _ = os.path.splitext(file)
    f = open(out, 'w')
    sys.stdout = f
    df = pd.read_csv(file, header=None)
    zz = df.iloc[:, 0]
    print("length,")
    for j in zz:
        length = calculate_length(j)
        print(length)
    f.truncate()



def aac_comp(file, out):
    filename, _ = os.path.splitext(file)
    f = open(out, 'w')
    sys.stdout = f
    df = pd.read_csv(file, header=None)
    zz = df.iloc[:, 0]
    print("AAC_A,AAC_C,AAC_D,AAC_E,AAC_F,AAC_G,AAC_H,AAC_I,AAC_K,AAC_L,AAC_M,AAC_N,AAC_P,AAC_Q,AAC_R,AAC_S,AAC_T,AAC_V,AAC_W,AAC_Y,")
    for j in zz:
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
            composition = (count / len(j)) * 100
            print("%.2f" % composition, end=",")
        print("")
    f.truncate()


def dpc_comp(file, q, out):
    filename, _ = os.path.splitext(file)
    f = open(out, 'w')
    sys.stdout = f
    df = pd.read_csv(file, header=None)
    df1 = pd.DataFrame(df[0].str.upper())
    zz = df1.iloc[:, 0]
    for s in std:
        for u in std:
            print("DPC" + str(q) + "_" + s + u, end=',')
    print("")
    for i in range(0, len(zz)):
        for j in std:
            for k in std:
                count = 0
                temp = j + k
                for m3 in range(0, len(zz[i]) - q):
                    b = zz[i][m3:m3 + q + 1:q]
                    if b == temp:
                        count += 1
                composition = (count / (len(zz[i]) - (q))) * 100
                print("%.2f" % composition, end=',')
        print("")
    f.truncate()


def tpc_comp(file, out):
    filename, _ = os.path.splitext(file)
    f = open(out, 'w')
    sys.stdout = f
    df = pd.read_csv(file, header=None)
    zz = df.iloc[:, 0]
    for s in std:
        for u in std:
            for m in std:
                print('TPC_' + s + u + m, end=",")
    print("")
    for i in range(0, len(zz)):
        for j in std:
            for k in std:
                for m1 in std:
                    count = 0
                    temp = j + k + m1
                    for m3 in range(0, len(zz[i])):
                        b = zz[i][m3:m3 + 3]
                        if b == temp:
                            count += 1
                    composition = (count / (len(zz[i]) - 2)) * 100
                    print("%.2f" % composition, end=',')
        print("")
    f.truncate()
def atc(file,out):
    filename,file_ext = os.path.splitext(file)
    atom=pd.read_csv("Model/Data/atom.csv",header=None)
    at=pd.DataFrame()
    i = 0
    C_atom = []
    H_atom = []
    N_atom = []
    O_atom = []
    S_atom = []

    while i < len(atom):
        C_atom.append(atom.iloc[i,1].count("C"))
        H_atom.append(atom.iloc[i,1].count("H"))
        N_atom.append(atom.iloc[i,1].count("N"))
        O_atom.append(atom.iloc[i,1].count("O"))
        S_atom.append(atom.iloc[i,1].count("S"))
        i += 1
    atom["C_atom"]=C_atom
    atom["O_atom"]=O_atom
    atom["H_atom"]=H_atom
    atom["N_atom"]=N_atom
    atom["S_atom"]=S_atom
##############read file ##########
    test1 = pd.read_csv(file,header=None)
    dd = []
    for i in range(0, len(test1)):
        dd.append(test1[0][i].upper())
    test = pd.DataFrame(dd)
    count_C = 0
    count_H = 0
    count_N = 0
    count_O = 0
    count_S = 0
    count = 0
    i1 = 0
    j = 0
    k = 0
    C_ct = []
    H_ct = []
    N_ct = []
    O_ct = []
    S_ct = []
    while i1 < len(test) :
        while j < len(test[0][i1]) :
            while k < len(atom) :
                if test.iloc[i1,0][j]==atom.iloc[k,0].replace(" ","") :
                    count_C = count_C + atom.iloc[k,2]
                    count_H = count_H + atom.iloc[k,3]
                    count_N = count_N + atom.iloc[k,4]
                    count_O = count_O + atom.iloc[k,5]
                    count_S = count_S + atom.iloc[k,6]
                #count = count_C + count_H + count_S + count_N + count_O
                k += 1
            k = 0
            j += 1
        C_ct.append(count_C)
        H_ct.append(count_H)
        N_ct.append(count_N)
        O_ct.append(count_O)
        S_ct.append(count_S)
        count_C = 0
        count_H = 0
        count_N = 0
        count_O = 0
        count_S = 0
        j = 0
        i1 += 1
    test["C_count"]=C_ct
    test["H_count"]=H_ct
    test["N_count"]=N_ct
    test["O_count"]=O_ct
    test["S_count"]=S_ct

    ct_total = []
    m = 0
    while m < len(test) :
        ct_total.append(test.iloc[m,1] + test.iloc[m,2] + test.iloc[m,3] + test.iloc[m,4] + test.iloc[m,5])
        m += 1
    test["count"]=ct_total
##########final output#####
    final = pd.DataFrame()
    n = 0
    p = 0
    C_p = []
    H_p = []
    N_p = []
    O_p = []
    S_p = []
    while n < len(test):
        C_p.append((test.iloc[n,1]/test.iloc[n,6])*100)
        H_p.append((test.iloc[n,2]/test.iloc[n,6])*100)
        N_p.append((test.iloc[n,3]/test.iloc[n,6])*100)
        O_p.append((test.iloc[n,4]/test.iloc[n,6])*100)
        S_p.append((test.iloc[n,5]/test.iloc[n,6])*100)
        n += 1
    final["ATC_C"] = C_p
    final["ATC_H"] = H_p
    final["ATC_N"] = N_p
    final["ATC_O"] = O_p
    final["ATC_S"] = S_p

    (final.round(2)).to_csv(out, index = None, encoding = 'utf-8')

def bond(file,out) :
    tota = []
    hy = []
    Si = []
    Du = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    bb = pd.DataFrame()
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    bonds=pd.read_csv("Model/Data/bonds.csv", sep = ",")
    for i in range(0,len(df)) :
        tot = 0
        h = 0
        S = 0
        D = 0
        tota.append([i])
        hy.append([i])
        Si.append([i])
        Du.append([i])
        for j in range(0,len(df[0][i])) :
            temp = df[0][i][j]
            for k in range(0,len(bonds)) :
                if bonds.iloc[:,0][k] == temp :
                    tot = tot + bonds.iloc[:,1][k]
                    h = h + bonds.iloc[:,2][k]
                    S = S + bonds.iloc[:,3][k]
                    D = D + bonds.iloc[:,4][k]
        tota[i].append(tot)
        hy[i].append(h)
        Si[i].append(S)
        Du[i].append(D)
    for m in range(0,len(df)) :
        b1.append(tota[m][1])
        b2.append(hy[m][1])
        b3.append(Si[m][1])
        b4.append(Du[m][1])

    bb["BTC_T"] = b1
    bb["BTC_H"] = b2
    bb["BTC_S"] = b3
    bb["BTC_D"] = b4

    bb.to_csv(out, index=None, encoding="utf-8")
############################PhysicoChemical Properties###################################

PCP= pd.read_csv('Model/Data/PhysicoChemical.csv', header=None)

headers = ['PCP_PC','PCP_NC','PCP_NE','PCP_PO','PCP_NP','PCP_AL','PCP_CY','PCP_AR','PCP_AC','PCP_BS','PCP_NE_pH','PCP_HB','PCP_HL','PCP_NT','PCP_HX','PCP_SC','PCP_SS_HE','PCP_SS_ST','PCP_SS_CO','PCP_SA_BU','PCP_SA_EX','PCP_SA_IN','PCP_TN','PCP_SM','PCP_LR','PCP_Z1','PCP_Z2','PCP_Z3','PCP_Z4','PCP_Z5'];

def encode(peptide):
    l=len(peptide);
    encoded=np.zeros(l);
    for i in range(l):
        if(peptide[i]=='A'):
            encoded[i] = 0;
        elif(peptide[i]=='C'):
            encoded[i] = 1;
        elif(peptide[i]=='D'):
            encoded[i] = 2;
        elif(peptide[i]=='E'):
            encoded[i] = 3;
        elif(peptide[i]=='F'):
            encoded[i] = 4;
        elif(peptide[i]=='G'):
            encoded[i] = 5;
        elif(peptide[i]=='H'):
            encoded[i] = 6;
        elif(peptide[i]=='I'):
            encoded[i] = 7;
        elif(peptide[i]=='K'):
            encoded[i] = 8;
        elif(peptide[i]=='L'):
            encoded[i] = 9;
        elif(peptide[i]=='M'):
            encoded[i] = 10;
        elif(peptide[i]=='N'):
            encoded[i] = 11;
        elif(peptide[i]=='P'):
            encoded[i] = 12;
        elif(peptide[i]=='Q'):
            encoded[i] = 13;
        elif(peptide[i]=='R'):
            encoded[i] = 14;
        elif(peptide[i]=='S'):
            encoded[i] = 15;
        elif(peptide[i]=='T'):
            encoded[i] = 16;
        elif(peptide[i]=='V'):
            encoded[i] = 17;
        elif(peptide[i]=='W'):
            encoded[i] = 18;
        elif(peptide[i]=='Y'):
            encoded[i] = 19;
        else:
            print('Wrong residue!');
    return encoded;
def lookup(peptide,featureNum):
    l=len(peptide);
    peptide = list(peptide);
    out=np.zeros(l);
    peptide_num = encode(peptide);

    for i in range(l):
        out[i] = PCP[peptide_num[i]][featureNum];
    return sum(out);
def pcp_1(file,out123):

    if(type(file) == str):
        seq = pd.read_csv(file,header=None);
        #seq=seq.T
        seq[0].values.tolist()
        seq=seq[0];
    else:
        seq  = file;

    l = len(seq);

    rows = PCP.shape[0]; # Number of features in our reference table
    col = 20 ; # Denotes the 20 amino acids

    seq=[seq[i].upper() for i in range(l)]
    sequenceFeature = [];
    sequenceFeature.append(headers); #To put property name in output csv

    for i in range(l): # Loop to iterate over each sequence
        nfeatures = rows;
        sequenceFeatureTemp = [];
        for j in range(nfeatures): #Loop to iterate over each feature
            featureVal = lookup(seq[i],j)
            if(len(seq[i])!=0):
                sequenceFeatureTemp.append(round(featureVal/len(seq[i]),3));
            else:
                sequenceFeatureTemp.append('NaN')

        sequenceFeature.append(sequenceFeatureTemp);

    out = pd.DataFrame(sequenceFeature);
    file = open(out123,'w')
    with file:
        writer = csv.writer(file);
        writer.writerows(sequenceFeature);
    return sequenceFeature;

def DDOR(file,out) :
    df = pd.read_csv(file, header = None)
    df1 = pd.DataFrame(df[0].str.upper())
    f = open(out,'w')
    sys.stdout = f
    for i in std:
        print('DDR_'+i, end=",")
    print("")
    for i in range(0,len(df1)):
        s = df1[0][i]
        p = s[::-1]
        for j in std:
            zz = ([pos for pos, char in enumerate(s) if char == j])
            pp = ([pos for pos, char in enumerate(p) if char == j])
            ss = []
            for i in range(0,(len(zz)-1)):
                ss.append(zz[i+1] - zz[i]-1)
            if zz == []:
                ss = []
            else:
                ss.insert(0,zz[0])
                ss.insert(len(ss),pp[0])
            cc1=  (sum([e for e in ss])+1)
            cc = sum([e*e for e in ss])
            zz2 = cc/cc1
            print("%.2f"%zz2,end=",")
        print("")
    f.truncate()
    
##################################

def entropy_single(seq):
    seq=seq.upper()
    num, length = Counter(seq), len(seq)
    return -sum( freq/length * math.log(freq/length, 2) for freq in num.values())

def SE(filename,out):
    data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
#     print(data)
    Val=[]
    header=["SEP"]
    for i in range(len(data)):
        data1=''
        data1=str(data[i])
        data1=data1.upper()
        allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
        is_data_invalid = set(data1).issubset(allowed)
        if is_data_invalid==False:
            print("Error: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
            return
        Val.append(round((entropy_single(str(data[i]))),3))
        #print(Val[i])
        file= open(out,'w', newline='\n')#output file
        with file:
            writer=csv.writer(file,delimiter='\n');
            writer.writerow(header)
            writer.writerow(Val);
    return Val


def SE_residue_level(filename,out):
    data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    data2=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    Val=np.zeros(len(data))
    GH=[]
    for i in range(len(data)):
        my_list={'A':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
        data1=''
        data1=str(data[i])
        data1=data1.upper()
        allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
        is_data_invalid = set(data1).issubset(allowed)
        if is_data_invalid==False:
            print("Error: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
            return
        seq=data[i]
        seq=seq.upper()
        num, length = Counter(seq), len(seq)
        num=dict(sorted(num.items()))
        C=list(num.keys())
        F=list(num.values())
        for key, value in my_list.items():
             for j in range(len(C)):
                if key == C[j]:
                    my_list[key] = round(((F[j]/length)* math.log(F[j]/length, 2)),3)
        GH.append(list(my_list.values()))
    file= open(out,'w', newline='')#output file
    with file:
        writer=csv.writer(file);
        writer.writerow(('SER_A','SER_C','SER_D','SER_E','SER_F','SER_G','SER_H','SER_I','SER_K','SER_L','SER_M','SER_N','SER_P','SER_Q','SER_R','SER_S','SER_T','SER_V','SER_W','SER_Y'));
        writer.writerows(GH);
    return(GH)

def RAAC(file,out):
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df1 = pd.DataFrame(df[0].str.upper())
    count = 0
    cc = []
    i = 0
    x = 0
    temp = pd.DataFrame()
    f = open(out,'w')
    sys.stdout = f
    print("RRI_A,RRI_C,RRI_D,RRI_E,RRI_F,RRI_G,RRI_H,RRI_I,RRI_K,RRI_L,RRI_M,RRI_N,RRI_P,RRI_Q,RRI_R,RRI_S,RRI_T,RRI_V,RRI_W,RRI_Y,")
    for q in range(0,len(df1)):
        while i < len(std):
            cc = []
            for j in df1[0][q]:
                if j == std[i]:
                    count += 1
                    cc.append(count)
                else:
                    count = 0
            while x < len(cc) :
                if x+1 < len(cc) :
                    if cc[x]!=cc[x+1] :
                        if cc[x] < cc[x+1] :
                            cc[x]=0
                x += 1
            cc1 = [e for e in cc if e!= 0]
            cc = [e*e for e in cc if e != 0]
            zz= sum(cc)
            zz1 = sum(cc1)
            if zz1 != 0:
                zz2 = zz/zz1
            else:
                zz2 = 0
            print("%.2f"%zz2,end=',')
            i += 1
        i = 0
        print(" ")
    f.truncate()

##########################################PRI###########################
PCP= pd.read_csv('Model/Data/PhysicoChemical.csv', header=None) #Our reference table for properties
headers_1 = ['PRI_PC','PRI_NC','PRI_NE','PRI_PO','PRI_NP','PRI_AL','PRI_CY','PRI_AR','PRI_AC','PRI_BS','PRI_NE_pH','PRI_HB','PRI_HL','PRI_NT','PRI_HX','PRI_SC','PRI_SS_HE','PRI_SS_ST','PRI_SS_CO','PRI_SA_BU','PRI_SA_EX','PRI_SA_IN','PRI_TN','PRI_SM','PRI_LR']

# Amino acid encoding dictionary
amino_acid_dict = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
def encode(peptide):
    # Use a list comprehension for fast lookup
    encoded = np.array([amino_acid_dict.get(aa, -1) for aa in peptide])
    
    # Check for invalid residues
    if -1 in encoded:
        invalid_residues = [peptide[i] for i in range(len(peptide)) if encoded[i] == -1]
        for residue in invalid_residues:
            print(residue, 'is a wrong residue!')
    
    return encoded

def lookup_1(peptide,featureNum):
    l=len(peptide);
    peptide = list(peptide);
    out=[];
    peptide_num = encode(peptide);

    for i in range(l):
        out.append(PCP[peptide_num[i]][featureNum]);
    return out;

def binary_profile_1(file, featureNumb):
    if isinstance(file, str):
        seq = pd.read_csv(file, header=None, sep=',')[0].str.upper().tolist()
    else:
        seq = file    
    bin_prof = [lookup_1(peptide, featureNumb) for peptide in seq]
    
    return bin_prof

def repeats(file, out123):
    seq1 = pd.read_csv(file, header=None)
    seq = seq1[0].tolist()
    dist = [headers_1]
    
    for peptide in seq:
        temp = []
        for j in range(25):
            bin_prof = binary_profile_1([peptide], j)[0]  # Get the profile for the current feature number
            k, num, denom, ones, zeros = 0, 0, 0, 0, 0
            
            # Calculate the required values
            for idx, val in enumerate(bin_prof):
                if val == 0:
                    num += k * k
                    denom += k
                    k = 0
                    zeros += 1
                else:
                    k += 1
                    ones += 1
                
                if idx == len(bin_prof) - 1 and val != 0:
                    num += k * k
                    denom += k
            if ones != 0:
                temp.append(round(num / (ones * ones), 2))
            else:
                temp.append(0)
        dist.append(temp)
    out = pd.DataFrame(dist)
    file1 = open(out123,'w')
    with file1:
        writer = csv.writer(file1);
        writer.writerows(dist);
    return out


def pcp(file):
    SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
    if(type(file) == str):
        seq = pd.read_csv(file,header=None, sep=',');
        seq=seq.T
        seq[0].values.tolist()
        seq=seq[0];
    else:
        seq  = file;
    l = len(seq);
    rows = PCP.shape[0]; # Number of features in our reference table
    col = 20 ; # Denotes the 20 amino acids
    seq=[seq[i].upper() for i in range(l)]
    sequenceFeature = [];
    sequenceFeature.append(SEP_headers); #To put property name in output csv
    
    for i in range(l): # Loop to iterate over each sequence
        nfeatures = rows;
        sequenceFeatureTemp = [];
        for j in range(nfeatures): #Loop to iterate over each feature
            featureVal = lookup(seq[i],j)   
            if(len(seq[i])!=0):
                sequenceFeatureTemp.append(featureVal/len(seq[i]))
            else:
                sequenceFeatureTemp.append('NaN')
        sequenceFeature.append(sequenceFeatureTemp);
    out = pd.DataFrame(sequenceFeature);
    return sequenceFeature;
def phyChem(file,mode='all',m=0,n=0):
    if(type(file) == str):
        seq1 = pd.read_csv(file,header=None, sep=',');
        seq1 = pd.DataFrame(seq1[0].str.upper())
        seq=[]
        [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
    else:
        seq  = file;
    l = len(seq);
    newseq = [""]*l; # To store the n-terminal sequence
    for i in range(0,l):
        l = len(seq[i]);
        if(mode=='NT'):
            n=m;
            if(n!=0):
                newseq[i] = seq[i][0:n];
            elif(n>l):
                print('Warning! Sequence',i,"'s size is less than n. The output table would have NaN for this sequence");
            else:
                print('Value of n is mandatory, it cannot be 0')
                break;
        elif(mode=='CT'):
            n=m;
            if(n!=0):
                newseq[i] = seq[i][(len(seq[i])-n):]
            elif(n>l):
                print('WARNING: Sequence',i+1,"'s size is less than the value of n given. The output table would have NaN for this sequence");
            else:
                print('Value of n is mandatory, it cannot be 0')
                break;
        elif(mode=='all'):
            newseq = seq;
        elif(mode=='rest'):
            if(m==0):
                print('Kindly provide start index for rest, it cannot be 0');
                break;
            else:
                if(n<=len(seq[i])):
                    newseq[i] = seq[i][m-1:n+1]
                elif(n>len(seq[i])):
                    newseq[i] = seq[i][m-1:len(seq[i])]
                    print('WARNING: Since input value of n for sequence',i+1,'is greater than length of the protein, entire sequence starting from m has been considered')
        else:
            print("Wrong Mode. Enter 'NT', 'CT','all' or 'rest'");        
    output = pcp(newseq);
    return output
def shannons(filename,out123):
    SEP_headers = ['SEP_PC','SEP_NC','SEP_NE','SEP_PO','SEP_NP','SEP_AL','SEP_CY','SEP_AR','SEP_AC','SEP_BS','SEP_NE_pH','SEP_HB','SEP_HL','SEP_NT','SEP_HX','SEP_SC','SEP_SS_HE','SEP_SS_ST','SEP_SS_CO','SEP_SA_BU','SEP_SA_EX','SEP_SA_IN','SEP_TN','SEP_SM','SEP_LR']
    if(type(filename) == str):
        seq1 = pd.read_csv(filename,header=None, sep=',');
        seq1 = pd.DataFrame(seq1[0].str.upper())
    else:
        seq1  = filename;
    seq=[]
    [seq.append(seq1.iloc[i][0]) for i in range(len(seq1))]
    comp = phyChem(seq);
    new = [comp[i][0:25] for i in range(len(comp))]
    entropy  = [];
    entropy.append(SEP_headers[0:25])
    for i in range(1,len(new)):
        seqEntropy = [];
        for j in range(len(new[i])):
            p = new[i][j]; 
            if((1-p) == 0. or p==0.):
                temp = 0;#to store entropy of each sequence
            else:
                temp = -(p*math.log2(p)+(1-p)*math.log2(1-p));
            seqEntropy.append(round(temp,3));
        entropy.append(seqEntropy);
    out = pd.DataFrame(entropy);
    file = open(out123,'w')
    with file:
        writer = csv.writer(file);
        writer.writerows(entropy);
    return entropy;

##################paac####################
def val(AA_1, AA_2, aa, mat):
    return sum([(mat[i][aa[AA_1]] - mat[i][aa[AA_2]]) ** 2 for i in range(len(mat))]) / len(mat)
def paac_1(file,lambdaval,w=0.05):
    data1 = pd.read_csv("Model/Data/data", sep = "\t")
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df1 = pd.DataFrame(df[0].str.upper())
    dd = []
    cc = []
    pseudo = []
    aa = {}
    for i in range(len(std)):
        aa[std[i]] = i
    for i in range(0,3):
        mean = sum(data1.iloc[i][1:])/20
        rr = math.sqrt(sum([(p-mean)**2 for p in data1.iloc[i][1:]])/20)
        dd.append([(p-mean)/rr for p in data1.iloc[i][1:]])
        zz = pd.DataFrame(dd)
    head = []
    for n in range(1, lambdaval + 1):
        head.append('_lam' + str(n))
    head = ['PAAC'+str(lambdaval)+sam for sam in head]
    pp = pd.DataFrame()
    ee = []
    for k in range(0,len(df1)):
        cc = []
        pseudo1 = [] 
        for n in range(1,lambdaval+1):
            cc.append(sum([val(df1[0][k][p], df1[0][k][p + n], aa, dd) for p in range(len(df1[0][k]) - n)]) / (len(df1[0][k]) - n))
            qq = pd.DataFrame(cc)
        pseudo = pseudo1 + [(w * p) / (1 + w * sum(cc)) for p in cc]
        ee.append(pseudo)
        ii = round(pd.DataFrame(ee, columns = head),4)
    ii.to_csv(filename+".lam",index = None)
		
def paac(file,lambdaval,out,w=0.05):
    filename, file_extension = os.path.splitext(file)
    paac_1(file,lambdaval,w=0.05)
    aac_comp(file,filename+".aac")
    data1 = pd.read_csv(filename+".aac")
    header = ['PAAC'+str(lambdaval)+'_A','PAAC'+str(lambdaval)+'_C','PAAC'+str(lambdaval)+'_D','PAAC'+str(lambdaval)+'_E','PAAC'+str(lambdaval)+'_F','PAAC'+str(lambdaval)+'_G','PAAC'+str(lambdaval)+'_H','PAAC'+str(lambdaval)+'_I','PAAC'+str(lambdaval)+'_K','PAAC'+str(lambdaval)+'_L','PAAC'+str(lambdaval)+'_M','PAAC'+str(lambdaval)+'_N','PAAC'+str(lambdaval)+'_P','PAAC'+str(lambdaval)+'_Q','PAAC'+str(lambdaval)+'_R','PAAC'+str(lambdaval)+'_S','PAAC'+str(lambdaval)+'_T','PAAC'+str(lambdaval)+'_V','PAAC'+str(lambdaval)+'_W','PAAC'+str(lambdaval)+'_Y','Un']	
    data1.columns = header    
    data2 = pd.read_csv(filename+".lam")
    data3 = pd.concat([data1.iloc[:,:-1],data2], axis = 1).reset_index(drop=True)
    data3.to_csv(out,index=None)
    os.remove(filename+".lam")
    os.remove(filename+".aac")
######################apaac############################	
def apaac_1(file,lambdaval,w=0.05):
    data1 = pd.read_csv("Model/Data/data", sep = "\t")
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df1 = pd.DataFrame(df[0].str.upper())
    dd = []
    cc = []
    pseudo = []
    aa = {}
    for i in range(len(std)):
        aa[std[i]] = i
    for i in range(0,3):
        mean = sum(data1.iloc[i][1:])/20
        rr = math.sqrt(sum([(p-mean)**2 for p in data1.iloc[i][1:]])/20)
        dd.append([(p-mean)/rr for p in data1.iloc[i][1:]])
        zz = pd.DataFrame(dd)
    head = []
    for n in range(1, lambdaval + 1):
        for e in ('HB','HL','SC'):
            head.append(e+'_lam' + str(n))
    head = ['APAAC'+str(lambdaval)+'_'+sam for sam in head]
    pp = pd.DataFrame()
    ee = []
    for k in range(0,len(df1)):
        cc = [] 
        for n in range(1,lambdaval+1):
            for b in range(0,len(zz)):
                cc.append(sum([zz.loc[b][aa[df1[0][k][p]]] * zz.loc[b][aa[df1[0][k][p + n]]] for p in range(len(df1[0][k]) - n)]) / (len(df1[0][k]) - n))
                qq = pd.DataFrame(cc)
        pseudo = [(w * p) / (1 + w * sum(cc)) for p in cc]
        ee.append(pseudo)
        ii = round(pd.DataFrame(ee, columns = head),4)
    ii.to_csv(filename+".plam",index = None)
		
def apaac(file,lambdaval,out,w=0.05):
    filename, file_extension = os.path.splitext(file)
    apaac_1(file,lambdaval,w=0.05)
    aac_comp(file,filename+".aac")
    data1 = pd.read_csv(filename+".aac")
    headaac = []
    for i in std:
        headaac.append('APAAC'+str(lambdaval)+'_'+i)
    headaac.insert(len(headaac),0)
    data1.columns = headaac
    data2 = pd.read_csv(filename+".plam")
    data3 = pd.concat([data1.iloc[:,:-1],data2], axis = 1).reset_index(drop=True)
    data3.to_csv(out, index = None)
    os.remove(filename+".plam")
    os.remove(filename+".aac")
###################################qos#######################################
def qos(file,gap,out,w=0.1):
    ff = []
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df2 = pd.DataFrame(df[0].str.upper())
    for i in range(0,len(df2)):
        ff.append(len(df2[0][i]))
    if min(ff) < gap:
        print("Error: All sequences' length should be higher than :", gap)
    else:
        mat1 = pd.read_csv("Model/Data/Schneider-Wrede.csv", index_col = 'Name')
        mat2 = pd.read_csv("Model/Data/Grantham.csv", index_col = 'Name')
        s1 = []
        s2 = []
        for i in range(0,len(df2)):
            for n in range(1, gap+1):
                sum1 = 0
                sum2 = 0
                for j in range(0,(len(df2[0][i])-n)):
                    sum1 = sum1 + (mat1[df2[0][i][j]][df2[0][i][j+n]])**2
                    sum2 = sum2 + (mat2[df2[0][i][j]][df2[0][i][j+n]])**2
                s1.append(sum1)
                s2.append(sum2)
        zz = pd.DataFrame(np.array(s1).reshape(len(df2),gap))
        zz["sum"] = zz.sum(axis=1)
        zz2 = pd.DataFrame(np.array(s2).reshape(len(df2),gap))
        zz2["sum"] = zz2.sum(axis=1)
        c1 = []
        c2 = []
        c3 = []
        c4 = []
        h1 = []
        h2 = []
        h3 = []
        h4 = []
        for aa in std:
            h1.append('QSO'+str(gap)+'_SC_' + aa)
        for aa in std:
            h2.append('QSO'+str(gap)+'_G_' + aa)
        for n in range(1, gap+1):
            h3.append('SC' + str(n))
        h3 = ['QSO'+str(gap)+'_'+sam for sam in h3]
        for n in range(1, gap+1):
            h4.append('G' + str(n))
        h4 = ['QSO'+str(gap)+'_'+sam for sam in h4]
        for i in range(0,len(df2)):
            AA = {}
            for j in std:
                AA[j] = df2[0][i].count(j)
                c1.append(AA[j] / (1 + w * zz['sum'][i]))
                c2.append(AA[j] / (1 + w * zz2['sum'][i]))
            for k in range(0,gap):
                c3.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
                c4.append((w * zz[k][i]) / (1 + w * zz['sum'][i]))
        pp1 = np.array(c1).reshape(len(df2),len(std))
        pp2 = np.array(c2).reshape(len(df2),len(std))
        pp3 = np.array(c3).reshape(len(df2),gap)
        pp4 = np.array(c4).reshape(len(df2),gap)
        zz5 = round(pd.concat([pd.DataFrame(pp1, columns = h1),pd.DataFrame(pp2,columns = h2),pd.DataFrame(pp3, columns = h3),pd.DataFrame(pp4, columns = h4)], axis = 1),4)
        zz5.to_csv(out, index = None, encoding = 'utf-8')	
##########################soc################
def soc(file,gap,out):
    ff = []
    filename, file_extension = os.path.splitext(file)
    df = pd.read_csv(file, header = None)
    df2 = pd.DataFrame(df[0].str.upper())
    for i in range(0,len(df2)):
        ff.append(len(df2[0][i]))
    if min(ff) < gap:
        print("Error: All sequences' length should be higher than :", gap)
        return 0
    mat1 = pd.read_csv("Model/Data/Schneider-Wrede.csv", index_col = 'Name')
    mat2 = pd.read_csv("Model/Data/Grantham.csv", index_col = 'Name')
    h1 = []
    h2 = []
    for n in range(1, gap+1):
        h1.append('SC' + str(n))
    for n in range(1, gap + 1):
        h2.append('G' + str(n))
    h1 = ['SOC'+str(gap)+'_'+sam for sam in h1]
    h2 = ['SOC'+str(gap)+'_'+sam for sam in h2]
    s1 = []
    s2 = []
    for i in range(0,len(df2)):
        for n in range(1, gap+1):
            sum = 0
            sum1 =0
            sum2 =0
            sum3 =0
            for j in range(0,(len(df2[0][i])-n)):
                sum = sum + (mat1[df2[0][i][j]][df2[0][i][j+n]])**2
                sum1 = sum/(len(df2[0][i])-n)
                sum2 = sum2 + (mat2[df2[0][i][j]][df2[0][i][j+n]])**2
                sum3 = sum2/(len(df2[0][i])-n)
            s1.append(sum1)
            s2.append(sum3)
    zz = np.array(s1).reshape(len(df2),gap)
    zz2 = np.array(s2).reshape(len(df2),gap)
    zz3 = round(pd.concat([pd.DataFrame(zz, columns = h1),pd.DataFrame(zz2,columns = h2)], axis = 1),4)
    zz3.to_csv(out, index = None, encoding = 'utf-8') 
##########################################CTC###################################
x = [1, 2, 3, 4, 5, 6,7]
p=[]
Y=[]
LS=[]


for i in range(len(x)):
    p=itertools.product(x,repeat=3)
    p=list(p)

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

for i in range(len(p)):
    LS.append(concatenate_list_data(p[i]))

def repstring(string):
    string=string.upper()
    char={"A":"1","G":"1","V":"1","I":"2","L":"2","F":"2","P":"2","Y":"3","M":"3","T":"3","S":"3","H":"4","N":"4","Q":"4","W":"4","R":"5","K":"5","D":"6","E":"6","C":"7"}
    string=list(string)
    for index,item in enumerate(string):
        for key,value in char.items():
            if item==key:
                string[index]=value
    return("".join(string))

def occurrences(string, sub_string):
    count=0
    beg=0
    while(string.find(sub_string,beg)!=-1) :
        count=count+1
        beg=string.find(sub_string,beg)
        beg=beg+1
    return count


def CTC(filename,out):
    df = pd.DataFrame(columns=['Sequence','Triad:Frequency'])
    data=list((pd.read_csv(filename,sep=',',header=None)).iloc[:,0])
    for i in range(len(data)):
        data1=''
        data1=str(data[i])
        data1=data1.upper()
        allowed = set(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
        is_data_invalid = set(data1).issubset(allowed)
        if is_data_invalid==False:
            print("Errror: Please check for invalid inputs in the sequence.","\nError in: ","Sequence number=",i+1,",","Sequence = ",data[i],",","\nNOTE: Spaces, Special characters('[@_!#$%^&*()<>?/\|}{~:]') and Extra characters(BJOUXZ) should not be there.")
            return
        df.at[i,'Sequence'] = data[i]
        Y.append("".join(repstring(str(data[i]))))
    val2=[[]]
    for f in range(len(LS)):
        val2[0]=val2[0]+["CTC_"+str(LS[f])]
    for j in range(len(data)):
        MM=[]
        for m in range(len(LS)):
            MM=MM+[occurrences(Y[j],LS[m])]
        Min_MM=min(MM)
        Max_MM=max(MM)
        if (Max_MM==0):
            print("Errror: Splits/ Sequence length should be greater than equal to 3")
            return
        val=[]
#         val.append(data[j])
        for k in range(len(LS)):
            val=val+[round(((occurrences(Y[j],LS[k])-Min_MM)/Max_MM),3)]
        val2.append(val)
#     print(val2)
    #file= open(sys.argv[2],'w', newline='')#output file
    file= open(out,'w', newline='')
    with file:
        writer=csv.writer(file);
        writer.writerows(val2);
    return val2

######################################CETD###################################
def ctd(file,out):
    attr=pd.read_csv("Model/Data/aa_attr_group.csv", sep="\t")
    filename, file_extension = os.path.splitext(file)
    df1 = pd.read_csv(file, header = None)
    df = pd.DataFrame(df1[0].str.upper())
    n = 0
    stt1 = []
    m = 1
    for i in range(0,len(attr)) :
        st =[]
        stt1.append([])
        for j in range(0,len(df)) :
            stt1[i].append([])
            for k in range(0,len(df[0][j])) :
                while m < 4 :
                    while n < len(attr.iloc[i,m]) :
                        if df[0][j][k] == attr.iloc[i,m][n] :
                            st.append(m)
                            stt1[i][j].append(m)
                        n += 2
                    n = 0
                    m += 1
                m = 1
#####################Composition######################
    f = open(os.path.join(output_directory,"compout_1"), 'w')
    sys.stdout = f
    std = [1,2,3]
    print("1,2,3,")
    for p in range (0,len(df)) :
        for ii in range(0,len(stt1)) :
            #for jj in stt1[ii][p]:
            for pp in std :
                count = 0
                for kk in stt1[ii][p] :
                    temp1 = kk
                    if temp1 == pp :
                        count += 1
                    composition = (count/len(stt1[ii][p]))*100
                print("%.2f"%composition, end = ",")
            print("")
    f.truncate()

#################################Transition#############
    tt = []
    tr=[]
    kk =0
    for ii in range(0,len(stt1)) :
        tt = []
        tr.append([])
        for p in range (0,len(df)) :
            tr[ii].append([])
            while kk < len(stt1[ii][p]) :
                if kk+1 <len(stt1[ii][p]):
                #if  stt1[ii][p][kk] < stt1[ii][p][kk+1] or stt1[ii][p][kk] > stt1[ii][p][kk+1]: # condition for adjacent values
                    tt.append(stt1[ii][p][kk])
                    tt.append(stt1[ii][p][kk+1])
                    tr[ii][p].append(stt1[ii][p][kk])
                    tr[ii][p].append(stt1[ii][p][kk+1])

                kk += 1
            kk = 0

    pp = 0
    xx = []
    xxx = []
    for mm in range(0,len(tr)) :
        xx = []
        xxx.append([])
        for nn in range(0,len(tr[mm])):
            xxx[mm].append([])
            while pp < len(tr[mm][nn]) :
                xx .append(tr[mm][nn][pp:pp+2])
                xxx[mm][nn].append(tr[mm][nn][pp:pp+2])
                pp+=2
            pp = 0

    f1 = open(os.path.join(output_directory,"compout_2"), 'w')
    sys.stdout = f1
    std1 = [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
    print("1->1,1->2,1->3,2->1,2->2,2->3,3->1,3->2,3->3,")
    for rr in range(0,len(df)) :
        for qq in range(0,len(xxx)):
            for tt in std1 :
                count = 0
                for ss in xxx[qq][rr] :
                    temp2 = ss
                    if temp2 == tt :
                        count += 1
                print(count, end = ",")
            print("")
    f1.truncate()

    #################################Distribution#############
    c_11 = []
    c_22 = []
    c_33 = []
    zz = []
    #print("0% 25% 50% 75% 100%")
    for x in range(0,len(stt1)) :
        #c_11.append([])
        c_22.append([])
        #c_33.append([])
        yy_c_1 = []
        yy_c_2 = []
        yy_c_3 = []
        ccc = []

        k = 0
        j = 0
        for y in range(0,len(stt1[x])):
            #c_11[x].append([])
            c_22[x].append([])
            for i in range(1,4) :
                cc = []
                c1 = [index for index,value in enumerate(stt1[x][y]) if value == i]
                c_22[x][y].append(c1)
    cc = []
    for ss in range(0,len(df)):
        for uu in range(0,len(c_22)):
            for mm in range(0,3):
                for ee in range(0,101,25):
                    k = (ee*(len(c_22[uu][ss][mm])))/100
                    cc.append(math.floor(k))
    f2 = open(os.path.join(output_directory,'compout_3'), 'w')
    sys.stdout = f2
    print("0% 25% 50% 75% 100%")
    for i in range (0,len(cc),5):
        print(*cc[i:i+5])
    f2.truncate()
    head = []
    header1 = ['CeTD_HB','CeTD_VW','CeTD_PO','CeTD_PZ','CeTD_CH','CeTD_SS','CeTD_SA']
    for i in header1:
        for j in range(1,4):
            head.append(i+str(j))
    df11 = pd.read_csv(os.path.join(output_directory,"compout_1"))
    df_1 = df11.iloc[:,:-1]
    zz = pd.DataFrame()
    for i in range(0,len(df_1),7):
        zz = pd.concat([zz, pd.DataFrame(pd.concat([df_1.loc[i],df_1.loc[i+1],df_1.loc[i+2],df_1.loc[i+3],df_1.loc[i+4],df_1.loc[i+5],df_1.loc[i+6]],axis=0)).transpose()], ignore_index=True)
    zz.columns = head
    #zz.to_csv(filename+".ctd_comp", index=None, encoding='utf-8')
    head2 = []
    header2 = ['CeTD_11','CeTD_12','CeTD_13','CeTD_21','CeTD_22','CeTD_23','CeTD_31','CeTD_32','CeTD_33']
    for i in header2:
        for j in ('HB','VW','PO','PZ','CH','SS','SA'):
            head2.append(i+'_'+str(j))
    df12 = pd.read_csv(os.path.join(output_directory,"compout_2"))
    df_2 = df12.iloc[:,:-1]
    ss = pd.DataFrame()
    for i in range(0,len(df_2),7):
        ss = pd.concat([ss, pd.DataFrame(pd.concat([df_2.loc[i], df_2.loc[i+1], df_2.loc[i+2], df_2.loc[i+3], df_2.loc[i+4], df_2.loc[i+5], df_2.loc[i+6]], axis=0)).transpose()], ignore_index=True)
    ss.columns = head2
    head3 = []
    header3 = ['CeTD_0_p','CeTD_25_p','CeTD_50_p','CeTD_75_p','CeTD_100_p']
    header4 = ['HB','VW','PO','PZ','CH','SS','SA']
    for j in range(1,4):
        for k in header4:
            for i in header3:
                head3.append(i+'_'+k+str(j))
    df_3 = pd.read_csv(os.path.join(output_directory,"compout_3"), sep=" ")
    rr = pd.DataFrame()
    for i in range(0,len(df_3),21):
        rr = pd.concat([rr, pd.DataFrame(pd.concat([df_3.loc[i], df_3.loc[i+1], df_3.loc[i+2], df_3.loc[i+3], df_3.loc[i+4], df_3.loc[i+5], df_3.loc[i+6], df_3.loc[i+7], df_3.loc[i+8], df_3.loc[i+9], df_3.loc[i+10], df_3.loc[i+11], df_3.loc[i+12], df_3.loc[i+13], df_3.loc[i+14], df_3.loc[i+15], df_3.loc[i+16], df_3.loc[i+17], df_3.loc[i+18], df_3.loc[i+19], df_3.loc[i+20]], axis=0)).transpose()], ignore_index=True)
    rr.columns = head3
    cotrdi= pd.concat([zz,ss,rr],axis=1)
    cotrdi.to_csv(out, index=None, encoding='utf-8')
    os.remove(os.path.join(output_directory,'compout_1'))
    os.remove(os.path.join(output_directory,'compout_2'))
    os.remove(os.path.join(output_directory,'compout_3'))

# Call each function with the appropriate arguments


file_path = sys.argv[1]
output_directory = sys.argv[2]
output_file = sys.argv[3]


Mw_out_file = os.path.join(output_directory,"Mw_output2")
molecular_weight_comp(file_path, Mw_out_file)


length_out_file = os.path.join(output_directory,"length_output2")
length_comp(file_path, length_out_file)

aac_out_file = os.path.join(output_directory,"aac_output2")
aac_comp(file_path, aac_out_file)

dpc_out_file = os.path.join(output_directory,"dpc_output2")
dpc_comp(file_path,1, dpc_out_file)

atc_out_file = os.path.join(output_directory,"atc_output2")
atc(file_path, atc_out_file)

btc_out_file = os.path.join(output_directory,"btc_output2")
bond(file_path, btc_out_file)

pcp_1_out_file = os.path.join(output_directory,"pcp_output2")
pcp_1(file_path, pcp_1_out_file)

rri_res_out_file = os.path.join(output_directory,"rri_output2")
RAAC(file_path, rri_res_out_file)

pri_res_out_file = os.path.join(output_directory,"pri_output2")
repeats(file_path, pri_res_out_file)

ddor_out_file = os.path.join(output_directory,"ddor_output2")
DDOR(file_path, ddor_out_file)

sep_out_file = os.path.join(output_directory,"sep_output2")
SE(file_path, sep_out_file)

ser_res_out_file = os.path.join(output_directory,"ser_output2")
SE_residue_level(file_path, ser_res_out_file)

spc_res_out_file = os.path.join(output_directory,"spc_output2")
shannons(file_path, spc_res_out_file)

paac_res_out_file = os.path.join(output_directory,"paac_output2")
paac(file_path, 1, paac_res_out_file)

apaac_out_file = os.path.join(output_directory,"apaac_output2")
apaac(file_path, 1, apaac_out_file)

qso_out_file = os.path.join(output_directory,"qso_output2")
qos(file_path, 1, qso_out_file)

ctc_out_file = os.path.join(output_directory,"ctc_output2")
CTC(file_path, ctc_out_file)

cetd_out_file = os.path.join(output_directory,"cetd_output2")
ctd(file_path, cetd_out_file)

dfs = []
for out_file in [Mw_out_file, length_out_file, aac_out_file, dpc_out_file, atc_out_file, btc_out_file, pcp_1_out_file,
                 rri_res_out_file, pri_res_out_file, ddor_out_file, ser_res_out_file, sep_out_file,
                 ctc_out_file, cetd_out_file,paac_res_out_file, apaac_out_file, qso_out_file,
                 spc_res_out_file]:
    df = pd.read_csv(out_file)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    dfs.append(df)
concatenated_df = pd.concat(dfs, axis=1)
concatenated_df.to_csv(output_file, index=False)

os.remove(os.path.join(output_directory,"Mw_output2"))
os.remove(os.path.join(output_directory,"length_output2"))
os.remove(os.path.join(output_directory,'aac_output2'))
os.remove(os.path.join(output_directory,'dpc_output2'))
os.remove(os.path.join(output_directory,'atc_output2'))
os.remove(os.path.join(output_directory,'btc_output2'))
os.remove(os.path.join(output_directory,'pcp_output2'))
os.remove(os.path.join(output_directory,'rri_output2'))
os.remove(os.path.join(output_directory,'pri_output2'))
os.remove(os.path.join(output_directory,'ddor_output2'))
os.remove(os.path.join(output_directory,'sep_output2'))
os.remove(os.path.join(output_directory,'ser_output2'))
os.remove(os.path.join(output_directory,'paac_output2'))
os.remove(os.path.join(output_directory,'apaac_output2'))
os.remove(os.path.join(output_directory,'spc_output2'))
os.remove(os.path.join(output_directory,'qso_output2'))
os.remove(os.path.join(output_directory,'ctc_output2'))
os.remove(os.path.join(output_directory,'cetd_output2'))
