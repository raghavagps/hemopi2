# HemoPI2.0
A method for predicting hemolytic activity of the peptides
# Introduction
HemoPI2.0 is developed for identification (Classification) as well as quantification (regression) methods for predicting hemolytic activity peptides with their hemolytic concentration (HC50 value), especially targets for mammalian red blood cells (RBCs). It uses various composition based features for predicting hemolytic activity peptides. The final model also deploys a motif-based module which has been implemented using MERCI. More information on HemoPI2.0 is available from its web server http://webs.iiitd.edu.in/raghava/hemopi2. Please read/cite the content about HemoPI2.0 for complete information including algorithm behind the approach.

## PIP Installation
PIP version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
pip install hemopi2
```
To know about the available option for the pip package, type the following command:
```
hemopi2_regression.py -h
hemopi2_classification.py -h
```

# Standalone

Standalone version of HemoPI2.0 is written in python3 and the following libraries are necessary for a successful run:

- scikit-learn
```
  pip install scikit-learn==1.3.1
```
- Pandas
- Numpy
- PyTorch: PyTorch is an open-source machine learning library. You can install it using pip (Python’s package installer). Open your terminal and type:
```
!pip install torch
```
- Transformers: The Transformers library provides state-of-the-art machine learning models like ESM. Install it with:
```
!pip install transformers
```
- ESM: ESM (Evolutionary Scale Modeling) is a library for protein sequence modeling.
```
!pip install git+https://github.com/facebookresearch/esm.git
```
# Important Note

- Due to large size of the model file, we have compressed model. 
- It is crucial to unzip the file before attempting to use the code or model. The compressed file must be extracted to its original form for the code to function properly.


# Regression
Predicts the Hazardous Concentration (HC50) or Half Maximum Effective Concentration (EC50) in μM. This indicates the concentration at which 50% of red blood cells (RBCs) are lysed. This model operates on the Random Forest Regressor (RFR) algorithm. 

**Minimum USAGE**
To know about the available option for the standalone, type the following command: 
```
hemopi2_regrssion.py -h
```
To run the example, type the following command:
```
hemopi2_regrssion.py -i peptide.fa

```
**Full Usage**: 
```
Following is complete list of all options, you may get these options
usage: hemopi2_regrssion.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
                     [-j {1,2,3,4}] 
                     [-d {1,2}]
                     [-wd Working Directory]
```
```
Please provide following arguments

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -j {1,2,3,4}, --job {1,2,3,4}
                        Job Type: 1: Predict, 2: Protein Scanning, 3: Design, 4: Design all possible mutants,by default 1
  -p POSITION, --Position POSITION
                        Position of mutation (1-indexed)
  -r RESIDUES, --Residues RESIDUES
                        Mutated residues (one or two of the 20 essential amino acids in upper case)
  -w {8,9,10,11,12,13,14,15,16,17,18,19,20}, --winleng {8,9,10,11,12,13,14,15,16,17,18,19,20}
                        Window Length: 8 to 20 (scan mode only), by default 8
  -d {1,2}, --display {1,2}
                        Display: 1: Hemolytic, 2: All peptides, by default 2
  -wd WORKING, --working WORKING
                        Working Directory: Location for writing results

```
**Input File**: It allow users to provide input in two format; i) FASTA format (standard) (e.g. peptide.fa) and ii) Simple Format. In case of simple format, file should have one peptide sequence in a single line in single letter code (eg. peptide.seq). 

**Output File**: Program will save result in CSV format, in case user do not provide output file name, it will be stored in outfile.csv.

**Jobs**:  In this program, two models have been incorporated;  1) Prediction for predicting given input peptide sequence as hemolytic and non-hemolytic peptide 2) Protein Scanning: for the prediction of hemolytic regions in a protein sequence. 3) Design: generates mutant peptides with a single amino acid or dipeptide at particulal position provided by user and predict their hemolytic activity 4) Design all possible mutants predict their hemolytic activity

**Position**: User can provide position at which he/she wants insert any single amino acid or dipeptide for creating mutation. This option is available for only Design module.

**Residue**: Mutated residues (one or two of the 20 essential amino acids in upper case) (e.g., A for Alanine)

**Window length**: User can choose any pattern length between 8 and 20 in long sequences. This option is available for only protein scan module.

**Working Directory**: Location for writing results

# Classification
Determines whether peptides are hemolytic or non-hemolytic based on their primary sequence. We have employed machine learning models and protein language models. The provided options include RF and ESM2-t6 models, as well as their hybrids with MERCI. You can select your preferred model for prediction. By default, this use the Hybrid1 (ESM2-t6+MERCI) approach, which has demonstrated best performance on our evaluation on independent dataset as well as runtime efficient.

**Minimum USAGE**
To know about the available option for the standalone, type the following command: 
```
hemopi2_classification.py -h
```
To run the example, type the following command:
```
hemopi2_classification -i peptide.fa

```
**Full Usage**: 
```
Following is complete list of all options, you may get these options
usage: toxinpred3.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-j {1,2,3,4}]
                     [-m {1,2,3,4}] 
                     [-d {1,2}]
                     [-wd Working Directory]
```
```
Please provide following arguments

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -j {1,2,3,4,5}, --job {1,2,3,4,5}
                        Job Type: 1: Predict, 2: Protein Scanning, 3: Design, 4: Design all possible mutants, 5: Motif Scanning, by default 1
  -m {1,2,3,4}, --model {1,2,3,4}
                        Model: 1: Random Forest, 2: Hybrid1 (RF+MERCI), 3: ESM2-t6, 4: Hybrid2 (ESM+MERCI) by default 4
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.46 (For RF and Hybrid1) and 0.55 (For ESM and Hybrid2)
  -p POSITION, --Position POSITION
                        Position of mutation (1-indexed)
  -r RESIDUES, --Residues RESIDUES
                        Mutated residues (one or two of the 20 essential amino acids in upper case)
  -w {8,9,10,11,12,13,14,15,16,17,18,19,20}, --winleng {8,9,10,11,12,13,14,15,16,17,18,19,20}
                        Window Length: 8 to 20 (scan mode only), by default 8
  -wd WORKING, --working WORKING
                        Working Directory: Location for writing results
  -d {1,2}, --display {1,2}
                        Display: 1: Hemolytic, 2: All peptides, by default 2

```

**Input File**: It allow users to provide input in two format; i) FASTA format (standard) (e.g. peptide.fa) and ii) Simple Format. In case of simple format, file should have one peptide sequence in a single line in single letter code (eg. peptide.seq). 

**Output File**: Program will save result in CSV format, in case user do not provide output file name, it will be stored in outfile.csv.

**Threshold**: User should provide threshold between 0 and 1, please note score is proportional to hemolytic potential of peptide.

**Jobs**:  In this program, two models have been incorporated;  1) Prediction for predicting given input peptide sequence as hemolytic and non-hemolytic peptide 2) Protein Scanning: for the prediction of hemolytic regions in a protein sequence. 3) Design: generates mutant peptides with a single amino acid or dipeptide at particulal position provided by user and predict their hemolytic activity 4) Design all possible mutants predict their hemolytic activity

**Models**:  In this program, four models have been incorporated;  i) Model1 for predicting given input peptide sequence as hemolytic and non-hemolytic peptide using Random Forest (RF) algorithm based on various composition based features using Pfeature tool of the peptide; 

ii) Model3 Model1 for predicting given input peptide sequence as hemolytic and non-hemolytic peptide using protein language model ESM2-t6.

iii) Model2 & Model4 for predicting given input peptide sequence as hemolytic and non-hemolytic peptide using Hybrid approach, the first ensemble is ESM2-t6 and MERCI second is RF and MERCI. It combines the scores generated from machine learning (ET) and protein language model (ESM2-t6), and MERCI as Hybrid Score, and the prediction is based on Hybrid Score.

**Position**: User can provide position at which he/she wants insert any single amino acid or dipeptide for creating mutation. This option is available for only Design module.

**Residue**: Mutated residues (one or two of the 20 essential amino acids in upper case) (e.g., A for Alanine)

**Window length**: User can choose any pattern length between 8 and 20 in long sequences. This option is available for only protein scan module.

**Working Directory**: Location for writing results



HemoPI2.0 Package Files
=======================
It contain following files, brief description of these files given below

INSTALLATION  	: Installation instructions

LICENSE       	: License information

merci : This folder contains the program to run MERCI

README.md     	: This file provide information about this package

hemopi2_regrssion.py 	:  Python program for regrssion

hemopi2_classification.py  :  Python program for classification

peptide.fa	: Example file contain peptide sequences in FASTA format

peptide.seq	: Example file contain peptide sequences in simple format

## Installation via PIP
User can install Hemopi2.0 via PIP also
```
pip install hemopi2
```

