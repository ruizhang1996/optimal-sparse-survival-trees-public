import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import os

def read_data(filename, has_header=True, has_index=False, sep=',', decimal='.', missing_value='null'):
    """
    Assume file has proper format:

    Users should specify missing value if present.
    No delimiter allowed in entry.
    First line is header if has_header=True.
    First column is index if has_index=True.
    Last column is target.

    """
    data = []
    types = []
    with open(filename, 'r', encoding='unicode_escape') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]  # strip empty lines
        if sep == 't':
            sep = '\t'
        first_line = lines[0].split(sep)
        first_line = list(filter(None, first_line))

        num_col = len(first_line)
        if has_header:
            header = first_line
        else:
            header = list(map(str, range(not (has_index), num_col)))
            header = ['feature_' + e for e in header] + ['target']

        if has_index:
            header = header[1:]

        for line in lines[has_header:]:
            line = line.split(sep)[has_index:]
            line = list(filter(None, line))
            line = [e.strip() for e in line]
            if len(line) < len(header): continue
            if len(types) : break
            if missing_value not in line:
                for element in line:
                    if decimal in element:
                        types.append('float')
                    elif element.isdigit():
                        types.append('int')
                    else:
                        types.append('O')

        for line in lines[has_header:]:

            if decimal == ',':
                line = line.replace(',', '.')

            line = line.split(sep)[has_index:]
            line = list(filter(None, line))
            line = [e.strip() for e in line]

            if missing_value in line or len(line) < len(header):
                continue  # skip lines contain missing values

            for i, e in enumerate(line):
                if types[i] == 'int' and '.' in e:
                    types[i] = 'float'

            data.append(tuple(line))

    dtype = list(zip(header, types))
    data = np.array(data, dtype=dtype)

    return pd.DataFrame(data)


def encode_feature(df, filename, num_bin='all'):
    """
    transform features to binary features
    all categorical values will be encoded to binary, user should mannually determine cateogorical columns

    data: pandas dataframe
    num_bin: # threshold used to encode continuous feature that has threshold over 10

    """
    root_ext = os.path.splitext(filename)
    # encode categorical features
    df_categorical = df.select_dtypes(include=object)
    if df_categorical.shape[1]:
        df_categorical = pd.get_dummies(df_categorical, drop_first=True)
    # encode continuous features
    df_real = pd.DataFrame()
    if num_bin == 'all':
        for col in df.iloc[:, :-1].select_dtypes(exclude=object).columns:
            df_real[col] = df[col]
    else:
        with open(root_ext[0] + '_intervals.txt', 'w') as f:
            for col in df.iloc[:, :-1].select_dtypes(exclude=object).columns:
                if df[col].nunique() > 10:
                    df_real[col], interval = pd.cut(df[col], bins=int(num_bin), labels=False, retbins=True)
                    f.write(col + ":  ")
                    f.write(",".join(map(str,interval)) + '\n')
                else:
                    df_real[col] = df[col]

    df_real = pd.get_dummies(df_real.astype(str), drop_first=True)
    df_binary = pd.concat([df_categorical, df_real, df.iloc[:, -1]], axis=1)

    df_binary.to_csv(root_ext[0] + '_binary.csv', index=False)

    return df_binary



def train_test_split(filename, fold=5):
    """
    split dataset to training and testing

    filename: dataset should be binarized
    """
    df = pd.read_csv(filename)
    root_ext = os.path.splitext(filename)

    for i, e in enumerate(KFold(n_splits=fold, shuffle=True, random_state=666).split(df)):
        train_index = e[0]
        test_index = e[1]
        df.iloc[train_index].to_csv(root_ext[0] + '_train_fold_' + str(i + 1) + '.csv', index=False)
        df.iloc[test_index].to_csv(root_ext[0] + '_test_fold_' + str(i + 1) + '.csv', index=False)


def create_dataset_for_evtree(filename):
    """
    :param filename: binary feature csv, no index column
    :return:
    """
    old = pd.read_csv(filename)
    new = pd.DataFrame()
    for col in old.columns[:-1]:
        new[col] = np.where(old[col] == 1, 'yes', 'no')

    data = pd.concat([new, old.iloc[:, -1]], axis=1)
    dir_path = os.path.dirname(filename) + '/evtree_data/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    data.to_csv(dir_path + os.path.basename(filename),  index=False)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("file_name", type=str, help="dataset file name")
parser.add_argument("--sep", type=str, default=",", help="dataset seperator")
parser.add_argument("--dec", type=str, default=".", help="dataset decimal")
parser.add_argument("--missing", type=str, default="nan", help="dataset missing value")
parser.add_argument("-h", "--header", action="store_false", help="dataset has no header")
parser.add_argument("-i", "--index", action="store_true", help="dataset has index")
parser.add_argument("-b", "--binarize", type=str, default='all', help="binarize raw dataset")
parser.add_argument("-s", "--split", type=int, help="train test split")
parser.add_argument("-e", "--evtree", help="generate evtree dataset",action="store_true")
parser.add_argument("-A", "--All", action="store_true", help="do all above work")

args = parser.parse_args()
file_name = args.file_name
sep = args.sep
dec = args.dec
missing = args.missing
has_header = args.header
has_index = args.index
bin_num = args.binarize
fold = args.split
generate_evtree_dataset = args.evtree

if args.All:
    df = read_data(file_name, has_header=has_header, has_index=has_index, sep=sep, decimal=dec, missing_value=missing)
    df_binary = encode_feature(df, file_name, num_bin=bin_num)
    file_name_binary = os.path.splitext(file_name)[0] + '_binary.csv'
    if not fold:
        fold = 5
    train_test_split(file_name_binary, fold=fold)

    create_dataset_for_evtree(file_name_binary)
    for f in range(fold):
        create_dataset_for_evtree(os.path.splitext(file_name_binary)[0] + '_train_fold_' + str(f + 1) + '.csv')
        create_dataset_for_evtree(os.path.splitext(file_name_binary)[0] + '_test_fold_' + str(f + 1) + '.csv')

elif generate_evtree_dataset:
    create_dataset_for_evtree(file_name)

elif fold:
    train_test_split(file_name, fold=fold)

else:
    df = read_data(file_name, has_header=has_header, has_index=has_index, sep=sep, decimal=dec, missing_value=missing)
    encode_feature(df, file_name, num_bin=bin_num)
