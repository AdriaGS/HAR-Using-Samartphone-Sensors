#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:41:58 2017

@author: pau
"""
import os
import pandas as pd

def import_all_files(path):
    """
    Path points to a directory containing a directory per each
    user, each of which contains the csv files they collected.
    Returns a, b
    a: map user -> list of filenames recorded by the user
    b: list of all filenames of recordings available
    """
    users = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    files = []
    by_user = dict()
    for user in users:
        by_user[user] = [f for f in os.listdir(path + user) if '~' not in f]
        files.extend(by_user[user])
    return by_user, files

def rename_labels(df):
    # Remove emoji and leave only the main word
    if 'label' in df.columns:
        df['label'] = [label.split(' ')[0] for label in df['label']]
    else:
        print(' - No label column. Moving on.')

def splitfun(x):
    if (type(x) == str):
        return x.split(";")

    elif (pd.isnull(x)):
        return [""]

    else:
        print(type(x), x)
        raise Exception(" - Cannot split this value")

def get_ith_number(x, i):
    if len(x) > i:
        return pd.to_numeric(x[i])
    return None

def add_multidimensional_columns(df):
    try:
        multidimensional_columns = [c for c in df.columns if df[c].dtype == 'object' and 'android.sensor' in c and (len(df[c].iloc[df[c].first_valid_index()].split(';')) > 1)]
        for multicol in multidimensional_columns:
            try:
                list_col = df[multicol].apply(splitfun)
                num_axis = len(list_col.iloc[df[multicol].first_valid_index()])
                axis_names = ['x', 'y', 'z', 'x2', 'y2', 'z2']
                for i in range(num_axis):
                    df[multicol + '_' + axis_names[i]] = list_col.apply(lambda x: get_ith_number(x, i))
            except Exception as e:
                print("Error while separating %s into different columns" % multicol)
                raise(e)
    except Exception as e:
        print("Error while splitting columns")
        print([df[c].first_valid_index() for c in df.columns])
        raise(e)

def rename_columns(df):
    renamed_columns_map = {n: n.split('.')[-1] for n in df.columns if 'android.sensor.' in n}
    df.rename(index=str, columns=renamed_columns_map, inplace=True)

def post_process_df(df, feedback=False):

    if feedback:
        print('Renaming labels...')
    rename_labels(df)

    if feedback:
        print('Creating columns for each dimension...')
    add_multidimensional_columns(df)

    if feedback:
        print('Renaming columns...')
    rename_columns(df)

    return df
