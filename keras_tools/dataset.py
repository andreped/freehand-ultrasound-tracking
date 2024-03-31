import os
from copy import copy
from math import ceil
from os.path import join
from random import randint, shuffle
from typing import List
import numpy as np


def create_validation_set(set, subjects_for_validation, exclude=[]):
    validation_set = []
    while len(validation_set) < subjects_for_validation:
        subject = set[randint(0, len(set) - 1)]
        while subject in exclude:
            subject = set[randint(0, len(set) - 1)]

        exclude.append(subject)
        validation_set.append(subject)

    return validation_set


def get_dataset_files(path, exclude=None, only=None, prefix='./', postfix='.hd5'):
    filelist = []
    for file in os.listdir(path):
        if file[-3:] != 'hd5':
            continue
        exclude_file = False
        if only:
            exclude_file = True
            for pattern in only:
                if ('./' + file).find(prefix + str(pattern) + postfix) >= 0:
                    exclude_file = False
                    break
        elif exclude:
            exclude_file = False
            for ex in exclude:
                if ('./' + file).find(prefix + str(ex) + postfix) >= 0:
                    exclude_file = True
                    break
        if exclude_file:
            continue
        filelist.append(join(path, file))

    return filelist


def create_cross_validation_file(patients: List, nb_folds: int, filepath: str, overwrite: bool = False, test: bool = True):
    """ Create a cross validation file. Assumes one hd5 files per patient."""
    if os.path.exists(filepath) and not overwrite:
        print('Cross validation file already exists, doing nothing.')
        return

    shuffle(patients)
    test_patients_per_fold = ceil(len(patients) / nb_folds)
    print('Test patients per fold: ', test_patients_per_fold)

    file = open(filepath, 'w')
    for fold in range(nb_folds):
        test_set = patients[test_patients_per_fold * fold:test_patients_per_fold * fold + test_patients_per_fold]
        # Use next fold as validation set
        val_set = patients[test_patients_per_fold * ((fold + 1) % nb_folds):test_patients_per_fold * (
                    (fold + 1) % nb_folds) + test_patients_per_fold]

        # Training set is all of the other patients
        training_set = copy(patients)
        for x in test_set:
            training_set.remove(x)
        for x in val_set:
            training_set.remove(x)
        print('Fold:', fold)
        print('Train', len(training_set), training_set)
        print('Test', len(test_set), test_set)
        print('Val', len(val_set), val_set)
        file.write('Fold ' + str(fold) + '\n')
        file.write(' '.join([str(x) for x in training_set]) + '\n')
        file.write(' '.join([str(x) for x in val_set]) + '\n')
        file.write(' '.join([str(x) for x in test_set]) + '\n')
    file.close()

    if not test:
        return

    # Check that everything is correct
    print('Testing folds for errors ....')
    all_test_data = []
    for i in range(nb_folds):
        train, val, test = get_fold(i, filepath)
        # Test and val sets should not overlap
        total = val + test + train
        if len(total) != len(np.unique(total)):
            raise Exception('Error in folds: Duplicates detected')
        all_test_data.extend(test)

    # Test sets should cover all patients
    if len(all_test_data) != len(patients):
        raise Exception('Error in folds: Test data does not cover all patients: ' + str(len(all_test_data)) + ' vs ' + str(len(patients)))

    print('Tests OK.')


def get_fold(fold: int, filepath: str, test:bool = True):
    """ Get training, validation and set for fold i, stored in filename"""

    with open(filepath) as f:
        # Skip to correct fold in file
        for i in range(fold):
            _ = f.readline()
            train_line = f.readline()
            val_line = f.readline()
            test_line = f.readline()

        _ = f.readline()
        train_line = f.readline().strip()
        val_line = f.readline().strip()
        test_line = f.readline().strip()
        train_set = train_line.split(' ')
        train_set = [x for x in train_set]
        val_set = val_line.split(' ')
        val_set = [x for x in val_set]
        test_set = test_line.split(' ')
        test_set = [x for x in test_set]

    total = val_set + test_set + train_set
    if len(total) != len(np.unique(total)):
        raise Exception('Error in fold: Duplicates detected')

    return train_set, val_set, test_set


def get_number_of_folds(filepath: str):
    with open(filepath) as f:
        lines = 0
        for line in f:
            if line.strip() == '':
                break
            lines += 1
        return int(lines / 4)
