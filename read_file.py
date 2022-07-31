
import pandas as pd
import pathlib


def extract():
    dataset = pathlib.Path(".keras3")

    train_dir_benign = dataset / 'train_extract'/'benign'
    extract_benign=pd.read_csv(train_dir_benign/'benign.csv',encoding='utf-8')
    where_to_exstract=dataset/'train'/'benign'
    target=extract_benign.pop('Info')
    count=0
    for i in target:
        f = open(format(where_to_exstract)+"\\"+format(count)+'.txt', 'w')
        f.write(i)
        count=count+1
        f.close()

    train_dir_blacklist = dataset / 'train_extract' / 'blacklist'
    extract_blacklist = pd.read_csv(train_dir_blacklist / 'blacklist.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'blacklist'
    target=extract_blacklist.pop('Info')
    count=0
    for i in target:
        f = open(format(where_to_exstract)+"\\"+format(count)+'.txt', 'w')
        f.write(i)
        count=count+1
        f.close()

    train_dir_time_based = dataset / 'train_extract' / 'blind'
    extract_time_based = pd.read_csv(train_dir_time_based / 'blind.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'blind'
    target = extract_time_based.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_dir_trav = dataset / 'train_extract' / 'directory_traversal'
    extract_dir_trav = pd.read_csv(train_dir_dir_trav / 'directory_traversal.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'directory_traversal'
    target = extract_dir_trav.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()


    train_dir_error_based = dataset / 'train_extract' / 'in_band'
    extract_error_based = pd.read_csv(train_dir_error_based / 'in_band2.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'in_band'
    target = extract_error_based.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_SSTI = dataset / 'train_extract' / 'SSTI'
    extract_SSTI = pd.read_csv(train_dir_SSTI / 'SSTI.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'SSTI'
    target = extract_SSTI.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_XSS = dataset / 'train_extract' / 'XSS'
    extract_XSS = pd.read_csv(train_dir_XSS / 'xss.csv',encoding='utf-8')
    where_to_exstract = dataset / 'train' / 'XSS'
    target = extract_XSS.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()


    ## now test vaulues

    train_dir_benign = dataset / 'test_extract' / 'benign'
    extract_benign = pd.read_csv(train_dir_benign / 'test_benign.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'benign'
    target = extract_benign.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_blacklist = dataset / 'test_extract' / 'blacklist'
    extract_blacklist = pd.read_csv(train_dir_blacklist / 'test_blacklist.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'blacklist'
    target = extract_blacklist.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_time_based = dataset / 'test_extract' / 'blind'
    extract_time_based = pd.read_csv(train_dir_time_based / 'test_blind.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'blind'
    target = extract_time_based.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_dir_trav = dataset / 'test_extract' / 'directory_traversal'
    extract_dir_trav = pd.read_csv(train_dir_dir_trav / 'test_directory_traversal.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'directory_traversal'
    target = extract_dir_trav.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()


    train_dir_error_based = dataset / 'test_extract' / 'in_band'
    extract_error_based = pd.read_csv(train_dir_error_based / 'test_in_band2.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'in_band'
    target = extract_error_based.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_SSTI = dataset / 'test_extract' / 'SSTI'
    extract_SSTI = pd.read_csv(train_dir_SSTI / 'test_SSTI.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'SSTI'
    target = extract_SSTI.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_XSS = dataset / 'test_extract' / 'XSS'
    extract_XSS = pd.read_csv(train_dir_XSS / 'test_xss.csv',encoding='utf-8')
    where_to_exstract = dataset / 'test' / 'XSS'
    target = extract_XSS.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()