# coding:utf-8
import jieba
import os
import numpy as np
import re

def shuffle(d):
    return np.random.permutation(d)

def shuffle2(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return d

def dropout(d, p=0.3):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d


def clean(xx):
    xx2 = re.sub(r'\?', "", xx)
    xx1= xx2.split(' ')
    return xx1


def dataaugment(X, times):
    l = len(X)
    while times != 0:
        for i in range(int(l/2)):
            item = clean(X[i])
            d1 = shuffle2(item)
            d11 = ' '.join(d1)
            X.extend([d11])
        for i in range(int(l/2), l):
            item = clean(X[i])
            d2 = dropout(item)
            d22 = ' '.join(d2)
            X.extend([d22])
        times -= 1
    return X

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def FenCiAug(path, read_filename, write_file, class_proportion):
    read_file = open(path + read_filename, 'rb')
    read_file.readline()
    xs = []
    for i in range(len(class_proportion)):
        xs.append([])
    xs_p = []
    for line in read_file:
        #数据增强准备
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').strip('\r').split('\t')
        xs[int(line_split[-1].strip())].append(line_split[1].strip())
    for i in range(len(class_proportion)):
        xs_p.append(dataaugment(xs[i], class_proportion[i]-1))
    for i in range(len(xs_p)):
        for text in xs_p[i]:
            # print(text)
            newline = jieba.cut(find_chinese(text), cut_all=False)
            write_line = str(i) + ',' + ''.join(newline) + '\n'
            write_line = write_line.encode('utf-8', 'ignore')
            write_file.write(write_line)
    read_file.close()


def FenCi(path, read_filename, write_filename, is_train):
    read_file = open(path + read_filename, 'rb')
    write_file = open(path + write_filename, 'wb')
    if is_train:
        write_file.write("label,ques\n".encode('utf-8'))
    else:
        write_file.write("ques\n".encode('utf-8'))
    read_file.readline()
    for line in read_file:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').strip('\r').split('\t')
        newline = jieba.cut(find_chinese(line_split[1]), cut_all=False)
        if is_train:
            write_line = line_split[-1] + ',' + ' '.join(newline) + '\n'
        else:
            write_line = ' '.join(newline) + '\n'
        write_line = write_line.encode('utf-8', 'ignore')
        write_file.write(write_line)
    write_file.close()
    read_file.close()

"""

[73034, 25466, 500, 1000, 2000]

"""

