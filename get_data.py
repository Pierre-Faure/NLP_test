#!/usr/bin/env python
# encoding: utf-8

"""
Script: get_data.py
Auteur: PF
Date: 16/04/2021 10:58
"""

# Imports
import pandas as pd
import os, os.path


# Functions

def get_data(file):
    df = pd.read_csv(file, index_col="index")
    print('Chargement de {} avis.'.format(df.shape[0]))
    return df


def get_bbc_data(dir='data/BBC News Summary/BBC News Summary'):
    data = pd.DataFrame(columns=['New', 'Summary'])
    categories = ["business", "entertainment", "politics", "sport", "tech"]

    for category in categories:
        news_dir = dir + "/News Articles/" + category + "/"
        summ_dir = dir + "/Summaries/" + category + "/"
        for file in os.listdir(news_dir):
            news_path = news_dir + file
            summ_path = summ_dir + file

            news = open(news_path, 'r')
            summ = open(summ_path, 'r')

            row = {'New': news.read(), 'Summary': summ.read(), 'Category': category}

            data = data.append(row, ignore_index=True)

    return data

# Main
def main():
    pass


if __name__ == '__main__':
    main()
