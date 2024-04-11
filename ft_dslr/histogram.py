import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Histogram:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

        # Get the list of houses
        try:
            self.house_list = self.dataframe['Hogwarts House'].unique()
        except Exception as e:
            print("Error: cannot find", e, "column in the dataframe.")
            sys.exit(1)

        # Get the list of courses
        if len(self.dataframe.columns) < 7:
            print("Error: not enough columns in the dataframe to find courses.")
            sys.exit(1)
        self.courses_list = self.dataframe.columns[6:].to_list()

        # data = {course1: {house1: 0, house2: 0, house3: 0, house4: 0} ...}
        self.data = None


def set_data(histogram: Histogram):
    # initialize all house scores to 0
    histogram.data = {course: {house: 0 for house in histogram.house_list} for course in histogram.courses_list}
    # For each course, get all scores for each student in all house
    for course in histogram.courses_list:
        for house in histogram.house_list:
            # Get the scores of the students in the house for the course
            scores = histogram.dataframe[histogram.dataframe['Hogwarts House'] == house][course]
            # Remove NaN values
            scores = scores.dropna()
            # Get average
            average = scores.mean()
            histogram.data[course][house] = average


def get_dataframe(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

df = get_dataframe(sys.argv[1])
histogram = Histogram(df)
set_data(histogram)

for course in histogram.courses_list:
    print(histogram.data[course])

new_df = pd.DataFrame(histogram.data)
print(new_df['Astronomy'])
#

plt.hist(new_df['Astronomy']['Ravenclaw'], s=10, alpha=0.5, label='Ravenclaw')
plt.hist(new_df['Astronomy']['Slytherin'], s=10, alpha=0.5, label='Slytherin')
plt.hist(new_df['Astronomy']['Gryffindor'], s=10, alpha=0.5, label='Gryffindor')
plt.hist(new_df['Astronomy']['Hufflepuff'], s=10, alpha=0.5, label='Hufflepuff')
plt.show()


# -> Recuperer toutes les maisons possible
# -> Recuperer tous les cours possible
# -> Calculer la moyenne des notes des etudiants pour chaque cours de chaque maison ( 13 listes (cours) avec 4 listes (maisons) )
