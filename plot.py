from pprint import pprint

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from sql_manager import SqlManager
from ECA import one_max, peak, trap
import numpy as np
import matplotlib.pyplot as plt


def str_to_list(string):
    return [int(ch) for ch in string]


def query_by_fitness_function(function_name, data_frame):
    return data_frame.query(f"fitness_function == '{function_name}'")


def calculate_fitness_value(string, fitness_function):
    return fitness_function(str_to_list(string))


def calculate_color(row):
    problem_size = int(row["problem_size"])
    if problem_size == 10:
        row["color"] = "y"
    elif problem_size == 20:
        row["color"] = "g"
    elif problem_size == 40:
        row["color"] = "b"
    elif problem_size == 60:
        row["color"] = "m"
    elif problem_size == 80:
        row["color"] = "r"

    return row


def draw_plot1(input_df, name):
    fig = plt.figure(f"{name} ")

    ax = Axes3D(fig)

    for index, (prob_size, color) in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')]):
        data_frame = input_df.loc[input_df["problem_size"] == prob_size]
        pop_sizes = input_df["pop_size"].copy().drop_duplicates()
        max_gens = input_df["max_gen"].copy().drop_duplicates()
        df2 = data_frame.copy()
        data = []
        for max_gen in max_gens.tolist():
            raw = []
            error_raw = []
            max_gen_df = df2.loc[df2["max_gen"] == max_gen].copy()
            for pop_size in pop_sizes.tolist():
                value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["best_fitness"].values[0]
                raw.append(value)
            data.append(raw)

        data = np.array(data)

        lx = len(data[0])  # Work out matrix dimensions
        ly = len(data[:, 0])
        xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + index / 10, ypos + 0.05)
        #
        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        dx = 0.1 * np.ones_like(zpos)
        dy = dx.copy()
        dz = data.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
    r = np.linspace(0, 1, 10)
    for i, item in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')], start=1):
        plt.plot(0, 0, color=item[1], label=str(item[0]))
    plt.legend(loc='best')
    ax.set_xlabel("POP_SIZE")
    ax.set_ylabel('MAX_GEN')
    ax.set_zlabel('BEST_FITNESS')
    x_labels = []
    for x in pop_sizes.tolist():
        x_labels.append(" ")
        x_labels.append(x)

    y_labels = []
    for y in max_gens.tolist():
        y_labels.append(" ")
        y_labels.append(y)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()


def draw_plot2(input_df, name):
    fig = plt.figure(f"{name} ")

    ax = Axes3D(fig)

    for index, (prob_size, color) in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')]):
        data_frame = input_df.loc[input_df["problem_size"] == prob_size]
        pop_sizes = input_df["pop_size"].copy().drop_duplicates()
        max_gens = input_df["max_gen"].copy().drop_duplicates()
        df2 = data_frame.copy()
        data = []
        for max_gen in max_gens.tolist():
            raw = []
            max_gen_df = df2.loc[df2["max_gen"] == max_gen].copy()
            for pop_size in pop_sizes.tolist():
                value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["fitness_value"].values[0]
                raw.append(value)
            data.append(raw)

        data = np.array(data)

        lx = len(data[0])  # Work out matrix dimensions
        ly = len(data[:, 0])
        xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + index / 10, ypos + 0.05)
        #
        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        dx = 0.1 * np.ones_like(zpos)
        dy = dx.copy()
        dz = data.flatten()
        # d_err = error_data.flatten()
        # err_pos = dz.copy() - d_err / 2

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
        # ax.bar3d(xpos, ypos, err_pos, 0.1, 0.1, d_err, color="black")
    r = np.linspace(0, 1, 10)
    for i, item in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')], start=1):
        plt.plot(0, 0, color=item[1], label=str(item[0]))
    plt.legend(loc='best')
    ax.set_xlabel("POP_SIZE")
    ax.set_ylabel('MAX_GEN')
    ax.set_zlabel('FITNESS')
    x_labels = []
    for x in pop_sizes.tolist():
        x_labels.append(" ")
        x_labels.append(x)

    y_labels = []
    for y in max_gens.tolist():
        y_labels.append(" ")
        y_labels.append(y)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()


def draw_plot3(input_df, name):
    fig = plt.figure(f"{name} ")

    ax = Axes3D(fig)

    for index, (prob_size, color) in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')]):
        data_frame = input_df.loc[input_df["problem_size"] == prob_size]
        pop_sizes = input_df["pop_size"].copy().drop_duplicates()
        max_gens = input_df["max_gen"].copy().drop_duplicates()
        df2 = data_frame.copy()
        data = []
        for max_gen in max_gens.tolist():
            raw = []
            error_raw = []
            max_gen_df = df2.loc[df2["max_gen"] == max_gen].copy()
            for pop_size in pop_sizes.tolist():
                value = max_gen_df.loc[max_gen_df["pop_size"] == pop_size]["time"].values[0]
                raw.append(value)
            data.append(raw)

        data = np.array(data)

        lx = len(data[0])  # Work out matrix dimensions
        ly = len(data[:, 0])
        xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + index / 10, ypos + 0.05)
        #
        xpos = xpos.flatten()  # Convert positions to 1D array
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)
        dx = 0.1 * np.ones_like(zpos)
        dy = dx.copy()
        dz = data.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
    r = np.linspace(0, 1, 10)
    for i, item in enumerate([(10, 'y'), (20, 'g'), (40, 'b'), (60, 'm'), (80, 'r')], start=1):
        plt.plot(0, 0, color=item[1], label=str(item[0]))
    plt.legend(loc='best')
    ax.set_xlabel("POP_SIZE")
    ax.set_ylabel('MAX_GEN')
    ax.set_zlabel('TIME')
    x_labels = []
    for x in pop_sizes.tolist():
        x_labels.append(" ")
        x_labels.append(x)

    y_labels = []
    for y in max_gens.tolist():
        y_labels.append(" ")
        y_labels.append(y)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.show()


if __name__ == '__main__':
    sql_manager = SqlManager("information.sqlite")
    df = pd.read_sql(sql="select * from information", con=sql_manager.conn)
    for func in [one_max, peak, trap]:
        func_df = query_by_fitness_function(func.__name__, df)
        print("____________________")
        # group_df = func_df.groupby(["problem_size", "pop_size", "max_gen"]).agg(
        #     {'fitness_value': ['mean', 'std'], 'time': ['mean', 'std']}).reset_index()
        group_df = func_df.apply(calculate_color, axis=1)
        print(group_df)
        draw_plot1(group_df, func.__name__)
        draw_plot2(group_df, func.__name__)
        draw_plot3(group_df, func.__name__)
