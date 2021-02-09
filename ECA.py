import random
import time
import numpy as np
from sql_manager import SqlManager
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def one_max(data):
    return np.sum(data)


def peak(data):
    return np.prod(data)


def trap(data: str):
    return 3 * len(data) * peak(data) - one_max(data)


class ECA:
    def __init__(self, population_size, problem_size, fitness_function, max_gen):
        self.population_size = int(population_size / 2) * 2
        self.problem_size = problem_size
        self.fitness_function = fitness_function
        self.population = np.random.randint(2, size=(self.population_size, self.problem_size))
        self.max_gen = max_gen
        self.generation = 1
        self.best_chromosome = np.ones((self.problem_size,), dtype=int)
        self.mutation_probability = 0.2
        self.selection_rate = 0.8
        self.population_df = pd.DataFrame(self.population)
        self.population_df["fitness"] = self.population_df.apply(func=self.fitness_function, axis=1)
        self.fitness_mean = np.mean(self.population_df["fitness"])
        self.coefficient_selection = 30
        self.mutation_probability_history = [self.mutation_probability]
        self.selection_rate_history = [self.selection_rate]
        self.coefficient_selection_history = [self.coefficient_selection]
        self.fitness_mean_history = [self.fitness_mean]
        self.best_result = self.population_df.sort_values(by=["fitness"], ascending=False).iloc[:1, :]

    def terminate(self):
        if self.generation >= self.max_gen:
            return True
        return True in self.population_df.iloc[:, :-1].apply(lambda row: list(row) == list(self.best_chromosome),
                                                             axis=1).values

    def uniform_cross_over(self, chromosome1, chromosome2, p=0.8):
        if random.random() > p:
            return np.array(chromosome1), np.array(chromosome2)
        else:
            new_chromosome1 = np.array(chromosome1)
            new_chromosome2 = np.array(chromosome2)
            for gen_index in range(self.problem_size):
                rand = random.randint(1, 2)
                if rand == 1:
                    pass
                else:
                    gen = new_chromosome1[gen_index]
                    new_chromosome1[gen_index] = new_chromosome2[gen_index]
                    new_chromosome2[gen_index] = gen

            return new_chromosome1, new_chromosome2

    def bit_flipping(self, chromosome):
        chromosome = np.vectorize(
            lambda item: (0 if item == 1 else 1) if random.random() <= self.mutation_probability else item)(chromosome)
        return chromosome

    def generate_offsprings(self):
        shuffle_pop = self.population_df.sample(frac=1)
        off_springs = []

        for i in range(0, self.population_size, 2):
            parents = shuffle_pop.iloc[i:i + 2, :-1].values
            offspring_pair = self.uniform_cross_over(parents[0], parents[1], p=0.8)
            off_springs.append(self.bit_flipping(offspring_pair[0]))
            off_springs.append(self.bit_flipping(offspring_pair[1]))

        return pd.DataFrame(off_springs)

    def update_fitness_mean(self):
        self.fitness_mean = np.mean(self.population_df["fitness"])

    def update_mutation_probability_selection_rate(self):
        before_fitness_mean = self.fitness_mean
        self.update_fitness_mean()
        self.coefficient_selection = self.coefficient_selection * (1.5 - self.selection_rate)
        if before_fitness_mean is not None:
            # print(f"delta_F = {delta_f}")
            # self.selection_rate = (math.e ** delta_f - math.e ** delta_f) / (math.e ** delta_f + math.e ** delta_f)
            self.selection_rate = 30 * (abs(before_fitness_mean - self.fitness_mean) / self.problem_size) + 0.3
            # self.selection_rate = (abs(before_fitness_mean - self.fitness_mean) / (before_fitness_mean + 1)) + 0.7
            if self.selection_rate > 0.9:
                self.selection_rate = 0.9

            self.mutation_probability = 1 - self.selection_rate

    def update_population(self, offsprings):
        all_chromosome_df = pd.concat([self.population_df.iloc[:, :-1], pd.DataFrame(offsprings)])
        all_chromosome_df["fitness"] = all_chromosome_df.apply(self.fitness_function, axis=1)
        all_chromosome_df = all_chromosome_df.sort_values(by=["fitness"], ascending=False)
        top_select_number = int(self.selection_rate * self.population_size)
        self.population_df = all_chromosome_df.iloc[:1, :]
        self.population_df = self.population_df.append(
            all_chromosome_df.iloc[1:self.population_size, :].sample(n=top_select_number))

        self.population_df = self.population_df.append(
            all_chromosome_df.sample(n=self.population_size - top_select_number - 1))

    def update_histories(self):
        self.mutation_probability_history.append(self.mutation_probability)
        self.coefficient_selection_history.append(self.coefficient_selection)
        self.fitness_mean_history.append(self.fitness_mean)
        self.selection_rate_history.append(self.selection_rate)

    def run(self):
        while not self.terminate():
            offsprings = self.generate_offsprings()
            self.update_population(offsprings)
            self.generation += 1
            self.update_mutation_probability_selection_rate()
            self.update_histories()
            self.best_result = self.population_df.iloc[:1, :]

    def draw_plots(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        plt.plot(range(self.generation), self.fitness_mean_history)
        plt.xlabel("generation")
        plt.ylabel("mean_fitness")
        plt.savefig(f"{folder_path}fitness_mean_history.png")
        plt.close()

        plt.plot(range(self.generation), self.selection_rate_history)
        plt.xlabel("generation")
        plt.ylabel("selection_rate")
        plt.savefig(f"{folder_path}selection_rate_history.png")
        plt.close()

        plt.plot(range(self.generation), self.coefficient_selection_history)
        plt.xlabel("generation")
        plt.ylabel("coefficient_selection")
        plt.savefig(f"{folder_path}coefficient_selection_history.png")
        plt.close()

        plt.plot(range(self.generation), self.mutation_probability_history)
        plt.xlabel("generation")
        plt.ylabel("mutation_probability")
        plt.savefig(f"{folder_path}mutation_probability_history.png")
        plt.close()


if __name__ == '__main__':
    sql_manager = SqlManager(file="information.sqlite")
    sql_manager.create_database()
    for problem_size in [10, 20, 40, 60, 80]:
        for function in [one_max, peak, trap]:

            for pop_size in [100, 200, 300]:
                for max_gen in [100, 200, 300]:
                    time1 = time.time()
                    eca_list = []
                    generations = []
                    fitness_values = []
                    for i in range(10):
                        print("\n_________________________________________________")
                        print(
                            f"problem_size={problem_size}\npop_size={pop_size}\nmax_gen={max_gen}\nfitness ={function.__name__}\niteration={i}")
                        eca = ECA(population_size=pop_size, problem_size=problem_size, fitness_function=function,
                                  max_gen=max_gen)
                        eca.run()
                        eca_list.append(eca)
                        fitness_values.append(eca.best_result.values[0][-1])
                        generations.append(eca.generation)

                    best_sga = max(eca_list, key=lambda item: item.best_result.values[0][-1])
                    best_sga.draw_plots(f"outs/{function.__name__}_{problem_size}_{pop_size}_{max_gen}/")
                    sql_manager.add_row(fitness_function=function.__name__, max_gen=max_gen, problem_size=problem_size,
                                        pop_size=pop_size, result="".join(
                            [str(bit) for bit in best_sga.best_result.iloc[:, :-1].values[0]]),
                                        generation=statistics.mean(generations),
                                        fitness_value=statistics.mean(fitness_values),
                                        best_fitness=best_sga.best_result.values[0][-1], time=time.time() - time1)
