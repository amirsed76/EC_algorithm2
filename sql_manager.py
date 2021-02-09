import sqlite3


class SqlManager:
    def __init__(self, file):
        self.conn = sqlite3.connect(file)
        self.crs = self.conn.cursor()

    def create_database(self):
        self.crs.execute("CREATE TABLE IF NOT EXISTS information ("
                         "fitness_function VARCHAR(100) NOT NULL  ,"
                         "problem_size INT NOT NULL , "
                         "max_gen INT NOT NULL ,"
                         "pop_size INT NOT NULL , "
                         "result VARCHAR(110) NOT NULL , "
                         "generation FLOAT NOT NULL ,"
                         "fitness_value FLOAT NOT NULL ,"
                         "best_fitness INT NOT NULL , "
                         "time FLOAT NOT NULL"
                         ")")

        self.crs.execute("DELETE FROM information")
        self.conn.commit()

    def add_row(self, fitness_function, max_gen, problem_size, pop_size, result, generation, fitness_value, time,
                best_fitness):
        sql = f"INSERT INTO information values ('{fitness_function}' , {problem_size} , {max_gen} , {pop_size} , '{result}'  ,{generation},{fitness_value} , {best_fitness} , {time} ) "
        self.crs.execute(sql)
        self.conn.commit()
