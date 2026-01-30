# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Integer
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import math


# Database setup

engine = create_engine("sqlite:///assignment.db", echo=False)
Base = declarative_base()


# Database Models

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)


class IdealFunctions(Base):
    __tablename__ = "ideal_functions"
    id = Column(Integer, primary_key=True)
    x = Column(Float)


class TestResults(Base):
    __tablename__ = "test_results"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_function = Column(Integer)


Base.metadata.create_all(engine)


# Base class (Inheritance)

class Dataset:
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def get_dataframe(self):
        return self.data


# Training Dataset

class TrainingDataset(Dataset):
    pass


# Ideal Dataset

class IdealDataset(Dataset):
    pass


# Test Dataset

class TestDataset(Dataset):
    pass


# Core Logic

class IdealFunctionSelector:
    """
    Selects the best ideal functions using least squares
    """

    def __init__(self, training_df, ideal_df):
        self.training_df = training_df
        self.ideal_df = ideal_df
        self.best_functions = {}

    def select_best_functions(self):
        for i in range(1, 5):
            train_y = self.training_df[f"y{i}"]
            min_error = float("inf")
            best_col = None

            for col in self.ideal_df.columns[1:]:
                error = np.sum((train_y - self.ideal_df[col]) ** 2)
                if error < min_error:
                    min_error = error
                    best_col = col

            self.best_functions[f"y{i}"] = best_col

        return self.best_functions


# Test Data Mapping

class TestDataMapper:
    """
    Maps test data to ideal functions using deviation criterion
    """

    def __init__(self, test_df, ideal_df, best_functions, training_df):
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.best_functions = best_functions
        self.training_df = training_df

    def map_test_data(self):
        results = []

        for _, row in self.test_df.iterrows():
            x = row["x"]
            y = row["y"]
            min_dev = float("inf")
            best_func = None

            for train_col, ideal_col in self.best_functions.items():
                ideal_y = self.ideal_df.loc[self.ideal_df["x"]
                                            == x, ideal_col].values
                if len(ideal_y) == 0:
                    continue

                deviation = abs(y - ideal_y[0])

                max_train_dev = np.max(
                    abs(self.training_df[train_col] - self.ideal_df[ideal_col])
                )

                if deviation <= max_train_dev * math.sqrt(2):
                    if deviation < min_dev:
                        min_dev = deviation
                        best_func = ideal_col

            results.append([x, y, min_dev if best_func else None, best_func])

        return pd.DataFrame(
            results,
            columns=["x", "y", "delta_y", "ideal_function"]
        )


# Visualization (Bokeh)

def visualize(training_df, ideal_df, best_functions, test_results):
    p = figure(title="Ideal Function Mapping", width=900, height=500)

    for key, col in best_functions.items():
        p.line(
            ideal_df["x"],
            ideal_df[col],
            legend_label=f"Ideal {col}",
            line_width=2
        )

    p.circle(
        test_results["x"],
        test_results["y"],
        size=6,
        color="red",
        legend_label="Test Data"
    )

    show(p)


# Visualization (Matplotlib)

def plot_train_vs_ideal(training_df, ideal_df, best_functions):
    """
    Line plot: Training data vs best-fit ideal functions
    """
    plt.figure(figsize=(10, 6))

    for train_col, ideal_col in best_functions.items():
        plt.plot(
            training_df["x"],
            training_df[train_col],
            label=f"Train {train_col}",
            linestyle="dashed"
        )
        plt.plot(
            ideal_df["x"],
            ideal_df[ideal_col],
            label=f"Ideal {ideal_col}"
        )

    plt.title("Training Data vs Best-Fit Ideal Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_test_scatter(test_results):
    """
    Scatter plot: Test data points
    """
    plt.figure(figsize=(8, 5))

    plt.scatter(
        test_results["x"],
        test_results["y"]
    )

    plt.title("Test Data Scatter Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


# Main Execution

def main():
    try:
        training = TrainingDataset("train.csv").get_dataframe()
        ideal = IdealDataset("ideal.csv").get_dataframe()
        test = TestDataset("test.csv").get_dataframe()

        selector = IdealFunctionSelector(training, ideal)
        best_functions = selector.select_best_functions()

        mapper = TestDataMapper(test, ideal, best_functions, training)
        test_results = mapper.map_test_data()

        training.to_sql("training_data", engine,
                        if_exists="replace", index=False)
        ideal.to_sql("ideal_functions", engine,
                     if_exists="replace", index=False)
        test_results.to_sql("test_results", engine,
                            if_exists="replace", index=False)

# Matplotlib plot

        plot_train_vs_ideal(training, ideal, best_functions)
        plot_test_scatter(test_results)

# Bokeh plot

        visualize(training, ideal, best_functions, test_results)

        print("Best ideal functions selected:")
        print(best_functions)
        print("\nTest data mapping complete.")

    except FileNotFoundError:
        print("CSV file not found. Please check file paths.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
