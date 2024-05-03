import subprocess

BLUE = "\033[94m"
DEF = "\033[0m"

commands = ["hist", "scatter", "pair", "describe", "train", "predict"]
programs = [
    "visuals/histogram.py",
    "visuals/scatter_plot.py",
    "visuals/pair_plot.py",
    "visuals/describe.py",
    "model/logreg_train.py",
    "model/logreg_predict.py",
]


def main() -> None:
    execution = ["python3"]
    command = input(f"{BLUE}Enter the program you want to run:{DEF} ")
    while command not in commands:
        command = input(f"{BLUE}Options: {commands}:{DEF} ")
    program = programs[commands.index(command)]
    execution.append(program)
    dataset = input(f"{BLUE}Enter your desired dataset with path:{DEF} ")
    if dataset == "":
        dataset = "datasets/dataset_train.csv"
    execution.append(dataset)
    if program == programs[0] or program == programs[1]:
        feature = input(f"{BLUE}Enter a feature you want to look at:{DEF} ")
        if feature != "":
            execution.append(feature)
    if program == programs[1]:
        feature = input(f"{BLUE}Enter a feature to compare to:{DEF} ")
        if feature != "":
            execution.append(feature)
    if program == programs[4]:
        feature = input(f"{BLUE}Enter a an optimization alogrithm:{DEF} ")
        if feature != "":
            execution.append(feature)
    print(f"{BLUE}Executing {program} please stay patient ...{DEF}")
    subprocess.run(execution)


if __name__ == "__main__":
    main()
