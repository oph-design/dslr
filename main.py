import subprocess

BLUE = "\033[94m"
DEF = "\033[0m"

programs = ["histogram", "scatter_plot", "pair_plot", "describe", "train", "predict"]


def main():
    execution = ["python3"]
    program = input(f"{BLUE}Enter the program you want to run:{DEF} ")
    while program not in programs:
        program = input(
            f"{BLUE}Options: describe | histogram | "
            + f"scatterplot | pairplot | train | predict:{DEF} "
        )
    if program == "train" or program == "predict":
        program = "model/logreg_" + program
    else:
        program = "visuals/" + program
    execution.append(program + ".py")
    dataset = input(f"{BLUE}Enter your desired dataset with path:{DEF} ")
    if dataset == "":
        dataset = "datasets/dataset_train.csv"
    execution.append(dataset)
    if program == "visuals/histogram" or program == "visuals/scatter_plot":
        feature = input(f"{BLUE}Enter a feature you want to look at:{DEF} ")
        if feature != "":
            execution.append(feature)
    if program == "visuals/scatter_plot":
        feature = input(f"{BLUE}Enter a feature to compare to:{DEF} ")
        if feature != "":
            execution.append(feature)
    print(f"{BLUE}Executing {program} please stay patient ...{DEF}")
    subprocess.run(execution)


if __name__ == "__main__":
    main()
