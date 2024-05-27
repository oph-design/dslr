<a name="readme-top"></a>




<h1 align="center">Data Science & Logictic Regression</h1>
<p align="center">
	<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/oph-design/dslr?color=lightblue" />
	<img alt="Code language count" src="https://img.shields.io/github/languages/count/oph-design/dslr?color=yellow" />
	<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/oph-design/dslr?color=blue" />
	<img alt="GitHub last commit" src="https://img.shields.io/github/created-at/oph-design/dslr?color=green" />
</p>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p align="center">
<img width="590" alt="Screen Shot 2024-05-27 at 3 32 46 AM" src="https://github.com/oph-design/dslr/assets/115570424/2e25c6d4-c480-4430-9ae9-44a682c55a00">
</p>


DSLR is a collection of python programs visualising and analysing the Hogwarts Dataset.</br>
The Dataset is consisting of multiple Hogwarts Students as Data points containing their house and grades.</br>
The Goal was to analyze the data and create a logistic regression model for telling the house of any student.</br>

The Programs are ...</br>
`describe.py`: replication of the pandas describe function</br>
`histogram.py`: plotting the histogram for every subject</br>
`scatter_plot.py`: plotting the comparison between each subject</br>
`pair_plot.py`: plotting a pair plot for the entire Dataset</br>
`logreg_train.py`: training the regression model</br>
`logreg_predict.py`: make a prediction using the created model</br>

For the training the classic gradient descent method is used. The Model focuses just on a few chosen features based on the previous analysis, namely 
`Ancient Runes`, `Defense Against the Dark Arts` and `Herbology`. The features were chosen because there distribution is split in 2 with each having 2 houses with good
and 2 house with bad grades, which creates a matrix able to tell a students house with almost 100% accuracy. The training is design to find this pattern and has a 99% accuracy
on the test Dataset.



<!-- GETTING STARTED -->
## Getting Started

The following contains a description of how to use the program.

### Prerequisites

To run the programs you have to have python3 and pip3 installed. See an installation guide <a href="https://kinsta.com/knowledgebase/install-python/">here</a>
After, you have to clone the repository and install the libraries used in this project.
  ```sh
   git clone https://github.com/oph-design/ft_linear_regression
   pip3 install -r requirements.txt
  ```


<!-- USAGE EXAMPLES -->
## Usage

You can start each program from the below list by typing `python3` and the path to the program.
It is recommended to start each program from the root of the repository to avoid pathing issue.
To make is more comfortable the program `controller.py` was added to provide a cli for starting the program.
`controller.py` takes as first input the program you want to use and ask for further input, using a default value if ypu keep the prompt blank.
   ```sh
   pyhton3 controller.py
   ```
Following is a list of the programs and their arguments:
| Program    | ARG1 | ARG2 | ARG3 |
| -------- | ------- | ------- | ------- |
| `describe.py`  | Path to Dataset |||
| `histogram.py` | Path to Dataset | Subject to show (shows all if empty) ||
| `scatter_plot.py`   | Path to Dataset | Subject to show  | Subject to compare too |
| `pair_plot.py`   | Path to Dataset |||
| `logreg_train.py`   | Path to Dataset  | Optimisation algorithm (GD, stochastic GD, mini-batch GD)||
| `logreg_predict.py`   | Path to Dataset  |||




<!-- EXMAPLES -->
## Examples

### Histogram
   ```sh
   python3 visuals/histogram.py datasets/dataset_train.csv
   ```
<img width="2381" alt="Screen Shot 2024-05-27 at 4 17 01 AM" src="https://github.com/oph-design/dslr/assets/115570424/b589ed0b-2028-4456-b109-b90878347635">


<p></p>

### Scatterplot 1 Argument
   ```sh
   python3 visuals/scatter_plot.py datasets/dataset_train.csv Herbology
   ```
<img width="1486" alt="Screen Shot 2024-05-27 at 4 17 46 AM" src="https://github.com/oph-design/dslr/assets/115570424/d47c455d-891b-49e4-8e48-ff3d392728d2">


<p></p>

### Scatterplot 2 Arguments
   ```sh
   python3 visuals/scatter_plot.py datasets/dataset_train.csv Flying "Ancient Runes"
   ```
<img width="592" alt="Screen Shot 2024-05-27 at 4 18 28 AM" src="https://github.com/oph-design/dslr/assets/115570424/e15758e2-f453-4d03-9210-7d36267dfe67">

<p></p>

### Pairplot
   ```sh
   python3 visuals/pair_plot.py datasets/dataset_train.csv
   ```
<img width="1299" alt="Screen Shot 2024-05-27 at 4 22 42 AM" src="https://github.com/oph-design/dslr/assets/115570424/bd69879c-164e-4c02-a9c8-c0b3b738da61">




<!-- CONTACT -->
## Contact

Ole-Paul Heinzelmann</br>
ole.paul.heinzelmann@protonmail.com </br>
<p></p>
<a href="https://www.linkedin.com/in/ole-paul-heinzelmann-a08304258/">
<img alt="linkedin shield" src="https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555" />
</a></br> 

<p align="right">(<a href="#readme-top">back to top</a>)</p>
