# Income Prediction Based on Census Data

## Project Overview

This project is focused on predicting whether a person's annual income exceeds $50,000 based on census information. Utilizing a dataset extracted from the 1994 Census database, this project employs machine learning models to achieve accurate predictions. The dataset comprises 14 attributes, including both categorical and numerical data, posing a challenge in terms of data preprocessing and model selection.

## Motivation

In the era of big data, machine learning offers invaluable tools for processing and analyzing large volumes of information efficiently. This project leverages machine learning to automate and expedite the process of income prediction, highlighting the power of machine learning in generating precise, scalable, and efficient solutions.

## Technical Details

### Technologies Used

- Python
- Decision Tree Model with ID3 Algorithm and Information Gain
- Random Forest Algorithm

### Key Learnings

- Handling missing values and preprocessing large datasets
- Balancing between model complexity and overfitting
- Effective data visualization and interpretation of model results

### Challenges

- Processing a large dataset with over 20,000 entries
- Dealing with missing values and diverse data types across different columns
- Optimizing model parameters to improve accuracy and reduce overfitting

## Getting Started

To run this project, ensure you have Python installed on your machine. Follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python random_forest.py` (or any other model script you wish to test).

## Data Preprocessing

Significant effort was put into preprocessing the data to enhance model performance, including:

- Dropping irrelevant columns to reduce model complexity.
- Simplifying categorical data for more straightforward analysis.
- Adjusting numerical data using thresholds or ranges to maintain accuracy while reducing complexity.

## Models and Analysis

### Decision Tree

The decision tree model provided a solid foundation for understanding the dataset and generating initial predictions. Through careful pruning and parameter tuning, the model achieved reasonable accuracy while avoiding overfitting.

### Random Forest

Expanding on the decision tree model, the random forest algorithm utilized multiple decision trees to improve prediction accuracy. Although the performance was similar to that of a single decision tree, the model demonstrated good resistance to noise and overfitting.

## Future Directions

Further improvements could involve exploring linear and nonlinear models, such as SVM or neural networks, to potentially enhance prediction accuracy on this complex dataset.

## Contribution

Your contributions are welcome! Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to Dr. Shandian Zhe for guidance throughout this project, and to Barry Becker for providing the dataset.
