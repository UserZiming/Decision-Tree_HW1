# Income Prediction Based on Census Data

## Project Overview

This project is focused on predicting whether a person's annual income exceeds $50,000 based on census information. Utilizing a dataset extracted from the 1994 Census database, this project employs machine learning models to achieve accurate predictions. The dataset comprises 14 attributes, including both categorical and numerical data, posing a challenge in terms of data preprocessing and model selection.

## Motivation

In the era of big data, machine learning offers invaluable tools for processing and analyzing large volumes of information efficiently. This project leverages machine learning to automate and expedite the process of income prediction, highlighting the power of machine learning in generating precise, scalable, and efficient solutions.

## Project Workflow

This section provides a step-by-step overview of the workflow for the income prediction project, highlighting the key phases from data preparation to model evaluation and future work considerations.

### 1. Data Collection and Preprocessing

- **Data Collection:** The project begins with collecting a dataset from the 1994 Census database, which includes various attributes related to individuals' demographics and income levels.
- **Data Preprocessing:** Essential preprocessing steps are undertaken to prepare the dataset for modeling. This includes handling missing values, encoded with "?" in the dataset, and selecting relevant features that contribute to the prediction of whether an individual's income exceeds $50K per year. The preprocessing phase aims to clean and structure the data effectively for the machine learning models.

### 2. Model Selection and Development

- **Decision Tree Model:** The initial approach involves developing a decision tree model to understand the data's underlying patterns. This model serves as a baseline for predicting income levels based on the dataset's attributes. Decision trees are chosen for their interpretability and ability to handle both numerical and categorical data.
- **Random Forest Model:** To improve upon the decision tree model, a Random Forest algorithm is employed. This ensemble method utilizes multiple decision trees to reduce overfitting and increase prediction accuracy. The random forest model is particularly valued for its robustness against noise and its capability to handle large datasets efficiently.

### 3. Model Training and Evaluation

- **Training:** Both models are trained on the dataset, which involves fitting the models to the training data to learn the relationships between the features and the target variable.
- **Evaluation:** The trained models are evaluated using a separate testing dataset to assess their accuracy in predicting income levels. The evaluation process helps in understanding the models' performance and identifying areas for improvement.

### 4. Data Adjustments and Model Refinement

- **Data Adjustments:** Based on initial results, further adjustments are made to the data, such as dropping less significant features and simplifying others to improve model performance.
- **Model Refinement:** The models are refined through hyperparameter tuning and additional training, aiming to enhance their predictive accuracy. The refinement process involves iterative testing and adjustments until the optimal model performance is achieved.

### 5. Experimental Results and Future Work

- **Results Analysis:** The project documents the experimental results, analyzing the effectiveness of the decision tree and random forest models in income prediction. This includes discussions on challenges faced, such as dealing with a large dataset and complex model structures.
- **Future Work:** Potential future directions are outlined, including exploring other modeling approaches like SVM or neural networks to improve prediction accuracy. The section on future work emphasizes the ongoing nature of machine learning projects and the potential for further advancements in income prediction models.

This workflow demonstrates a comprehensive approach to tackling the income prediction problem using machine learning techniques, from data preparation through model development, evaluation, and refinement.

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
