# Data Science

Current work and projects in development:

- Data cleaning and preparation
- Exploratory data analysis
- Predictive modeling
- Data visualization (histograms, correlation plots, heatmaps)
- Utilization of libraries such as Pandas, NumPy, and Matplotlib
- Neural network implementation for image analysis and flood detection

The main objective of this project is to apply these skills to solve real-world problems through data analysis and machine learning techniques.

## Project Structure

- `utils/`: Contains utilities for each project.
- `scripts/`: Python scripts used for data processing.
- `output/`: Generated results and visualizations.
- `Red Neuronal/`: Neural network projects, including FloodNet for flood detection.

## Development Environment

To work on this project, we use a virtual environment (venv). Here are the steps to set it up:

1. Create a virtual environment:
    ```
    python -m venv venv
2. Activate the virtual environment:
    ```
    .\venv\Scripts\activate    
3. Install the necessary dependencies:
   ```
    pip install numpy matplotlib pandas scikit-learn tensorflow keras
## Additional Details

### Data Normalization

Data normalization is a crucial step in our data analysis process. In this project, we have employed techniques such as standardization and min-max normalization to ensure our models function correctly.

### Data Training and Testing

In the **Testing** phase, we trained a normalized data model using INEGI statistics, both for visual representation and backend processing. Our neural network projects, particularly FloodNet, involve training on satellite imagery for flood detection and segmentation.

### Tools and Technologies Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For predictive modeling and machine learning algorithms.
- **TensorFlow & Keras**: For building and training neural networks, especially in our flood detection project.


### Neural Network Projects

Our repository includes advanced neural network projects, with a focus on:

- **FloodNet**: A convolutional neural network designed for flood detection and segmentation in satellite imagery.
- **Image Analysis**: Implementing various architectures for image classification and segmentation tasks.
