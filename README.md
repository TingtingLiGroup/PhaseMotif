# Introduction to PhaseMotif
PhaseMotif is a sequence-based PS IDR classifier built with interpretable deep attention framework. Phase separation is crucial for the formation of biomolecular condensates, which play key roles in regulating cellular activities such as gene expression and signal transduction. This repository provides the source code for PhaseMotif, enabling researchers to explore and analyze key regions within IDRs that drive phase separation.

To make it even easier to use, please visit the [PhaseMotif Website](http://predict.phasemotif.pro/)


# Install

```bash
# Step 1: Create a new conda virtual environment with Python 3.8
conda create --name myenv python=3.8

# Step 2: Activate the new environment
conda activate myenv

# Step 3: Install the specified versions of PyTorch and related libraries
# The --index-url flag ensures the packages are fetched from the official PyTorch repository
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Step 4: Clone the PhaseMotif repository from GitHub
git clone https://github.com/TingtingLiGroup/PhaseMotif.git

# Step 5: Navigate into the cloned repository directory
cd PhaseMotif

# Step 6: Install the package using setup.py
python setup.py install
```



# Function Documentation

## Quick Reference Table

| Function       | Description                                                  | Parameters                                                   | Returns                           |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------- |
| `analyse_main` | Analyzes IDR sequences and generates visualizations (optional). | `idr_list`, `idr_name=None`, `paint=False`                   | DataFrame with analysis results   |
| `predict_main` | Predicts the results for IDR sequences.                      | `idr_list`, `idr_name=None`                                  | DataFrame with prediction results |
| `generate`     | Generates new sequences based on a specified cluster.        | `cluster`, `epoch=20`, `overLap=3`, `nomalize_threshold=0.95` | DataFrame                         |

------



## `analyse_main`

### Description

The `analyse_main` function performs analysis on Intrinsically Disordered Regions (IDRs). It evaluates the density of significant points, frequency of selections, important points, and cluster labels for each IDR sequence.

### Parameters

+ **idr_list** (*list of str*): List of IDR sequences to be analyzed.
+ **idr_name** (*list of str, optional*): List of names for each IDR sequence. If not provided, names will be generated automatically.
+ **paint** (*bool, optional*): Flag indicating whether to generate visualizations for the results. Default is `False`.

### Returns

+ **analyse_result_df** (*[pd.DataFrame](https://pd.dataframe/)*): DataFrame containing the analysis results including IDR name, sequence, density, important positions, frequency of selections, cluster labels, and key regions.

### Example

```python
import PhaseMotif as pm

# Sample IDR sequences
idr_list = [
    "MSVAKTPKTAENAEKPHVNVGTIGPHEDTYYSEF",
    "GPTLSEDNLSYYKSQPGFQKMSADK"
]
idr_name = ["IDR1", "IDR2", ...]

# Analyze the IDR sequences without naming or painting
pm.analyse_main(idr_list)

# Analyze with name and visualization
pm.analyse_main(idr_list, idr_name, paint=True)

# If you need to further manipulate or visualize the results, you can store them in a DataFrame
# results_df = pm.analyse_main(idr_list)
```

### Notes

+ The function will check if the lengths of `idr_list` and `idr_name` match. If they don't, it raises a `ValueError`.
+ Each element in `idr_name` must be a non-empty string. If any element doesn't meet this criterion, a `ValueError` is raised.
+ If `paint` is set to `True`, visualizations for each IDR will be saved in the `PM_analyse/Pic_result` directory.
+ The results DataFrame is also saved as `PM_analyse/PM_analyse_result.csv`.

------



## `predict_main`

### Description

The `predict_main` function predicts the results for a list of Intrinsically Disordered Regions (IDRs). It evaluates the predict scores for each IDR sequence.

### Parameters

+ **idr_list** (*list of str*): List of IDR sequences to be analyzed.
+ **idr_name** (*list of str, optional*): List of names for each IDR sequence. If not provided, names will be generated automatically.

### Returns

+ **predict_result_list** (*[pd.DataFrame](https://pd.dataframe/)*): DataFrame containing the prediction results including IDR name, sequence, and predict score.

### Example

```python
import PhaseMotif as pm

idr_list = ["MSVAKTPKTAENAEKPHVNVGTIGPHEDTYYSEF", "GPTLSEDNLSYYKSQPGFQKMSADK"]
idr_name = ["IDR1", "IDR2"]

# Predict without naming the sequences
pm.predict_main(idr_list)

# Predict with named sequences
pm.predict_main(idr_list, idr_name)

#  you can also use results_df = pm.predict_main(...) directly for further manipulation.
results_df = pm.predict_main(idr_list, idr_name)
print(results_df)
```

### Notes

+ The function checks if the lengths of `idr_list` and `idr_name` match. If they don't, it raises a `ValueError`.
+ Each element in `idr_name` must be a non-empty string. If any element doesn't meet this criterion, a `ValueError` is raised.
+ The results DataFrame is saved as `PM_analyse/PM_predict_result.csv`.

------



## `generate`

### Description

The `generate` function uses a Variational Autoencoder (VAE) to generate new sequences based on a specified cluster. It normalizes, filters, and merges the generated sequences and saves them to a CSV file.

### Parameters

+ **cluster** (*str*): The cluster name to use for generation. Must be one of `['0', 'polar', 'pos_neg', 'P', 'G', 'pos', 'aliphatic', 'neg', 'Q']`.
+ **epoch** (*int, optional*): Number of generations to perform. Default is 20.
+ **overLap** (*int, optional*): Overlap parameter for merging sequences. Default is 3.
+ **nomalize_threshold** (*float, optional*): Normalization threshold to filter values. Default is 0.95.

### Returns

+ **[pd.DataFrame](https://pd.dataframe/)**: DataFrame containing the generated sequences.

### Example

```python
# For quick and easy use:
pm.generate('polar')

# For more detailed customization, you can use the following method:
result_df = generate(cluster='polar', epoch=30, overLap=5, nomalize_threshold=0.9)

# Print the resulting DataFrame
print(result_df)
```

### Notes

+ The function raises a `ValueError` if the provided `cluster` is not in the predefined list.
+ Generates sequences using a VAE model and filters them based on the normalization threshold.
+ Merges sequences based on the specified overlap and retains sequences that match the target cluster.
+ Saves the results to `PM_generate/generate_{cluster}.csv`. If the file exists, it appends the data; otherwise, it creates a new file.
+ You can further manipulate the resulting DataFrame as needed.
