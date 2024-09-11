# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:46:21 2023

@author: Bart Steemans. Govers Lab.
"""
import pickle
import pandas as pd
import numpy as np


class Curation:

    def __init__(self, svm_df):

        self.svm_df = svm_df
        self.features = None
        self.curated_df = None
        self.svm_model = None

    def compiled_curation(self, path_to_model, cols):
        """
        Perform compiled cell curation using a trained SVM model.

        This method curates the dataset by making predictions using a trained Support Vector Machine (SVM) model.
        Cells that do not meet the criteria specified by the model are flagged for potential removal.

        Parameters
        ----------
        path_to_model : str
            The path to the trained SVM model file.

        cols : int
            The number of columns to exclude in the dataset during curation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the curated dataset.

        """
        self.load_model(path_to_model)
        self.prepare_dataframe(cols)
        self.curated_df = self.make_predictions()
        return self.curated_df

    def load_model(self, path_to_model):
        """
        Load a trained SVM model from a file.

        This method loads a trained Support Vector Machine (SVM) model from a file located at the given path.

        Parameters
        ----------
        path_to_model : str
            The path to the trained SVM model file.

        """
        with open(path_to_model, "rb") as f:
            self.svm_model = pickle.load(f)

    def prepare_dataframe(self, cols):
        """
        Prepare the feature dataset for SVM curation.

        This method prepares the feature dataset for cell curation using a Support Vector Machine (SVM) model.
        It extracts the relevant feature columns while excluding the specified number of initial columns.

        Parameters
        ----------
        cols : int
            The number of columns to exclude from the feature dataset. Typically, this includes non-feature columns.

        """
        # Cols is generally 4 due to the non-feature columns: image_name, frame, cell_id and contour.
        exclude_columns = ["image_name", "frame", "contour", "cell_id", "label"]

        # Get the indices of the columns to include
        cols_to_include = [
            col for col in self.svm_df.columns if col not in exclude_columns
        ]
        cols_to_include.sort()
        # Select the columns to include in self.features
        self.features = self.svm_df[cols_to_include]
        # Convert numeric columns to float
        self.features[self.features.select_dtypes(include=[np.number]).columns] = (
            self.features.select_dtypes(include=[np.number]).astype(float)
        )

        # Check for infinite values and NaNs
        has_infinite = np.any(
            np.isinf(self.features.select_dtypes(include=[np.number]))
        )
        has_nan = np.any(np.isnan(self.features.select_dtypes(include=[np.number])))

        if has_infinite or has_nan:
            # Create a boolean mask for rows without infinite or NaN values
            mask = ~(
                np.isinf(self.features.select_dtypes(include=[np.number])).any(axis=1)
                | np.isnan(self.features.select_dtypes(include=[np.number])).any(axis=1)
            )

            svm_df_nan = self.svm_df[~mask]
            self.svm_df_nan = svm_df_nan
            self.svm_df_nan["label"] = 0

            # Filter the DataFrame to keep only rows without infinite or NaN values
            svm_training_filtered = self.features[mask]

            # Print the number of removed rows
            num_removed_rows = len(self.features) - len(svm_training_filtered)
            print(f"\nNumber of removed rows: {num_removed_rows}")

            # Update self.features with the filtered DataFrame
            self.features = svm_training_filtered
            self.svm_df = self.svm_df[mask]

        else:
            self.svm_df_nan = pd.DataFrame()
            print("\nNo rows removed")

    def get_label_proportions(self):
        """
        Calculate the proportions of labels in the curated dataset.

        This method calculates and returns the proportions of label 1 and label 0 in the curated dataset.
        It counts the occurrences of each label and divides them by the total number of samples in the dataset to compute the proportions.

        Returns
        -------
        float
            The proportion of label 1 in the curated dataset.
        float
            The proportion of label 0 in the curated dataset.

        """
        label_counts = self.curated_df["label"].value_counts()
        total_samples = len(self.curated_df)

        proportion_label_1 = label_counts.get(1, 0) / total_samples
        proportion_label_0 = label_counts.get(0, 0) / total_samples

        return proportion_label_1, proportion_label_0

    def make_predictions(self):
        """
        Make predictions using the SVM model.

        This method uses the trained Support Vector Machine (SVM) model to make predictions on the features in the curated dataset.
        The predicted labels are added to a new 'label' column in the DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted labels in the 'label' column.
        """
        # Make predictions using the SVM model
        predictions = self.svm_model.predict(self.features)

        # Assuming 'label' column is missing in the DataFrame, add the predicted labels
        self.curated_df = self.svm_df.copy()
        self.curated_df["label"] = predictions
        self.curated_df = pd.concat(
            [self.curated_df, self.svm_df_nan], ignore_index=True
        )

        return self.curated_df

    def get_control(self, num_pos=5, num_neg=5):
        """
        Get control cell IDs for positive and negative classes.

        This method retrieves cell IDs for positive and negative classes based on the curated dataset.
        It randomly selects a specified number of positive and negative cell IDs and returns them as lists, along with the corresponding 'frame' values.

        Parameters
        ----------
        num_pos : int, optional
            The number of positive class cell IDs to retrieve. The default is 5.
        num_neg : int, optional
            The number of negative class cell IDs to retrieve. The default is 5.

        Returns
        -------
        Tuple
            Two lists containing tuples of ('frame', 'cell_id') for positive and negative class control cell IDs.
            If there are not enough samples for one class, the corresponding list will be empty.
        """

        positive_indices = self.curated_df[self.curated_df["label"] == 1].index.tolist()
        negative_indices = self.curated_df[self.curated_df["label"] == 0].index.tolist()

        # Check if there are not enough positive or negative samples
        if len(positive_indices) < num_pos:
            print("No positive labels")
            return None, None  # Return None for positive and available negative indices
        elif len(negative_indices) < num_neg:
            print("No negative labels")
            return None, None  # Return available positive indices and None for negative

        positive_indices_random = np.random.choice(
            positive_indices, size=num_pos, replace=False
        )
        negative_indices_random = np.random.choice(
            negative_indices, size=num_neg, replace=False
        )

        # Retrieve cell_id and frame for selected indices
        positive_cell_frames = [
            (self.curated_df.loc[idx, "cell_id"], self.curated_df.loc[idx, "frame"])
            for idx in positive_indices_random
        ]
        negative_cell_frames = [
            (self.curated_df.loc[idx, "cell_id"], self.curated_df.loc[idx, "frame"])
            for idx in negative_indices_random
        ]

        return positive_cell_frames, negative_cell_frames
