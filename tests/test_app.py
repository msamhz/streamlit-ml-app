"""
Unit tests for the Breast Cancer Prediction App.
"""

import unittest
import pandas as pd
from app import load_data, get_min, get_max, selected_features

class TestBreastCancerApp(unittest.TestCase):
    """
    Test cases for the Breast Cancer Prediction App.
    """

    def setUp(self):
        """
        Set up the test case environment.
        """
        self.df, self.target_names = load_data()

    def test_load_data(self):
        """
        Test the load_data function.
        """
        df, target_names = load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), len(selected_features) + 1)  # +1 for 'cancer' column
        self.assertEqual(len(target_names), 2)  # Assuming binary classification

    def test_get_min(self):
        """
        Test the get_min function.
        """
        min_values = get_min(self.df)
        self.assertIsInstance(min_values, dict)
        for feature in selected_features:
            self.assertIn(feature, min_values)

    def test_get_max(self):
        """
        Test the get_max function.
        """
        max_values = get_max(self.df)
        self.assertIsInstance(max_values, dict)
        for feature in selected_features:
            self.assertIn(feature, max_values)

if __name__ == '__main__':
    unittest.main()