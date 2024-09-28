import unittest
import pandas as pd
from app import load_data, get_min, get_max, selected_features

class TestBreastCancerApp(unittest.TestCase):

    def setUp(self):
        self.df, self.target_names = load_data()

    def test_load_data(self):
        df, target_names = load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), len(selected_features) + 1)  # +1 for 'cancer' column
        self.assertEqual(len(target_names), 2)  # Assuming binary classification

    def test_get_min(self):
        min_values = get_min(self.df)
        self.assertIsInstance(min_values, dict)
        for feature in selected_features:
            self.assertIn(feature, min_values)

    def test_get_max(self):
        max_values = get_max(self.df)
        self.assertIsInstance(max_values, dict)
        for feature in selected_features:
            self.assertIn(feature, max_values)

if __name__ == '__main__':
    unittest.main()