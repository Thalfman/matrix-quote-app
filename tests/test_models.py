import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from core.models import build_preprocessor, predict_with_interval


class TestPredictIntervals(unittest.TestCase):
    def test_quantile_interval_bounds(self):
        rng = np.random.RandomState(7)
        df = pd.DataFrame(
            {
                "num_a": rng.normal(size=40),
                "num_b": rng.uniform(-1, 1, size=40),
                "category": rng.choice(["alpha", "beta"], size=40),
            }
        )
        y = 2.0 * df["num_a"] - 0.5 * df["num_b"] + rng.normal(0, 0.1, size=40)

        num_features = ["num_a", "num_b"]
        cat_features = ["category"]

        point_pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(num_features, cat_features)),
                ("model", GradientBoostingRegressor(n_estimators=20, random_state=7)),
            ]
        )
        lower_pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(num_features, cat_features)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=20,
                        random_state=7,
                        loss="quantile",
                        alpha=0.1,
                    ),
                ),
            ]
        )
        upper_pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(num_features, cat_features)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=20,
                        random_state=7,
                        loss="quantile",
                        alpha=0.9,
                    ),
                ),
            ]
        )

        point_pipe.fit(df, y)
        lower_pipe.fit(df, y)
        upper_pipe.fit(df, y)

        bundle = {
            "point": point_pipe,
            "lower": lower_pipe,
            "upper": upper_pipe,
            "interval_level": 0.8,
            "interval_method": "quantile",
        }

        p50, p10, p90, std = predict_with_interval(bundle, df.head(5))

        self.assertEqual(len(p50), 5)
        self.assertEqual(len(p10), 5)
        self.assertEqual(len(p90), 5)
        self.assertEqual(len(std), 5)

        for lower, mid, upper in zip(p10, p50, p90):
            self.assertLessEqual(lower, mid)
            self.assertLessEqual(mid, upper)


if __name__ == "__main__":
    unittest.main()
