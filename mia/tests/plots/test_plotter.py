import os
import pandas as pd
from mia.plots.plotter import plot_roc_curve


def test_plot_roc():
    d = {
            'labels': [1, 0, 1, 1, 0, 0, 1, 0],
            'confidences': [
                "{0:0;1:0.2}",
                "{0:0;1:0.3}",
                "{0:0;1:0.8}",
                "{0:0;1:0.2}",
                "{0:0;1:0.1}",
                "{0:0;1:0.1}",
                "{0:0;1:0.9}",
                "{0:0;1:0.10}"]
            }
    df = pd.DataFrame(data=d)
    plot_roc_curve(
        df['labels'],
        df['confidences'],
        os.getcwd(),
        "roccer")
    exists = os.path.exists(os.path.join(os.getcwd(), "roccer_roc.png"))
    assert exists is True
    os.remove(os.path.join(os.getcwd(), "roccer_roc.png"))
