# Copyright 2020 (c) Netguru S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from itertools import cycle
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from opoca.evaluation.utils import calculate_roc_auc, calculate_prauc


class Plotter:
    """Class for plotting metrics for evaluating models."""
    def plot_roc_auc(self, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str] = None,
                     classes_to_plot: List[str] = None, show: bool = True, save_path: str = None):
        """
        Plots Receiver Operating Characteristic Area Under Curve.
        Parameters
        ----------
        y_true: np.ndarray
            Array of true labels (last shape of array should equal to number of classes).
        y_prob: np.ndarray
            Array of predicted probabilities (last shape of array should equal to number of classes).
        class_names: List of strings, optional
            Names of classes.
        classes_to_plot: List of strings, optional
            If provided, only the given classes get plotted.
            Otherwise all classes are plotted.
        show: bool, optional, default = True
            If True, the plot gets displayed.
        save_path: str, optional
            If provided, the plot gets saved in given path.
            The path's filename should end in desired save format e.g. 'bar_plot.png'.
        """
        class_names = class_names if class_names else [f'class_{i}' for i in range(y_prob.shape[-1])]
        fpr, tpr, roc_auc_dict = calculate_roc_auc(y_true, y_prob, class_names)
        self._plot_roc_auc(fpr, tpr, roc_auc_dict, class_names, classes_to_plot, show, save_path)

    def _plot_roc_auc(self, fpr: Dict[str, np.ndarray], tpr: Dict[str, np.ndarray], roc_auc_dict: Dict[str, np.ndarray],
                      class_names: List[str], classes_to_plot: List[str] = None, show: bool = True,
                      save_path: str = None):
        """
        Plots Receiver Operating Characteristic Area Under Curve.
        Parameters
        ----------
        fpr: dict
            Dictionary in which keys are names of classes and values are False Positives Ratio arrays.
        tpr: dict
            Dictionary in which keys are names of classes and values are True Positives Ratio arrays.
        roc_auc_dict: dict
            Dictionary in which keys are names of classes and values are calculated ROC area under curve.
        class_names: List of strings
            Names of classes.
        classes_to_plot: List of strings, optional
            If provided, only the given classes get plotted.
            Otherwise all classes are plotted.
        show: bool, optional, default = True
            If True, the plot gets displayed.
        save_path: str, optional
            If provided, the plot gets saved in given path.
            The path's filename should end in desired save format e.g. 'bar_plot.png'.
        """
        lw = 2
        colors = cycle(['turquoise', 'darkorange', 'cornflowerblue', 'teal', 'rosybrown',
                        'maroon', 'lawngreen', 'lightskyblue', 'palevioletred', 'indigo'])

        # Plot all ROC curves
        plt.figure(figsize=(24, 16))
        ax = plt.subplot(111)

        plt.plot(fpr["macro avg"], tpr["macro avg"],
                 label=f'macro-average ROC curve (area = {roc_auc_dict["macro avg"]:0.2f})',
                 color='navy', linestyle=':', linewidth=4)

        classes_to_plot = classes_to_plot if classes_to_plot else class_names

        for i, color in zip(classes_to_plot, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f'ROC curve of class {i} (area = {roc_auc_dict[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', {'fontsize': 15})
        plt.ylabel('True Positive Rate', {'fontsize': 15})
        plt.title('Receiver Operating Characteristic curve', {'fontsize': 20})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=dict(size=14))
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_prauc(self, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str] = None,
                   classes_to_plot: List[str] = None, show: bool = True, save_path: str = None):
        """
        Plots Precision-Recall Area Under Curve.
        Parameters
        ----------
        y_true: np.ndarray
            Array of true labels (last shape of array should equal to number of classes).
        y_prob: np.ndarray
            Array of predicted probabilities (last shape of array should equal to number of classes).
        class_names: List of strings, optional
            Names of classes.
        classes_to_plot: List of strings, optional
            If provided, only the given classes get plotted.
            Otherwise all classes are plotted.
        show: bool, optional, default = True
            If True, the plot gets displayed.
        save_path: str, optional
            If provided, the plot gets saved in given path.
            The path's filename should end in desired save format e.g. 'bar_plot.png'.
        """
        precision, recall, average_precision = calculate_prauc(y_true, y_prob)

        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'rosybrown',
                        'maroon', 'lawngreen', 'lightskyblue', 'palevioletred', 'indigo'])

        plt.figure(figsize=(24, 16))
        ax = plt.subplot(111)

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            ax.annotate(f'f1={f_score:0.1f}', xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = ax.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append(f'Micro-average Precision-recall (area = {average_precision["micro"]:0.2f})')

        # get positions of classes_to_plot in class_names
        if classes_to_plot:
            if class_names:
                classes_to_plot_index = [i for i, name in enumerate(class_names) if name in classes_to_plot]
            else:
                classes_to_plot_index = [i for i in list(range(y_prob.shape[-1])) if i in classes_to_plot]
        else:
            classes_to_plot_index = list(range(y_prob.shape[-1]))

        for i, color in zip(classes_to_plot_index, colors):
            l, = ax.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            if class_names:
                labels.append(
                    f'Precision-recall for class {class_names[i]} (area = {average_precision[i]:0.2f})')
            else:
                labels.append(f'Precision-recall for class {i} (area = {average_precision[i]:0.2f})')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', {'fontsize': 15})
        plt.ylabel('Precision', {'fontsize': 15})
        plt.title('Precision-Recall curve', {'fontsize': 20})
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop=dict(size=14))

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
