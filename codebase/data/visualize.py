import os.path

import matplotlib.pyplot as plt
from PIL.Image import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from .hashset import HashSet
from .sample import Sample
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

from ..knn.inference import HashInferenceResult


class SampleVisualizer:
    """Class to visualize samples."""

    def __init__(self,
                 default_output_dir: str | Path = None,
                 max_cols: int = 3,
                 max_rows: int = 3,
                 plot_size: int = 15):
        """

        :param default_output_dir: Output dir for any plot. Defaults to "/home/frank/data/apps/default_plot_output "
        :param max_cols: Columns to display.
        :param max_rows: Rows to display.
        :param plot_size: Plot size.
        """
        self.default_output_dir = default_output_dir if default_output_dir else "/home/frank/data/apps" \
                                                                                "/default_plot_output"
        self.max_cols = max_cols
        self.max_rows = max_rows
        self.plot_size = (plot_size, 2 * plot_size)
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    def create_grid(self, samplelist: list[Sample]):
        """Pass a list of samples to visualize. ImageGrid does not work properly, needs similar AR on images
        :param samplelist: List of sample.Samples object
        """

        num_rows = len(samplelist) // self.max_cols if len(samplelist) % self.max_cols == 0 else (
                len(samplelist) // self.max_cols + 1)
        gs = gridspec.GridSpec(num_rows, self.max_cols)
        fig = plt.figure(figsize=self.plot_size)
        # fig.subplots_adjust(wspace=0.1, hspace=0)
        sample_representations = [sample.vis_representation() for sample in samplelist]
        for i, sample in enumerate(sample_representations):
            ax = fig.add_subplot(gs[i // self.max_cols, i % self.max_cols])
            ax.imshow(sample["image"], cmap="gray")
            ax.set_title(sample["id"], fontsize=8)
            ax.text(0.05, 0.95, sample["textboxstr"], transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=self.props)
            ax.axis("off")

        return fig

    def visualize_grid(self, samplelist: list[Sample]):
        plt.clf()
        self.create_grid(samplelist)
        plt.show()

    def save_grid(self, samplelist: list[Sample], filename=None):
        """Stores the grid plot to a file.

        :param samplelist: List of sample.Sample class
        :param filename: filename (optional)
        """
        fig = self.create_grid(samplelist)
        if filename is None:
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
            filename = f"grid_plot_{date_time}.jpg"
        fig.savefig(os.path.join(self.default_output_dir, filename))

    # here come the methods for knn index result vis
    def create_knn_result_grid(self, result, hashset: HashSet):
        """Pass a HashInferenceResult to visualize. ImageGrid does not work properly, needs similar AR on images
        :param hashset:
        :param result: List of sample.Samples object
        """
        result = hashset.process_result(result)
        num_rows = len(result.result_samples) // self.max_cols if len(result.result_samples) % self.max_cols == 0 else (
                len(result.result_samples) // self.max_cols + 1)
        gs = gridspec.GridSpec(num_rows + 1, self.max_cols)
        fig = plt.figure(figsize=self.plot_size)
        fig.tight_layout()
        sample_result_representations = [sample.vis_representation() for sample in result.result_samples]
        query_representation = result.query.vis_representation()

        # add the text box above
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title(f"Your query: {query_representation['id']} \n{query_representation['textboxstr']}", fontsize=8)
        """ax.text(0.05, 0.95, "text",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=self.props)"""
        ax.imshow(query_representation["image"], cmap="gray")
        ax.axis("off")

        for i, sample in enumerate(sample_result_representations):
            ax = fig.add_subplot(
                gs[((i + self.max_cols) // self.max_cols), (i + self.max_cols) % self.max_cols])
            """ax_text.text(0.05, 0.95, f"sample-id: {sample['id']} \n{sample['textboxstr']}", transform=ax.transAxes,
                         fontsize=8,
                         verticalalignment='top', bbox=self.props)"""
            ax.axis("off")
            ax.imshow(sample["image"], cmap="gray")
            ax.set_title(f"sample-id: {sample['id']} \n{sample['textboxstr']}", fontsize=8)

        return fig

    def save_knn_result_grid(self, result=None, hashset=None, filename=None):
        """Stores the grid plot to a file.

        :param samplelist: List of sample.Sample class
        :param filename: filename (optional)
        """
        fig = self.create_knn_result_grid(result, hashset)
        if filename is None:
            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
            filename = f"grid_plot_{date_time}.jpg"
        fig.savefig(os.path.join(self.default_output_dir, filename))
