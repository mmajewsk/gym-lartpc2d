from __future__ import print_function
import ROOT
from ROOT import TChain
from larcv import larcv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import tqdm
from argparse import ArgumentParser


class LarcvDataGenerator:
    def __init__(self, filepath):
        self.t_chain_name1 = 'image2d_data_tree'
        self.t_chain_name2 = 'image2d_segment_tree'
        self.filepath = filepath
        self.chain_image2d = ROOT.TChain(self.t_chain_name1)
        self.chain_image2d.AddFile(filepath)
        self.chain_label2d = ROOT.TChain(self.t_chain_name2)
        self.chain_label2d.AddFile(filepath)

    def show_event(self, entry=-1, projection=0):
        if entry < 0:
            entry = np.random.randint(0, self.chain_label2d.GetEntries())
        image2d, label2d = self.get_event(entry, projection=projection)
        xlim, ylim = get_view_range(image2d)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 12), facecolor='w')
        ax0.imshow(image2d, interpolation='none', cmap='jet', origin='lower')
        ax1.imshow(label2d, interpolation='none', cmap='jet', origin='lower', vmin=0., vmax=3.1)
        ax0.set_title('Data', fontsize=20, fontname='Georgia', fontweight='bold')
        ax0.set_xlim(xlim)
        ax0.set_ylim(ylim)
        ax1.set_title('Label', fontsize=20, fontname='Georgia', fontweight='bold')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        plt.show()

    def get_event(self, entry, projection=0):
        entrya = self.chain_label2d.GetEntry(entry)
        entryb = self.chain_image2d.GetEntry(entry)
        # @TODO this can be expanded by multiple projections
        image2d = larcv.as_ndarray(self.chain_image2d.image2d_data_branch.as_vector()[projection])
        label2d = larcv.as_ndarray(self.chain_label2d.image2d_segment_branch.as_vector()[projection])
        return (np.array(image2d), np.array(label2d))


if __name__ == "__main__":
    parser = ArgumentParser(description="This script will reprocess lartpc data to format readable by lartpc_game")
    parser.add_argument("-s", "--source_path", type=str, default='./data_lartpc2d/test_10k.root',
                        help="Path to data downloaded from lartpc http://deeplearnphysics.org/DataChallenge/ --> MULTIPLE PARTICLE SAMPLE --> train_15k.root or test_10k.root. Leaving this argument empty will default in home/data.")
    parser.add_argument("-d", "--destination", type=str, default='./dump/',
                        help="Path to destination folder to dump data to.")
    parser.add_argument("--entries", type=int, default=None,
                        help="Number of entries to process. Default, or none == all of the entries. The actual number of images will be (source and target) 2 * (number of projections) * 3 * entries. ")
    args = parser.parse_args()
    ldg = LarcvDataGenerator(args.source_path)
    if args.entries is None:
        number_of_entries = ldg.chain_image2d.GetEntries()
    else:
        number_of_entries = args.entries
    print("******************\nGenerating data, now. Number of entries = {}".format(number_of_entries))
    images, labels = [], []
    for i in tqdm.tqdm(range(number_of_entries)):
        for j in range(3):
            x, y = ldg.get_event(i, j)
            # x_sparse = scipy.sparse.csr_matrix(x)
            # y_sparse = scipy.sparse.csr_matrix(y)
            images.append(x)
            labels.append(y)

    for i, (im, l) in enumerate(zip(images, labels)):
        np.save(args.destination + 'image{}.npy'.format(i), im)
        np.save(args.destination + 'label{}.npy'.format(i), l)