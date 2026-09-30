from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

"""Trains a chemprop model on a dataset."""

from chemprop.train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
