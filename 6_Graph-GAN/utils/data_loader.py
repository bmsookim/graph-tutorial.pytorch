import pickle
import numpy as np
from rdkit import Chem

class SparseMolecularDataset():
    def load(self, filename, subset=1):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset),
                replace = False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                replace = False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset),
                replace = False)

        self.train_cnt = len(self.train_idx)
        self.validation_cnt = len(self.validation_idx)
        self.test_cnt = len(self.test_idx)

        self.__len = self.train_cnt + self.validation_cnt + self.test_cnt

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        if filename.endswith('.sdf'):
            self.data = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
        elif filename.endswith('.smi'):
            self.data = [Chem.MolFromSmiles(line) for line in open(filename, 'r').readlines()]

        self.data = list(map(Chem.AddHs, self.data)) if add_h else self.data
        self.data = list(filter(filters, self.data))
        self.data = self.data[:size]

        self.log('Extracted {} out of {} molecules {}adding Hydrogen!'.format(
            len(self.data), len(Chem.SDMolSupplier(filename)), '' if add_h else 'not '))

    @staticmethod
    def log(msg='', date=True):
        print(msg)

    def __len__(self):
        return self.__len

if __name__ == "__main__":
    data = SparseMolecularDataset()
    data.generate('data/gdb9.sdf', filters=lambda x: x.GetNumAtoms() <= 9)
