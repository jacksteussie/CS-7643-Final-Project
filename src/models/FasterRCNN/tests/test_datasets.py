import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.FasterRCNN.datasets import DotaDataset

class TestDotaDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.val_dataset = DotaDataset("val")
    
    def test_size(self):
        assert len(self.val_dataset) == 9859

    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()