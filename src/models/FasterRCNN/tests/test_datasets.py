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

    def test_labelcorrectness(self):
        _, target = self.val_dataset[10]
        assert target["image_id"] == 10
        assert len(target["labels"]) == 60
        assert [target["labels"][x] == 9 for x in range(41)]
        assert [target["labels"][x] == 10 for x in range(42, 58)]

    def tearDown(self) -> None:
        return super().tearDown()

if __name__ == '__main__':
    unittest.main()