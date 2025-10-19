import torch
import unittest
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy
from lerobot.policies.smolvla4.configuration_smolvla4 import SmolVLA4Config


class TestPrepareBoxAndPoint(unittest.TestCase):
    def setUp(self):
        # Create a minimal config for testing
        self.config = SmolVLA4Config(
            max_num_embeddings_box=5,
        )
        # Create a minimal policy instance for testing
        self.policy = SmolVLA4Policy(config=self.config)

    def test_prepare_box_and_point_with_bboxes(self):
        """Test prepare_box_and_point when bboxes are present in the batch"""
        batch_size = 2
        num_boxes = 3
        max_num_boxes = self.config.max_num_embeddings_box
        
        # Create sample bounding boxes in format [class, x, y, w, h]
        # For each box: [class_id, x, y, width, height]
        bboxes = torch.tensor([
            # Batch 0
            [
                [0.0, 10.0, 20.0, 30.0, 40.0],  # class=0, x=10, y=20, w=30, h=40 -> center=(25, 40)
                [1.0, 50.0, 60.0, 20.0, 30.0],  # class=1, x=50, y=60, w=20, h=30 -> center=(60, 75)
                [2.0, 100.0, 120.0, 40.0, 50.0],  # class=2, x=100, y=120, w=40, h=50 -> center=(120, 145)
            ],
            # Batch 1
            [
                [0.0, 5.0, 10.0, 15.0, 20.0],   # class=0, x=5, y=10, w=15, h=20 -> center=(12.5, 20)
                [1.0, 30.0, 40.0, 10.0, 25.0],  # class=1, x=30, y=40, w=10, h=25 -> center=(35, 52.5)
                [0.0, 0.0, 0.0, 0.0, 0.0],      # Empty box (all zeros)
            ]
        ], dtype=torch.float32)
        
        batch = {
            "bboxes": bboxes,
            "observation.state": torch.randn(batch_size, 10)  # Dummy state tensor
        }
        
        # Call the method
        boxes_tensor, box_masks, point_tensor, point_masks = self.policy.prepare_box_and_point(batch)
        
        # Check shapes
        self.assertEqual(boxes_tensor.shape, (batch_size, max_num_boxes, 4))
        self.assertEqual(point_tensor.shape, (batch_size, max_num_boxes, 2))
        self.assertEqual(box_masks.shape, (batch_size, max_num_boxes))
        self.assertEqual(point_masks.shape, (batch_size, max_num_boxes))
        
        # Check that boxes_tensor contains the correct box data (only x, y, w, h)
        expected_boxes = torch.tensor([
            # Batch 0
            [
                [10.0, 20.0, 30.0, 40.0],  # First box
                [50.0, 60.0, 20.0, 30.0],  # Second box
                [100.0, 120.0, 40.0, 50.0],  # Third box
            ],
            # Batch 1
            [
                [5.0, 10.0, 15.0, 20.0],   # First box
                [30.0, 40.0, 10.0, 25.0],  # Second box
                [0.0, 0.0, 0.0, 0.0],      # Empty box
            ]
        ], dtype=torch.float32)
        
        # Check the first 3 boxes for batch 0 (the ones we provided)
        torch.testing.assert_close(boxes_tensor[0, :num_boxes, :], expected_boxes[0, :num_boxes, :])
        torch.testing.assert_close(boxes_tensor[1, :num_boxes, :], expected_boxes[1, :num_boxes, :])
        
        # Check that point_tensor contains the correct center points
        expected_points = torch.tensor([
            # Batch 0
            [
                [25.0, 40.0],   # (10 + 30/2, 20 + 40/2) = (25, 40)
                [60.0, 75.0],   # (50 + 20/2, 60 + 30/2) = (60, 75)
                [120.0, 145.0], # (100 + 40/2, 120 + 50/2) = (120, 145)
            ],
            # Batch 1
            [
                [12.5, 20.0],   # (5 + 15/2, 10 + 20/2) = (12.5, 20)
                [35.0, 52.5],   # (30 + 10/2, 40 + 25/2) = (35, 52.5)
                [0.0, 0.0],     # (0 + 0/2, 0 + 0/2) = (0, 0)
            ]
        ], dtype=torch.float32)
        
        # Check the first 3 points for both batches
        torch.testing.assert_close(point_tensor[0, :num_boxes, :], expected_points[0, :num_boxes, :])
        torch.testing.assert_close(point_tensor[1, :num_boxes, :], expected_points[1, :num_boxes, :])
        
        # Check masks
        # For batch 0: all boxes and points should be valid (True)
        expected_box_masks_batch0 = torch.tensor([True, True, True, False, False])
        expected_point_masks_batch0 = torch.tensor([True, True, True, False, False])
        
        # For batch 1: first two boxes and points should be valid, third is empty (False)
        expected_box_masks_batch1 = torch.tensor([True, True, False, False, False])
        expected_point_masks_batch1 = torch.tensor([True, True, False, False, False])
        
        torch.testing.assert_close(box_masks[0], expected_box_masks_batch0)
        torch.testing.assert_close(point_masks[0], expected_point_masks_batch0)
        torch.testing.assert_close(box_masks[1], expected_box_masks_batch1)
        torch.testing.assert_close(point_masks[1], expected_point_masks_batch1)

    def test_prepare_box_and_point_without_bboxes(self):
        """Test prepare_box_and_point when bboxes are not present in the batch"""
        batch_size = 2
        max_num_boxes = self.config.max_num_embeddings_box
        
        # Create a batch without bboxes
        batch = {
            "observation.state": torch.randn(batch_size, 10)  # Dummy state tensor
        }
        
        # Call the method (should only return boxes_tensor and box_masks)
        result = self.policy.prepare_box_and_point(batch)
        boxes_tensor, box_masks = result
        
        # Check shapes
        self.assertEqual(boxes_tensor.shape, (batch_size, max_num_boxes, 4))
        self.assertEqual(box_masks.shape, (batch_size, max_num_boxes))
        
        # Check that tensors are initialized with zeros
        self.assertTrue(torch.all(boxes_tensor == 0))
        self.assertTrue(torch.all(~box_masks))  # All masks should be False


if __name__ == "__main__":
    unittest.main()