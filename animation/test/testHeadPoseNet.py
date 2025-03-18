import unittest
import torch
from animation.modules.headPose_net import HeadPoseNet

class TestHeadPoseNet(unittest.TestCase):
    def setUp(self):
        self.model = HeadPoseNet()
        self.input_tensor = torch.randn(1, 16, 3, 512, 512)

    def test_forward_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, ( 16, 320, 64, 64))

    def test_forward_values(self):
        output = self.model(self.input_tensor)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_initialize_weights(self):
        for m in self.model.conv_layers:
            if isinstance(m, torch.nn.Conv2d):
                self.assertTrue(torch.all(m.weight != 0))
                if m.bias is not None:
                    self.assertTrue(torch.all(m.bias == 0))
        self.assertTrue(torch.all(self.model.final_proj.weight == 0))
        if self.model.final_proj.bias is not None:
            self.assertTrue(torch.all(self.model.final_proj.bias == 0))

    def test_scale_parameter(self):
        self.assertEqual(self.model.scale.item(), 2)

    def test_from_pretrained(self):
        with self.assertRaises(FileNotFoundError):
            HeadPoseNet.from_pretrained("non_existent_path")

if __name__ == "__main__":
    unittest.main()