import unittest

import torch
from train import kd_loss


class Teacher(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.values = logits

    def forward(self, ids):
        return self.values


class KDTemperatureTests(unittest.TestCase):
    def test_calibrated_teacher_has_zero_kl_at_calibrated_student(self):
        logits = torch.tensor([[[3., -2., 1.], [0., 4., -3.]]])
        student = (logits / 1.2).clone().requires_grad_()
        loss = kd_loss(student, torch.tensor([0, 1]), Teacher(logits),
                       torch.zeros(1, 2, dtype=torch.long), 1., 3., 1.2,
                       torch.device('cpu'), 'fp32', 'teacher_calibrated')
        self.assertAlmostEqual(loss.item(), 0., places=6)
        loss.backward()
        self.assertLess(student.grad.abs().max().item(), 1e-6)

    def test_legacy_shared_has_zero_kl_at_raw_teacher(self):
        logits = torch.tensor([[[3., -2., 1.], [0., 4., -3.]]])
        loss = kd_loss(logits, torch.tensor([0, 1]), Teacher(logits),
                       torch.zeros(1, 2, dtype=torch.long), 1., 3., 1.2,
                       torch.device('cpu'), 'fp32')
        self.assertAlmostEqual(loss.item(), 0., places=6)


if __name__ == '__main__':
    unittest.main()
