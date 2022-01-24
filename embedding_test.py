import torch
import unittest
from embedding import *


class EmbeddingTest(unittest.TestCase):
    def test_embedding(self):
        emb = Embedding()
        self.assertRaises(Exception, emb.forward, None)
        self.assertRaises(Exception, emb.get_output_size)

    def test_rff(self):
        in_features, emb_size, emb_sigma = 3, 5, 3.14
        emb = RFF(in_features, emb_size, emb_sigma)

        self.assertEqual(emb.get_output_size(), emb_size*2)
        self.assertEqual(emb(torch.ones(32, in_features)).size(),
                         (32, emb_size*2))

    def test_pos_encoding(self):
        in_features, n_freqs = 3, 3

        # case 1: include inputs
        emb = PosEncoding(in_features, n_freqs, True)
        self.assertEqual(emb.get_output_size(), in_features*(2*n_freqs + 1))
        self.assertEqual(emb(torch.ones(32, in_features)).size(),
                         (32, in_features*(2*n_freqs + 1)))
        self.assertEqual(emb.freq_mat.requires_grad, False)

        # case 2: exclude inputs
        emb = PosEncoding(in_features, n_freqs, False)
        self.assertEqual(emb.get_output_size(), in_features*2*n_freqs)
        self.assertEqual(emb(torch.ones(32, in_features)).size(),
                         (32, in_features*2*n_freqs))


if __name__ == '__main__':
    unittest.main()

