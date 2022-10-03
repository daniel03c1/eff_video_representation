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

    def test_multi_hash_encoding(self):
        in_features, embedding_dim, n_levels = 3, 2, 3
        video = torch.rand(16, in_features, 90, 160)

        emb = MultiHashEncoding(video, embedding_dim, n_levels=n_levels)
        out_size = embedding_dim * n_levels
        self.assertEqual(emb.get_output_size(), out_size)
        self.assertEqual(emb(torch.rand(48, 32, in_features)).size(),
                         (48, 32, out_size))

    def test_test_encoding(self):
        in_features, embedding_dim = 3, 7
        volume = (16, 90, 160)

        emb = TestEmbedding(volume, embedding_dim)
        out_size = embedding_dim * 3
        self.assertEqual(emb.get_output_size(), out_size)
        self.assertEqual(emb(torch.rand(48, 32, in_features)).size(),
                         (48, 32, out_size))


if __name__ == '__main__':
    unittest.main()

