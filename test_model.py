import unittest
import torch
from model import LayerNorm, Head, MHA, Block

class TestLayerNorm(unittest.TestCase):

    def test_layer_norm(self):
        n_e = 8
        batch_size = 4

        u = torch.randn(batch_size, n_e)
        norm = LayerNorm(n_e)
        o = norm(u)

        self.assertEqual(o.size(), u.size())
        self.assertTrue(torch.allclose(o.mean(dim=-1), torch.zeros(batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(o.var(dim=-1), torch.ones(batch_size), atol=1e-5))

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.n_e = 8
        self.sz = 2
        self.n_c = 5
        self.n_h = 4
        self.batch_size = 4
        self.drop_out = 0.20

        self.u = torch.randn(self.batch_size, self.n_c, self.n_e)

        self.head = Head(self.n_e, self.sz)
        self.mha = MHA(self.n_e, self.n_h)
        self.block = Block(self.n_e, self.n_h, self.drop_out)
    
    def test_self_attention(self):

        self.assertEqual(self.head.q.in_features, self.n_e)
        self.assertEqual(self.head.q.out_features, self.sz)
        self.assertEqual(self.head.k.out_features, self.sz)
        self.assertEqual(self.head.k.in_features, self.n_e)
        self.assertEqual(self.head.v.in_features, self.n_e)
        self.assertEqual(self.head.v.out_features, self.sz)

        o = self.head(self.u)

        self.assertEqual(o.shape, torch.Size((self.batch_size, self.n_c, self.sz)))


    def test_mha(self):
        o = self.mha(self.u)
        self.assertEqual(self.u.size(), o.size())
    
    def test_block(self):
        o = self.block(self.u)
        self.assertEqual(o.size(), self.u.size())


        

if __name__ == '__main__':
    unittest.main()
