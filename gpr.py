import torch as tc
import numpy as np


class GPR(object):
    def __init__(self, x, y, cov, hp=None):

        self.x = tc.clone(x)
        self.y = tc.clone(y)
        self.cov = cov
        self.hp = hp

        self.krn = NotImplemented
        self.wt = NotImplemented
        self.krnchd = NotImplemented

        self.interpolate = NotImplemented

    def interpolant(self):

        self.krn = self.cov(self.x, hp=self.hp)
        print(self.krn)
        self.krnchd = tc.cholesky(self.krn)
        self.wt = tc.squeeze(
            tc.cholesky_solve(self.y.reshape(-1, 1), self.krnchd))

        def ff(xs):
            print(xs)
            krns = self.cov(self.x, xs=xs, hp=self.hp)

            ys = tc.mv(krns, self.wt)

            krnss = self.cov(xs, xs=xs, hp=self.hp)

            lks = tc.cholesky_solve(krns.transpose(0, 1), self.krnchd)

            covars = krnss - tc.mm(lks.transpose(0, 1), lks)

            return ys, covars

        self.interpolate = ff
