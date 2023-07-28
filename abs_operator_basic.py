from functools import wraps
from math import ceil

import bottleneck as bk
import numpy as np
import pandas as pd
import talib as ta
from scipy.stats import norm, rankdata


class OPTOOL:
    def __init__(
        self,
        uInstrNum=6000,
        diNum=99,
        tiNum=16,
        n_sample=99,
        dftp_s=99,
        dftp_l=99,
        fill="nofill",
        industry=np.array([]),
        marketsize=np.array([]),
        normd=21,
        eps=1e-5,
        correps=1e-5,
        adigit=7,
        timeperiod=120,
        sharedstyle=False,
        sharedpath=None,
        needpatch=True,
        mode="single",
    ):
        assert (sharedstyle) ^ (sharedpath is None)
        if mode == "mat":
            self.tick_corr = self.tick_corr_mat
            self.tick_cov = self.tick_cov_mat
            self.tick_tsmax = self.tick_tsmax_mat
            self.tick_tsmin = self.tick_tsmin_mat
            self.tick_tsstd = self.tick_tsstd_mat
            self.tick_tsmean = self.tick_tsmean_mat
            self.tick_tssum = self.tick_tssum_mat
            self.tick_tsrank = self.tick_tsrank_mat
            self.tick_tsargmax = self.tick_tsargmax_mat
            self.tick_tsargmin = self.tick_tsargmin_mat
        elif mode == "single":
            self.tick_corr = self.tick_corr_single
            self.tick_cov = self.tick_cov_single
            self.tick_tsmax = self.tick_tsmax_single
            self.tick_tsmin = self.tick_tsmin_single
            self.tick_tsstd = self.tick_tsstd_single
            self.tick_tsmean = self.tick_tsmean_single
            self.tick_tssum = self.tick_tssum_single
            self.tick_tsrank = self.tick_tsrank_single
            self.tick_tsargmax = self.tick_tsargmax_single
            self.tick_tsargmin = self.tick_tsargmin_single
        else:
            raise NotImplementedError

        self.uInstrNum = uInstrNum
        self.normd = normd
        self.tiNum = tiNum
        self.n_sample = n_sample
        self.dftp_s = dftp_s
        self.dftp_l = dftp_l
        self.fill = fill

        self.epsilon = eps
        self.correpsilon = correps
        self.adigit = adigit
        self.timeperiod = timeperiod
        self.needpatch = needpatch

        self.refresh_function()

    def refresh_function(
        self,
    ):
        self.timerange = (2, 50)

    def neg(self, x):
        return -x

    def crsrank(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        nx = np.round(x, 6).reshape(-1, self.uInstrNum)
        validnum = np.isfinite(nx).sum(axis=1, keepdims=True)
        return (bk.nanrankdata(nx, axis=1) - (validnum + 1) / 2).reshape(
            -1,
        )

    def biascrsrank(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        nx = np.round(x, 6).reshape(-1, self.uInstrNum)
        return bk.nanrankdata(nx, axis=1).reshape(-1)

    def normcrsrank(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        nx = np.round(x, 6).reshape(-1, self.uInstrNum)
        validnum = np.isfinite(nx).sum(axis=1, keepdims=True)
        return (bk.nanrankdata(nx, axis=1) / validnum - 0.5).reshape(
            -1,
        )

    def opmax(self, x, y):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        x = np.where(np.isfinite(x), x, y)
        y = np.where(np.isfinite(y), y, x)
        return self.fillnan(np.where(x > y, x, y))

    def opmin(self, x, y):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        x = np.where(np.isfinite(x), x, y)
        y = np.where(np.isfinite(y), y, x)
        return self.fillnan(np.where(x < y, x, y))

    def ffillbase(self, x):
        x = x.reshape(-1, self.uInstrNum).T
        mask = ~np.isfinite(x)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = x[np.arange(idx.shape[0])[:, None], idx].T.reshape(-1)
        out[np.isnan(out)] = 0
        return out

    def ffill(self, x):
        x = x.reshape(-1, self.uInstrNum)
        mask = np.isfinite(x)
        idx = np.where(mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        out = x[idx, np.arange(idx.shape[1])[None, :]].reshape(-1)
        out[np.isnan(out)] = 0
        return out

    def bfill(self, x):
        x = np.flipud(x.reshape(-1, self.uInstrNum))
        x = self.ffill(x).reshape(-1, self.uInstrNum)
        x = np.flipud(x).reshape(
            -1,
        )
        return x

    def fillinf(func):
        @wraps(func)
        def fillfunc(*args, **kwargs):
            mask = np.array(
                [np.isfinite(ag) for ag in args if isinstance(ag, np.ndarray)]
            ).all(axis=0)

            x = func(*args, **kwargs)
            # In Case that output is a integer array.
            if x.dtype == "float":
                x[np.isinf(x)] = np.nan
            x[~mask] = np.nan
            return x

        return fillfunc

    def filterinf(func):
        @wraps(func)
        def fillfunc(*args, **kwargs):
            args = [
                np.where(np.isinf(i), np.nan, i) if isinstance(i, np.ndarray) else i
                for i in args
            ]
            x = func(*args, **kwargs)
            return x

        return fillfunc

    # Used to Fill Unusual Value: Zero-Fill & Crsmean-fill
    def fillnan(self, x):
        if x.dtype == "int":
            x = x.astype("float")
        x[np.isinf(x)] = np.nan
        if self.fill == "zerofill":
            return self.zerofill(x)
        elif self.fill == "crsmeanfill":
            return self.crsmeanfill(x)
        elif self.fill == "nofill":
            return x
        elif self.fill == "ffill":
            return self.ffill(x)
        elif self.fill == "bfill":
            return self.bfill(x)
        else:
            print("Error: Illegal Filling Strategy:%s" % (self.fill))
            exit()

    def zerofill(self, x):
        return np.where(np.isfinite(x), x, 0)

    def crsmeanfill(self, x):
        # fill nan with the avaerage of crosssectional stock signal.
        if np.isfinite(x).all():
            return x
        else:
            nx = x.reshape((-1, self.uInstrNum))
            validmean = np.nanmean(nx, axis=1, keepdims=True)
            validmean[~np.isfinite(validmean)] = 0
            return np.where(np.isfinite(nx), nx, validmean).reshape((-1,))

    def crsstd(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        x = x.copy()
        x[np.isinf(x)] = np.nan
        res = x.reshape((-1, self.uInstrNum))
        res = self.div(
            (res - np.nanmean(res, axis=1, keepdims=True)),
            np.nanstd(res, axis=1, keepdims=True),
        )
        return self.fillnan(res.reshape((-1,)))

    @fillinf
    def capcrsstd(self, x, capthres=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        x = x.reshape(-1, self.uInstrNum)
        x = x - np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        mask = np.abs(x) < capthres * std
        sign = np.sign(x)
        x = np.where(mask, x, np.nan)
        x = self.div(
            (x - np.nanmean(x, axis=1, keepdims=True)),
            np.nanstd(x, axis=1, keepdims=True),
        )
        x = np.where(
            mask,
            x,
            (sign > 0) * np.nanmax(x, axis=1, keepdims=True)
            + (sign < 0) * np.nanmin(x, axis=1, keepdims=True),
        )
        return x.reshape(
            -1,
        )

    def cap(self, x):
        return np.clip(x, -10, 10)

    def crsabs(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        res = x.reshape((-1, self.uInstrNum))
        res = (res - np.nanmean(res, axis=1, keepdims=True)) / np.nansum(
            np.abs(res), axis=1, keepdims=True
        )
        return self.fillnan(res.reshape((-1,)))

    def destd(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        res = x.reshape((-1, self.uInstrNum))
        res = self.div(res, np.nanstd(res, axis=1, keepdims=True))
        return self.fillnan(res.reshape((-1,)))

    def denorm(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        res = x.reshape((-1, self.uInstrNum))
        res = self.div(res, np.sqrt(np.nanmean(res**2, axis=1, keepdims=True)))
        return self.fillnan(res.reshape((-1,)))

    def crsdemean(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        rx = x.reshape((-1, self.uInstrNum))
        ax = rx - (np.nanmean(rx, axis=1, keepdims=True))
        return ax.reshape((-1,))

    def crsmean(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        return x - self.crsdemean(x)

    def roll_np(self, x, shift, axis):
        if axis != 0:
            raise
        rolled = np.roll(x, axis=0, shift=shift)
        rolled[:shift] = np.nan
        return rolled

    def roll(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        res = x.reshape((-1, self.uInstrNum))
        return self.fillnan(np.roll(res, axis=0, shift=5).reshape((-1,)))

    def tick_nanroll(self, x, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        res = x.reshape((-1, self.uInstrNum))
        res = np.roll(res, axis=0, shift=tick)
        res[:tick] = np.nan
        return self.fillnan(res.reshape((-1,)))

    def delta(self, x):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        return x - self.roll(x)

    def tick_delta(self, x, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        return self.fillnan(x - self.tick_nanroll(x, tick))

    def tick_lineardecay(self, x, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        # not use diNum and tiNum to calculate correctly in little sample
        tick_ldecay_mat = np.array(
            [
                [[i] * self.uInstrNum] * int(x.shape[0] / self.uInstrNum)
                for i in range(tick, 0, -1)
            ]
        )

        nx = x.reshape((-1, self.uInstrNum))
        return (
            np.sum(tick_ldecay_mat * nx, axis=0) / np.sum(list(range(1, tick + 1)))
        ).reshape((-1,))

    def tick_ddratio(self, x, y, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            rawres = self.div(self.tick_delta(x, tick), self.tick_delta(y, tick)) - 1
        return self.fillnan(rawres)

    # Default 5 ticks
    def TSfunc(self, func, x, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        res = x.reshape((-1, self.uInstrNum))
        res = np.stack(
            [self.roll_np(res, shift=i, axis=0).reshape((-1,)) for i in range(tick)]
        )
        return self.fillnan(func(res))

    def tsmaxfunc(self, stacked_x, axis=0):
        return np.nanmax(stacked_x, axis=axis)

    def tsminfunc(self, stacked_x, axis=0):
        return np.nanmin(stacked_x, axis=axis)

    def tsargmaxfunc(self, stacked_x, axis=0):
        mask = np.isnan(stacked_x).all(axis=axis)
        stacked_x[:, mask] = 0
        res = np.nanargmax(stacked_x, axis=axis).astype("float")
        res[mask] = np.nan
        return res

    def tsargminfunc(self, stacked_x, axis=0):
        mask = np.isnan(stacked_x).all(axis=axis)
        stacked_x[:, mask] = 0
        res = np.nanargmin(stacked_x, axis=axis).astype("float")
        res[mask] = np.nan
        return res
        # return np.nanargmin(stacked_x, axis = axis).astype('float')

    def tsstdfunc(self, stacked_x, axis=0):
        return np.nanstd(stacked_x, axis=axis)

    def tsmeanfunc(self, stacked_x, axis=0):
        return np.nanmean(stacked_x, axis=axis)

    def tssumfunc(self, stacked_x, axis=0):
        return np.nansum(stacked_x, axis=axis)

    def tsrankfunc(self, stacked_x, axis=0):
        mask = np.isnan(stacked_x[0])
        validnum = np.sum(np.isfinite(stacked_x), axis=axis)
        low = np.sum(stacked_x < stacked_x[[0], :], axis=axis)
        high = validnum - np.sum(stacked_x > stacked_x[[0], :], axis=axis)
        rank = (high + low - 1) / (validnum - 1) - 1
        rank[validnum == 1] = 0
        rank[mask] = np.nan
        return rank

    # tick mirror

    """
    Next is single-step version of tsfunction
    """

    @fillinf
    def tick_tsmax_single(self, x, tick=5):
        x = x.reshape(-1, self.uInstrNum)
        return bk.move_max(x, tick, min_count=1, axis=0).reshape(
            -1,
        )

    @fillinf
    def tick_tsmin_single(self, x, tick=5):
        x = x.reshape(-1, self.uInstrNum)
        return bk.move_min(x, tick, min_count=1, axis=0).reshape(
            -1,
        )

    @fillinf
    def tick_tsargmax_single(self, x, tick=5):
        x = np.round(x, 5).reshape(-1, self.uInstrNum)
        return bk.move_argmax(x, tick, min_count=1, axis=0).reshape(
            -1,
        )

    @fillinf
    def tick_tsargmin_single(self, x, tick=5):
        x = np.round(x, 5).reshape(-1, self.uInstrNum)
        return bk.move_argmin(x, tick, min_count=1, axis=0).reshape(
            -1,
        )

    @fillinf
    def tick_tsstd_single(self, x, tick=5):
        x = x.reshape(-1, self.uInstrNum)
        res = bk.move_std(x, tick, min_count=1, axis=0).reshape(
            -1,
        )
        if self.needpatch:
            res = np.around(res, self.adigit)
        return res

        # return self.TSfunc(self.tsstdfunc, x, tick)

    @fillinf
    def tick_tsmean_single(self, x, tick=5):
        x = x.reshape(-1, self.uInstrNum)
        res = bk.move_mean(x, tick, min_count=1, axis=0).reshape(
            -1,
        )
        if self.needpatch:
            res = np.around(res, self.adigit)
        return res

        # return self.TSfunc(self.tsmeanfunc, x, tick)

    @fillinf
    def tick_tssum_single(self, x, tick=5):
        x = x.reshape(-1, self.uInstrNum)
        res = bk.move_sum(x, tick, min_count=1, axis=0).reshape(
            -1,
        )
        if self.needpatch:
            res = np.around(res, self.adigit)
        return res

    """
    @fillinf
    def tick_tsstd(self, x, tick=5):
        x = self.bfill(x).reshape(-1, self.uInstrNum)
        res = np.empty((self.uInstrNum, x.shape[0]))
        for i in range(self.uInstrNum):
            res[i] = ta.STDDEV(x[:, i], tick)
        res = res.T.reshape(-1, )
        #res[np.abs(res)<self.epsilon] = 0
        return res


    @fillinf
    def tick_tsmean(self, x, tick=5):
        x = self.bfill(x).reshape(-1, self.uInstrNum)
        res = np.empty((self.uInstrNum, x.shape[0]))
        for i in range(self.uInstrNum):
            res[i] = ta.MA(x[:, i], tick)
        res = res.T.reshape(-1, )
        #res[np.abs(res)<self.epsilon] = 0
        return res

    @fillinf
    def tick_tssum(self, x, tick=5):
        return tick*self.tick_tsmean(x, tick)
    """

    @fillinf
    def tick_tsmeandiff(self, x, tick=5):
        return x - self.tick_tsmean(x, tick)

    @fillinf
    def tick_tsrank_single(self, x, tick=5):
        x = np.round(x, 6).reshape(-1, self.uInstrNum)
        return (
            tick
            / 2
            * bk.move_rank(x, tick, min_count=1, axis=0).reshape(
                -1,
            )
        )

    """
    The Next is for matrix(long term sim) tsfunc
    """

    @fillinf
    def tick_tsmax_mat(self, x, tick=5):
        return self.TSfunc(self.tsmaxfunc, x, tick)

    @fillinf
    def tick_tsmin_mat(self, x, tick=5):
        return self.TSfunc(self.tsminfunc, x, tick)

    @fillinf
    def tick_tsstd_mat(self, x, tick=5):
        res = self.TSfunc(self.tsstdfunc, x, tick)
        return res

    @fillinf
    def tick_tsmean_mat(self, x, tick=5):
        return self.TSfunc(self.tsmeanfunc, x, tick)

    @fillinf
    def tick_tssum_mat(self, x, tick=5):
        # x = self.bfill(x)
        return self.TSfunc(self.tssumfunc, x, tick)

    @fillinf
    def tick_tsrank_mat(self, x, tick=5):
        x = np.round(x, 6)
        return self.TSfunc(self.tsrankfunc, x, tick) * tick / 2

    @fillinf
    def tick_tsargmax_mat(self, x, tick=5):
        x = np.round(x, 5)
        return self.TSfunc(self.tsargmaxfunc, x, tick)

    @fillinf
    def tick_tsargmin_mat(self, x, tick=5):
        x = np.round(x, 5)
        return self.TSfunc(self.tsargminfunc, x, tick)

    def alp_fitness(self, y, y_pred):
        if y.shape[0] % self.uInstrNum != 0:
            return np.nan

        y_pred = self.fillnan(y_pred)

        y = y.reshape((-1, self.uInstrNum))
        npred_y = y_pred.reshape((-1, self.uInstrNum))

        corrlist = self.corrfunc(y, npred_y, axis=1)
        corrlist = np.where(np.isfinite(corrlist), corrlist, 0)

        return np.nanmean(corrlist)

    def pearson(self, y, y_pred):
        if not np.isfinite(y_pred).all():
            print("Error: y_pred contains NaN or Inf, Exit.")
            exit()

        mask = np.isfinite(y)
        y = y[mask]
        y_pred = y_pred[mask]

        coef = np.corrcoef(y, y_pred)[0, 1]
        return np.corrcoef(y, y_pred)[0, 1]

    def tsic_list(self, y, y_pred):
        tsy = y.reshape((-1, self.tiNum, self.uInstrNum))
        tsy_pred = y_pred.reshape((-1, self.tiNum, self.uInstrNum))
        iclist = self.corrfunc(tsy, tsy_pred, axis=1)
        return iclist

    def tsic(self, y, y_pred):
        iclist = self.tsic_list(y, y_pred)
        iclist = np.where(np.isfinite(iclist), iclist, 0)
        return np.mean(iclist)

    def cossimic_list(self, y, y_pred):
        tsy = y.reshape((-1, self.tiNum, self.uInstrNum))
        tsy_pred = y_pred.reshape((-1, self.tiNum, self.uInstrNum))
        iclist = self.cossimfunc(tsy, tsy_pred, axis=1)
        return iclist

    def cossimic(self, y, y_pred):
        iclist = self.cossimic_list(y, y_pred)
        iclist = np.where(np.isfinite(iclist), iclist, 0)
        return np.mean(iclist)

    def aiocossimic_list(self, y, y_pred):
        tsy = y.reshape((-1, self.uInstrNum))
        tsy_pred = y_pred.reshape((-1, self.uInstrNum))
        iclist = self.cossimfunc(tsy, tsy_pred, axis=0)
        return iclist

    def aiocossimic(self, y, y_pred):
        iclist = self.aiocossimic_list(y, y_pred)
        iclist = np.where(np.isfinite(iclist), iclist, 0)
        return np.mean(iclist)

    def olcossimic(self, y, y_pred):
        mask = np.isfinite(y)
        my = y[mask]
        my_pred = y_pred[mask]
        return my.dot(my_pred) / np.linalg.norm(my) / np.linalg.norm(my_pred)

    def crscossimic(self, y, y_pred):
        y = y.reshape((-1, self.uInstrNum))
        y_pred = y_pred.reshape((-1, self.uInstrNum))
        iclist = np.nansum(y * y_pred, axis=1) / np.sqrt(
            np.nansum(y * y, axis=1) * np.nansum(y_pred * y_pred, axis=1)
        )
        iclist = self.zerofill(iclist)
        ic = np.mean(iclist)
        if np.isnan(ic):
            ic = 0

        return ic

    def longonlyolcossimic(self, y, y_pred):
        longic = self.olcossimic(y, partx(y_pred, 1)) / np.sqrt(2)
        if longic > 0:
            return longic
        else:
            return 0

    def shortonlyolcossimic(self, y, y_pred):
        shortic = self.olcossimic(y, partx(y_pred, 0)) / np.sqrt(2)
        if shortic > 0:
            return shortic
        else:
            return 0

    def capic(self, y, y_pred):
        return self.olcossimic(y, const_winsorize(y_pred, 5))

    def betacossimic(self, y, y_pred):
        return self.olcossimic(
            y.reshape((-1, self.uInstrNum))[:, 0],
            np.nansum(y_pred.reshape((-1, self.uInstrNum)), axis=1),
        )

    def topret(self, y, y_pred):
        index = np.argsort(y_pred)[::-1][: int(y.shape[0] * 0.1)]
        topret = np.nanmean(y[index])
        if topret > 0:
            return topret
        else:
            return 0

    def rankic(self, y, y_pred):
        ranky_pred = self.crsrank(y_pred)
        return self.olcossimic(y, ranky_pred)

    # Default 5 ticks
    def TSfunc_2(self, func, x, y, tick=5):
        if x.shape[0] % self.uInstrNum != 0:
            return np.nan

        stack1 = x.reshape((-1, self.uInstrNum))
        stack1 = np.stack(
            [self.roll_np(stack1, shift=i, axis=0).reshape((-1,)) for i in range(tick)]
        )

        stack2 = y.reshape((-1, self.uInstrNum))
        stack2 = np.stack(
            [self.roll_np(stack2, shift=i, axis=0).reshape((-1,)) for i in range(tick)]
        )

        return self.fillnan(func(stack1, stack2))

    def covfunc(self, stacked_x, stacked_y, axis=0):
        xmean = np.nanmean(stacked_x, axis=axis, keepdims=True)
        ymean = np.nanmean(stacked_y, axis=axis, keepdims=True)

        xstd = np.nanstd(stacked_x, axis=axis)
        ystd = np.nanstd(stacked_y, axis=axis)

        abcov = np.nanmean((stacked_x - xmean) * (stacked_y - ymean), axis=axis)
        return abcov

    def corrfunc(self, stacked_x, stacked_y, axis=0):
        xmean = np.nanmean(stacked_x, axis=axis, keepdims=True)
        ymean = np.nanmean(stacked_y, axis=axis, keepdims=True)

        xstd = np.nanstd(stacked_x, axis=axis)
        ystd = np.nanstd(stacked_y, axis=axis)

        abcov = np.nanmean((stacked_x - xmean) * (stacked_y - ymean), axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            tmpres = self.div(abcov, (xstd * ystd))
        return tmpres

    def cossimfunc(self, stacked_x, stacked_y, axis=0):
        xydot = (stacked_x * stacked_y).sum(axis=axis)
        xnorm = np.linalg.norm(stacked_x, 2, axis=axis)
        ynorm = np.linalg.norm(stacked_y, 2, axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            tmpres = xydot / (xnorm * ynorm)
        return tmpres

    @fillinf
    def tick_corr_single(self, x, y, tick=5):
        x = np.where(np.isfinite(y), x, np.nan)
        y = np.where(np.isfinite(x), y, np.nan)
        ca = self.tick_tsmean(x * y, tick)
        cb = self.tick_tsmean(x, tick) * self.tick_tsmean(y, tick)
        if self.needpatch:
            cb = np.around(cb, self.adigit)
        cc = self.tick_tsstd(x, tick)
        cd = self.tick_tsstd(y, tick)

        cor = self.div((ca - cb), (cc * cd))

        cor[cor == 0] = np.nan

        return cor.reshape(
            -1,
        )

    @fillinf
    def tick_cov_single(self, x, y, tick=5):
        x = np.where(np.isfinite(y), x, np.nan)
        y = np.where(np.isfinite(x), y, np.nan)
        x = x.reshape(-1, self.uInstrNum)
        y = y.reshape(-1, self.uInstrNum)
        cov = bk.move_mean(x * y, tick, min_count=1, axis=0) - bk.move_mean(
            x, tick, min_count=1, axis=0
        ) * bk.move_mean(y, tick, min_count=1, axis=0)
        return cov.reshape(
            -1,
        )

    @fillinf
    def tick_corr_mat(self, x, y, tick=5):
        x = np.where(np.isfinite(y), x, np.nan)
        y = np.where(np.isfinite(x), y, np.nan)
        res = self.TSfunc_2(self.corrfunc, x, y, tick)

        res[res == 0] = np.nan

        return res

    @fillinf
    def tick_cov_mat(self, x, y, tick=5):
        x = np.where(np.isfinite(y), x, np.nan)
        y = np.where(np.isfinite(x), y, np.nan)
        res = self.TSfunc_2(self.covfunc, x, y, tick)
        return res

    def tick_zcorr(self, x, y, tick=5):
        return self.tick_corr(x + y, x - y, tick=tick)

    def add(self, x, y):
        return self.fillnan(np.add(x, y))

    def sub(self, x, y):
        return self.fillnan(np.subtract(x, y))

    def mul(self, x, y):
        return self.fillnan(np.multiply(x, y))

    def mul_CONST(self, x, w):
        return self.fillnan(x * w)

    def div(self, x, y):
        # y = np.around(y, self.adigit)
        # x = np.around(x, self.adigit)
        y = np.where(np.abs(y) < self.epsilon, 0, y)
        x = np.where(np.abs(x) < self.epsilon, 0, x)
        res = self.fillnan(np.divide(x, y))
        res = np.where(np.isinf(res), np.nan, res)
        return self.fillnan(res)

    def AND(self, x, y):
        return x & y

    def OR(self, x, y):
        return x | y

    def greater(self, x, y):
        return x > y

    def greater2(self, x, y):
        return x >= y

    def less(self, x, y):
        return x < y

    def less2(self, x, y):
        return x <= y

    def equal(self, x, winsor):
        return x == winsor

    def notequal(self, x, winsor):
        return x != winsor

    def sqrt(self, x):
        res = np.sign(x) * (np.sqrt(np.abs(x)))
        return self.fillnan(res)

    def square(self, x):
        return self.fillnan(np.sign(x) * (x**2))

    def power(self, x, tick=2):
        return self.fillnan((x**tick))

    def deltaratio(self, x, y):
        return self.fillnan(self.div(x, y) - 1)

    def normdeltaratio(self, x, y):
        return self.fillnan(self.div((x - y), (x + y)))

    def vcabs(self, x):
        return self.fillnan(np.abs(x))

    def Wgttsmean(self, x, kernel):
        x = x.reshape(-1, self.uInstrNum)
        out = np.apply_along_axis(np.convolve, 0, x, kernel)[: x.shape[0]]
        return out.reshape(-1)

    def tick_exptsmean(self, x, tick=5, alpha=0.6):
        kernel = np.logspace(tick - 1, 0, tick, base=alpha)
        kernel = kernel / np.sum(kernel)
        print(kernel)
        return self.Wgttsmean(x, kernel)

    def tick_lineartsmean(self, x, tick=5):
        kernel = np.linspace(1, tick, tick)
        kernel = kernel / np.sum(kernel)
        return self.Wgttsmean(x, kernel)

    ########################################NOTIFICATION################################################
    ####################################################################################################
    # Please check your norm module is NAN-friendly at the very first time
    # Otherwise there is a very high probability that the result of the program will end up full of NAN
    ####################################################################################################

    # Not Safe.
    def pn_norm(self, x):
        rx = rankdata(x)
        r = (x < 0).sum()
        l = x.shape[0] + 1
        res = norm.ppf(np.where(x >= 0, (rx - r) / 2 / (l - r) + 0.5, rx / 2 / r))
        return res

    # Not Safe.
    def mynorm(self, x):
        return np.apply_along_axis(
            self.pn_norm, 0, x.reshape((-1, self.uInstrNum))
        ).reshape((-1,))

    # Safe
    def basenormmean(self, x, axis=0, l=2):
        return np.sqrt(np.nanmean(x**l, axis=axis))

    # Safe.
    def fdnorm(self, x):
        normrawdata = x.reshape((-1, self.uInstrNum))[:250, :]
        validnum = np.isfinite(normrawdata).sum(axis=0)
        normres = np.linalg.norm(
            np.where(np.isfinite(normrawdata), normrawdata, 0), axis=0
        ) / np.sqrt(validnum)
        return (x.reshape((-1, self.uInstrNum)) / normres).reshape((-1,))

    # Safe.
    # This Code is a first version of lmnorm.Align forward, which is not friendly to realtime and sim check.
    def lmnorm(self, x, cachemode=False):
        ticks = self.normd * self.tiNum
        allticks = int(x.shape[0] / self.uInstrNum)
        lent = allticks // ticks * ticks
        rx = x[: lent * self.uInstrNum].reshape((-1, ticks, self.uInstrNum))
        # norm = np.broadcast_to(np.linalg.norm(rx, axis = 1, keepdims = True)/np.sqrt(ticks), rx.shape).reshape((-1, ))
        norm = np.broadcast_to(
            np.sqrt(
                self.div(
                    np.nansum(rx * rx, axis=1, keepdims=True),
                    np.sum(np.isfinite(rx), axis=1, keepdims=True),
                )
            ),
            rx.shape,
        ).reshape((-1,))
        tail = np.broadcast_to(
            norm[-self.uInstrNum :], (allticks % ticks, self.uInstrNum)
        ).reshape((-1,))
        norm = np.append(norm, tail)
        norm = self.roll_np(
            norm.reshape((-1, self.uInstrNum)), axis=0, shift=ticks
        ).reshape((-1,))

        if cachemode:
            return self.fillnan(self.div(x, norm)), norm
        else:
            return self.fillnan(self.div(x, norm))

    # Safe.
    def rollingnorm(self, x, tick=50):
        res = x.reshape((-1, self.uInstrNum))
        res = np.stack(
            [np.roll(res, shift=i, axis=0).reshape((-1,)) for i in range(tick)]
        )
        return x / self.basenormmean(res, axis=0).reshape((-1,))

    """
    ----------------------------INTERVAL OP-----------------------------
    """

    def interval_baseop(self, x, interval, func):
        x = x.reshape(-1, self.uInstrNum)

        totalslot = x.shape[0]
        lap = totalslot % interval

        cand = func(x[: totalslot - lap].reshape(-1, interval, self.uInstrNum)).reshape(
            -1, self.uInstrNum
        )
        cand = np.concatenate(
            [np.full((interval, self.uInstrNum), np.nan), cand[: totalslot - interval]],
            axis=0,
        )
        return cand.reshape(
            -1,
        )

    def interval_meanfunc(self, x):
        return np.broadcast_to(np.nanmean(x, axis=1, keepdims=True), x.shape)

    def interval_stdfunc(self, x):
        return np.broadcast_to(np.nanstd(x, axis=1, keepdims=True), x.shape)

    @fillinf
    def interval_mean(self, x, interval):
        return self.interval_baseop(x, interval, self.interval_meanfunc)

    @fillinf
    def interval_std(self, x, interval):
        return self.interval_baseop(x, interval, self.interval_stdfunc)

    @fillinf
    def filtervalue(self, x, value):
        return np.where(np.abs(x) > value, np.nan, x)

    """
    --------------------------------CUM OP------------------------------------
    """

    def cum_baseop(self, x, func):
        x = x.reshape(-1, self.uInstrNum)

        totalslot = x.shape[0]
        lap = totalslot % self.tiNum

        cand = func(
            x[: totalslot - lap].reshape(-1, self.tiNum, self.uInstrNum)
        ).reshape(-1, self.uInstrNum)
        if lap > 0:
            cand2 = func(x[None, totalslot - lap :]).reshape(-1, self.uInstrNum)
            cand = np.concatenate([cand, cand2], axis=0)
        return cand.reshape(
            -1,
        )

    def cum_meanfunc(
        self,
        x,
    ):
        tinum = x.shape[1]
        coef = np.arange(1, tinum + 1)[None, :, None]
        x = np.cumsum(x, axis=1) / coef
        return x

    @fillinf
    def cum_mean(self, x):
        return self.cum_baseop(x, self.cum_meanfunc)

    """
    -----------------------------STD--------------------------------
    """

    @fillinf
    def minmax_std(self, x):
        x = x.reshape(-1, self.uInstrNum)
        xmax = np.nanpercentile(x, 99, axis=1, keepdims=True)
        xmin = np.nanpercentile(x, 1, axis=1, keepdims=True)
        return self.div((x - xmin), (xmax - xmin)).reshape(
            -1,
        )
