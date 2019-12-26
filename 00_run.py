#!/usr/bin/python3
# 2019/12/26 ST

import sys, pathlib, zipfile, wave, numpy
from numba import jit

def norm_wav(ifname, ofname):
    def sv56(X, sf=48000, DesireddB=-26, THRES_NO=15, T=0.03, M=15.9, MIN_LOG=1.0e-20): # https://www.itu.int/rec/T-REC-P.56
        def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):
            if (numpy.abs((upcount - upthr) - Margin) < tol):
                return upcount
            if (numpy.abs((lwcount - lwthr) - Margin) < tol):
                return lwcount
            i = 0
            midcount = (upcount + lwcount) / 2.0
            midthr = (upthr + lwthr) / 2.0
            diff = midcount - midthr - Margin
            while numpy.abs(diff) > tol:
                i = i+1
                if (i > 20):
                    tol *= 1.1

                if diff > tol:
                    midcount = (upcount + midcount) / 2.0
                    midthr = (upthr + midthr) / 2.0
                    lwcount = midcount
                    lwthr = midthr
                elif diff < -tol:
                    midcount = (midcount + lwcount) / 2.0
                    midthr = (midthr + lwthr) / 2.0
                    upcount = midcount
                    upthr = midthr
                diff = midcount - midthr - Margin

            return midcount

        @jit
        def cal_p():
            p = numpy.zeros(len(X), numpy.float32)
            p[0] = (1.0 - g) * numpy.abs(X[0])
            for i in range(1, len(X)):
                p[i] = g * p[i-1] + (1.0 - g) * numpy.abs(X[i])
            return p

        @jit
        def cal_q():
            q = numpy.zeros(len(X), numpy.float32)
            q[0] = (1.0 - g) * p[0]
            for i in range(1, len(X)):
                q[i] =  g * q[i-1] + (1.0 - g) * p[i]
            return q


        SHORT_MAX = 32768.0
        I = int(0.2 * sf + 0.5)
        c = numpy.power(0.5, numpy.arange(1, THRES_NO+1))
        g = numpy.exp(-1.0 / (sf * T))

        X /= SHORT_MAX
        sq = numpy.sum(X ** 2)

        p = cal_p()
        q = cal_q()

        a = numpy.zeros(THRES_NO, numpy.int32)
        for i in range(THRES_NO):
            b = q >= c[i]
            for j in numpy.where(numpy.diff(b.astype(numpy.int))<0)[0]:
                b[j+1:j+1+I] = True
            a[i] += numpy.sum(b)

        i = numpy.where(a>0)[0]
        c, a = c[i], a[i]
        AdB = 10 * numpy.log10(sq / a + MIN_LOG)
        CdB = 20 * numpy.log10(c + MIN_LOG)

        i = numpy.where(AdB - CdB <= M)[0][-1]
        ActivedB = bin_interp(AdB[i], AdB[i+1], CdB[i], CdB[i+1], M, 0.5)
        factor = numpy.power(10.0, (DesireddB-ActivedB) / 20.0)

        return (X * factor * SHORT_MAX).astype(numpy.int16)

    with wave.Wave_read(ifname) as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        sf = f.getframerate()
        X = numpy.frombuffer(f.readframes(f.getnframes()), numpy.int16).astype(numpy.float32)
    X = sv56(X, sf)
    with wave.Wave_write(ofname) as f:
        f.setparams((1, 2, sf, len(X), "NONE", "not compressed"))
        f.writeframes(X)

def main():
    ifname = sys.argv[1]
    ofname = sys.argv[2]
    tmpdir = './tmp'

    with zipfile.ZipFile(ifname, 'r') as zf:
        fnames = zf.namelist()
        for fname in fnames:
            (tmpdir / pathlib.Path(fname).parent).mkdir(parents=True, exist_ok=True)
            with zf.open(fname) as ifp:
                with open(tmpdir + '/' + fname, 'wb') as ofp:
                    ofp.write(ifp.read())
        wavfnames = [tmpdir + '/' + fname for fname in zf.namelist() if pathlib.Path(fname).suffix == '.wav']
        for wavfname in wavfnames:
            print('norm', wavfname)
            norm_wav(wavfname, wavfname)
    with zipfile.ZipFile(ofname, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in fnames:
            zf.write(tmpdir + '/' + fname, fname)

    print('Done')

if __name__ == '__main__':
    main()
