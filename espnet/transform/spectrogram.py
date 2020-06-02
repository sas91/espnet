import librosa
import pudb
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt

def stft(x, n_fft, n_shift, win_length=None, window='hann', center=True, pad_mode='reflect'):
    # x: [Time, Channel]
    if x.ndim == 1:
        single_channel = True
        # x: [Time] -> [Time, Channel]
        x = x[:, None]
    else:
        single_channel = False
    x = x.astype(np.float32)

    # FIXME(kamo): librosa.stft can't use multi-channel?
    # x: [Time, Channel, Freq]
    x = np.stack([librosa.stft(
        x[:, ch],
        n_fft=n_fft,
        hop_length=n_shift,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode).T
        for ch in range(x.shape[1])], axis=1)

    if single_channel:
        # x: [Time, Channel, Freq] -> [Time, Freq]
        x = x[:, 0]
    return x


def stv_af(X, angle1, angle2, radius=0.035, nfreqs=257, sndvelocity=340,
           samplerate=16000, mask=False):
    RR=[]
    sv=[]
    angles = [angle1, angle2]
    mic_pairs = np.array([[0, 3], [1, 4], [2, 5], [0, 1], [2, 3], [4, 5]])
    IPD = []
    for i in range(mic_pairs.shape[0]):
        mic_pair = mic_pairs[i]
        theta = np.angle(X[:, mic_pair[0], :]) - np.angle(X[:, mic_pair[1], :])
        y = np.exp(1j * theta)
        IPD.append(y)
    IPD = np.asarray(IPD)
    IPD = IPD / mic_pairs.shape[0]

    for angle in angles:
        distances =  np.stack((
            radius * np.cos(angle),
            radius * np.cos(angle - np.pi / 3),
            radius * np.cos(angle - 2 * np.pi / 3),
            radius * np.cos(angle - 3 * np.pi / 3),
            radius * np.cos(angle - 4 * np.pi / 3),
            radius * np.cos(angle - 5 * np.pi / 3 )), axis=0)
        deltas = distances / sndvelocity * samplerate
        steervecs = []
        for f in range(nfreqs):
            steervecs.append(np.exp(1j * deltas * np.pi * f / (nfreqs - 1)))
        steervecs = np.stack(steervecs)
        steervecs /= np.sqrt(6)
        sv.append(steervecs)

        distances =  np.stack((
            2*radius * np.cos(angle),             # for ipd [0, 3]
            2*radius * np.cos(angle - np.pi / 3), # for ipd [1, 4]
            2*radius * np.cos(angle - 2* np.pi / 3), # for ipd  [2, 5]
            1*radius * np.cos(angle - 5* np.pi / 3), # for ipd  [0, 1]
            1*radius * np.cos(angle - np.pi / 3), # for ipd  [2, 3]
            1*radius * np.cos(angle - np.pi )), axis=0) # for ipd  [4, 5]
        deltas = distances / sndvelocity * samplerate
        steervecs = []
        for f in range(nfreqs):
            steervecs.append(np.exp(1j * deltas * np.pi * f / (nfreqs - 1)))
        steervecs = np.stack(steervecs)
        steervecs /= np.sqrt(6)
        RR.append(np.abs(np.einsum('CF,CTF->TF', steervecs.T.conj(), IPD)))
        #RR.append(np.log(np.abs(np.einsum('CF,TCF->TF', steervecs.T.conj(),X))))

    sv = np.array(sv)
    svs=sv.transpose(1,2,0)
    IPD=IPD.transpose(1,0,2)
    Af_t = np.array(RR)[0]
    Af_i = np.array(RR)[1]
    Af_t[Af_t < Af_i] = 0
    Af_i[Af_i < Af_t] = 0
    Af_t=np.expand_dims(Af_t, axis=1)
    Af_i=np.expand_dims(Af_i, axis=1)
    if np.random.choice(2):
        Af_full=np.concatenate((IPD.real,IPD.imag,Af_t,Af_i), axis=1)
    else:
        Af_full=np.concatenate((IPD.real,IPD.imag,Af_i,Af_t), axis=1)
    return Af_full, svs


def stv_af_2(X, angle1, angle2, radius=0.035, nfreqs=257, sndvelocity=340,
             samplerate=16000, mask=False, cmplx=True):
    RR=[]
    sv=[]
    angles = [angle1, angle2]
    for angle in angles:
        distances =  np.stack((
            radius * np.cos(angle),
            radius * np.cos(angle - np.pi / 3),
            radius * np.cos(angle - 2 * np.pi / 3),
            radius * np.cos(angle - 3 * np.pi / 3),
            radius * np.cos(angle - 4 * np.pi / 3),
            radius * np.cos(angle - 5 * np.pi / 3 )), axis=0)
        deltas = distances / sndvelocity * samplerate
        steervecs = []
        for f in range(nfreqs):
            steervecs.append(np.exp(1j * deltas * np.pi * f / (nfreqs - 1)))
        steervecs = np.stack(steervecs)
        steervecs /= np.sqrt(6)
        if cmplx:
            RR.append(np.einsum('CF,TCF->TF', steervecs.T.conj(), X))
        else:
            RR.append(np.abs(np.einsum('CF,TCF->TF', steervecs.T.conj(), X)))
        sv.append(steervecs)
    sv = np.array(sv)
    svs=sv.transpose(1,2,0)
    Af = np.array(RR)[0]
    if mask:
        Af[np.abs(Af) < np.abs(RR[1])] = 0
    Af=np.expand_dims(Af, axis=1)
    return Af, svs


def istft(x, n_shift, win_length=None, length=None, window='hann', center=True):
    # x: [Time, Channel, Freq]
    if x.ndim == 2:
        single_channel = True
        # x: [Time, Freq] -> [Time, Channel, Freq]
        x = x[:, None, :]
    else:
        single_channel = False

    # x: [Time, Channel]
    x = np.stack([librosa.istft(
        x[:, ch].T,  # [Time, Freq] -> [Freq, Time]
        hop_length=n_shift,
        win_length=win_length,
        length=length,
        window=window,
        center=center)
        for ch in range(x.shape[1])], axis=1)

    if single_channel:
        # x: [Time, Channel] -> [Time]
        x = x[:, 0]
    return x


def stft2logmelspectrogram(x_stft, fs, n_mels, n_fft, fmin=None, fmax=None,
                           eps=1e-10):
    # x_stft: (Time, Channel, Freq) or (Time, Freq)
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax

    # spc: (Time, Channel, Freq) or (Time, Freq)
    spc = np.abs(x_stft)
    # mel_basis: (Mel_freq, Freq)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    # lmspc: (Time, Channel, Mel_freq) or (Time, Mel_freq)
    lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    return lmspc


def spectrogram(x, n_fft, n_shift, win_length=None, window='hann'):
    # x: (Time, Channel) -> spc: (Time, Channel, Freq)
    spc = np.abs(stft(x, n_fft, n_shift, win_length, window=window))
    return spc


def logmelspectrogram(x, fs, n_mels, n_fft, n_shift,
                      win_length=None, window='hann', fmin=None, fmax=None,
                      eps=1e-10):
    # stft: (Time, Channel, Freq) or (Time, Freq)
    x_stft = stft(x, n_fft=n_fft, n_shift=n_shift, win_length=win_length,
                  window=window)

    return stft2logmelspectrogram(x_stft, fs=fs, n_mels=n_mels, n_fft=n_fft,
                                  fmin=fmin, fmax=fmax, eps=eps)


class Spectrogram(object):
    def __init__(self, n_fft, n_shift, win_length=None, window='hann'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window))

    def __call__(self, x):
        return spectrogram(x,
                           n_fft=self.n_fft, n_shift=self.n_shift,
                           win_length=self.win_length,
                           window=self.window)


class LogMelSpectrogram(object):
    def __init__(self, fs, n_mels, n_fft, n_shift, win_length=None,
                 window='hann', fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ('{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, '
                'n_shift={n_shift}, win_length={win_length}, window={window}, '
                'fmin={fmin}, fmax={fmax}, eps={eps}))'
                .format(name=self.__class__.__name__,
                        fs=self.fs,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        eps=self.eps))

    def __call__(self, x):
        return logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft, n_shift=self.n_shift,
            win_length=self.win_length,
            window=self.window)


class Stft2LogMelSpectrogram(object):
    def __init__(self, fs, n_mels, n_fft, fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ('{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, '
                'fmin={fmin}, fmax={fmax}, eps={eps}))'
                .format(name=self.__class__.__name__,
                        fs=self.fs,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        eps=self.eps))

    def __call__(self, x):
        return stft2logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax)


class Stft(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x):
        return stft(x, self.n_fft, self.n_shift,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode=self.pad_mode)


class StftPIT(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 radius=0.035, nfreqs=257, sndvelocity=340, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.radius = radius
        self.nfreqs = nfreqs
        self.sndvelocity = sndvelocity
        self.samplerate = samplerate

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'radius={radius}, nfreqs={nfreqs},'
                'sndvelocity={sndvelocity}, samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        radius=self.radius,
                        nfreqs=self.nfreqs,
                        sndvelocity=self.sndvelocity,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x, s1=None, s2=None, s3=None, s4=None,
                 m=None, n=None, xv=None, ss=None, ss2=None, ss3=None):
        if s1 is None and m is None:
            #nc=np.max(np.abs(x))
            #x=x/nc
            #x = stft(x, self.n_fft, self.n_shift,
            #         win_length=self.win_length,
            #         window=self.window,
            #         center=self.center,
            #         pad_mode=self.pad_mode)
            #z = {'stft': x, 'nc': nc}
            z = x
        elif s1 is None:
            #if s1.shape[0] == s2.shape[0]:
            #    x = s1+s2+n
            #    s1=s1+s2
            #else:
            #    x = s1+n
            nc=np.max(np.abs(x))
            x=x/nc
            x = stft(x, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)
            m=m/nc
            m = stft(m, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            n=n/nc
            n = stft(n, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)

            p=np.angle(x)
            cp=np.angle(m)
            c=np.abs(m)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            m_psm = c*psm_f
            n_p=np.angle(n)
            n_mag=np.abs(n)
            psm_f=np.cos(p-n_p)
            psm_f[psm_f<0]=0
            n_psm = n_mag*psm_f
            z = {'stft': x, 'm': m_psm, 'n': n_psm, 'nc': nc}
        if ss is not None:
            nc_ss=np.max(np.abs(ss))
            ss = ss/nc_ss
            ss = stft(ss, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            if ss2 is not None:
                nc_ss=np.max(np.abs(ss2))
                ss2 = ss2/nc_ss
                ss2 = stft(ss2, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
            if ss3 is not None:
                nc_ss=np.max(np.abs(ss3))
                ss3 = ss3/nc_ss
                ss3 = stft(ss3, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
                ss=np.concatenate((ss,ss2,ss3), axis=0)
                ss=np.abs(ss)
            nc=np.max(np.abs(x))
            x=x/nc
            x = stft(x, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)
            s1=s1/nc
            s1 = stft(s1, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            n=n/nc
            n = stft(n, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)

            p=np.angle(x)
            cp=np.angle(s1)
            c=np.abs(s1)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s1_psm = c*psm_f
            n_p=np.angle(n)
            n_mag=np.abs(n)
            psm_f=np.cos(p-n_p)
            psm_f[psm_f<0]=0
            n_psm = n_mag*psm_f
            z = {'stft': x, 's1': s1_psm, 'n': n_psm, 'ss':ss, 'nc': nc}
        elif xv is not None:
            nc=np.max(np.abs(x))
            x=x/nc
            x = stft(x, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)
            s1=s1/nc
            s1 = stft(s1, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            n=n/nc
            n = stft(n, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)

            p=np.angle(x)
            cp=np.angle(s1)
            c=np.abs(s1)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s1_psm = c*psm_f
            n_p=np.angle(n)
            n_mag=np.abs(n)
            psm_f=np.cos(p-n_p)
            psm_f[psm_f<0]=0
            n_psm = n_mag*psm_f
            z = {'stft': x, 's1': s1_psm, 'n': n_psm, 'xv':xv, 'nc': nc}
        else:
            nc=np.max(np.abs(x))
            x=x/nc
            x = stft(x, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)
            x_abs = np.abs(x)
            if s1.shape[0] == s3.shape[0]:
                s3=s3/nc
                s3 = stft(s3, self.n_fft, self.n_shift,
                          win_length=self.win_length,
                          window=self.window,
                          center=self.center,
                          pad_mode=self.pad_mode)
            else:
                s3 = np.zeros((x.shape[0], x.shape[1]))
            if s1.shape[0] == s4.shape[0]:
                s4=s4/nc
                s4 = stft(s4, self.n_fft, self.n_shift,
                          win_length=self.win_length,
                          window=self.window,
                          center=self.center,
                          pad_mode=self.pad_mode)
            else:
                s4 = np.zeros((x.shape[0], x.shape[1]))
            s1=s1/nc
            s1 = stft(s1, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            s2=s2/nc
            s2 = stft(s2, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            n=n/nc
            n = stft(n, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)
            p=np.angle(x)
            cp=np.angle(s1)
            c=np.abs(s1)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s1_psm = c*psm_f
            try:
                m=c/(x_abs-c)
            except:
                m=c/(x_abs-c+1e-20)
            m[m>1]=1
            m[m<1]=0
            m1=m

            cp=np.angle(s2)
            c=np.abs(s2)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s2_psm = c*psm_f
            try:
                m=c/(x_abs-c)
            except:
                m=c/(x_abs-c+1e-20)
            m[m>1]=1
            m[m<1]=0
            m2=m

            cp=np.angle(s3)
            c=np.abs(s3)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s3_psm = c*psm_f
            try:
                m=c/(x_abs-c)
            except:
                m=c/(x_abs-c+1e-20)
            m[m>1]=1
            m[m<1]=0
            m3=m

            cp=np.angle(s4)
            c=np.abs(s4)
            psm_f=np.cos(p-cp)
            psm_f[psm_f<0]=0
            s4_psm = c*psm_f
            try:
                m=c/(x_abs-c)
            except:
                m=c/(x_abs-c+1e-20)
            m[m>1]=1
            m[m<1]=0
            m4=m

            n_p=np.angle(n)
            n_mag=np.abs(n)
            psm_f=np.cos(p-n_p)
            psm_f[psm_f<0]=0
            n_psm = n_mag*psm_f
            try:
                mn=n_mag/(x_abs-n_mag)
            except:
                mn=n_mag/(x_abs-n_mag+1e-20)
            mn[mn>1]=1
            mn[mn<1]=0

            z = {'stft': x, 's1': s1_psm, 's2': s2_psm, 'nc': nc,
                 's3': s3_psm, 's4': s4_psm, 'n': n_psm,
                 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'mn': mn}
        return z


class MagAf(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 radius=0.035, nfreqs=257, sndvelocity=340, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect', mask=False,
                 cmplx=True):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.radius = radius
        self.nfreqs = nfreqs
        self.sndvelocity = sndvelocity
        self.samplerate = samplerate
        self.mask = mask
        self.cmplx = cmplx

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'radius={radius}, nfreqs={nfreqs},'
                'sndvelocity={sndvelocity}, samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode},'
                'mask={mask}, cmplx={cmplx})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        radius=self.radius,
                        nfreqs=self.nfreqs,
                        sndvelocity=self.sndvelocity,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode,
                        mask=self.mask,
                        cmplx=self.cmplx))

    def __call__(self, x, angle1=None, angle2=None, xv=None, c=None, ss=None,
                 ss2=None, ss3=None):
        #mean_x=np.mean(x, axis=0)
        #nc=np.max(np.abs(x), axis=0)
        nc=np.max(np.abs(x))
        #nc=np.ones(6)
        #x=x-mean_x
        x=x/nc
        x = stft(x, self.n_fft, self.n_shift,
                 win_length=self.win_length,
                 window=self.window,
                 center=self.center,
                 pad_mode=self.pad_mode)

        if c is not None:
            c = c/nc
            c = stft(c, self.n_fft, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center,
                     pad_mode=self.pad_mode)

        p=np.angle(x[:,0,:])
        x1=x[:,0,:]
        n=x1-c
        cp=np.angle(c)
        xm=np.abs(x1)
        c=np.abs(c)
        n=np.abs(n)
        psm_f=np.cos(p-cp)
        psm_f[psm_f<0]=0
        c_psm = c*psm_f
        try:
            m=c/n
        except:
            m=c/(n+1e-20)
        m[m>1]=1
        m[m<1]=0
        is_mc_data=False

        if angle1 and angle2:
            angle1 = (int(angle1)/180) * np.pi
            angle2 = (int(angle2)/180) * np.pi

            y, sv = stv_af(x, angle1, angle2,
                       radius=self.radius,
                       nfreqs=self.nfreqs,
                       sndvelocity=self.sndvelocity,
                       samplerate=self.samplerate,
                       mask=self.mask)
            if ss is None:
                z = {'stft': x, 'af': y, 'sv': sv, 'c': c_psm, 'm': m, 'p': p, 'nc': nc, 'c_orig': c}
            is_mc_data=True
        if xv is not None:
            z = {'stft': x, 'xv': xv, 'c': c_psm, 'm': m, 'p': p, 'nc': nc, 'c_orig': c}
            is_mc_data=True
        if ss is not None:
            nc_ss=np.max(np.abs(ss))
            ss = ss/nc_ss
            ss = stft(ss, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            if ss2 is not None:
                nc_ss=np.max(np.abs(ss2))
                ss2 = ss2/nc_ss
                ss2 = stft(ss2, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
            if ss3 is not None:
                nc_ss=np.max(np.abs(ss3))
                ss3 = ss3/nc_ss
                ss3 = stft(ss3, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
                ss=np.concatenate((ss,ss2,ss3), axis=0)
                ss=np.abs(ss)
            if angle1 is None:
                z = {'stft': x, 'c': c_psm, 'm': m, 'p': p, 'nc': nc, 'c_orig': c, 'ss': ss}
            else:
                z = {'stft': x, 'af': y, 'sv': sv, 'c': c_psm, 'm': m, 'p': p,
                     'nc': nc, 'c_orig': c, 'ss': ss}
            is_mc_data=True
        if not is_mc_data:
            z=x
        return z


class StftAf(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 radius=0.035, nfreqs=257, sndvelocity=340, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect', mask=False,
                 cmplx=True):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.radius = radius
        self.nfreqs = nfreqs
        self.sndvelocity = sndvelocity
        self.samplerate = samplerate
        self.mask = mask
        self.cmplx = cmplx

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'radius={radius}, nfreqs={nfreqs},'
                'sndvelocity={sndvelocity}, samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode},'
                'mask={mask}, cmplx={cmplx})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        radius=self.radius,
                        nfreqs=self.nfreqs,
                        sndvelocity=self.sndvelocity,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode,
                        mask=self.mask,
                        cmplx=self.cmplx))

    def __call__(self, x, angle1=None, angle2=None, xv=None, ss=None, ss2=None,
                 ss3=None):
        #nc=np.max(np.abs(x), axis=0)
        nc=np.max(np.abs(x))
        #nc=np.ones(6)
        x=x/nc
        is_mc_data=False
        x = stft(x, self.n_fft, self.n_shift,
                 win_length=self.win_length,
                 window=self.window,
                 center=self.center,
                 pad_mode=self.pad_mode)

        if angle1 and angle2:
            angle1 = (int(angle1)/180) * np.pi
            angle2 = (int(angle2)/180) * np.pi

            y, sv = stv_af(x, angle1, angle2,
                       radius=self.radius,
                       nfreqs=self.nfreqs,
                       sndvelocity=self.sndvelocity,
                       samplerate=self.samplerate,
                       mask=self.mask)
            if ss is None:
                z = {'stft': x, 'af': y, 'sv': sv, 'nc': nc}
            is_mc_data=True
        if xv is not None:
            z = {'stft': x, 'xv': xv, 'nc': nc}
            is_mc_data=True
        if ss is not None:
            nc_ss=np.max(np.abs(ss))
            ss = ss/nc_ss
            ss = stft(ss, self.n_fft, self.n_shift,
                      win_length=self.win_length,
                      window=self.window,
                      center=self.center,
                      pad_mode=self.pad_mode)
            if ss2 is not None:
                nc_ss=np.max(np.abs(ss2))
                ss2 = ss2/nc_ss
                ss2 = stft(ss2, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
            if ss3 is not None:
                nc_ss=np.max(np.abs(ss3))
                ss3 = ss3/nc_ss
                ss3 = stft(ss3, self.n_fft, self.n_shift,
                           win_length=self.win_length,
                           window=self.window,
                           center=self.center,
                           pad_mode=self.pad_mode)
                ss=np.concatenate((ss,ss2,ss3), axis=0)
            ss=np.abs(ss)
            if angle1 is None:
                z = {'stft': x, 'ss': ss, 'nc': nc}
            else:
                z = {'stft': x, 'af': y, 'sv': sv, 'ss': ss, 'nc': nc}
            is_mc_data=True
        if not is_mc_data:
            z=x
        return z


class IStft(object):
    def __init__(self, n_shift, win_length=None, window='hann', center=True):
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center

    def __repr__(self):
        return ('{name}(n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'center={center})'
                .format(name=self.__class__.__name__,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        center=self.center))

    def __call__(self, x, length=None):
        return istft(x, self.n_shift,
                     win_length=self.win_length,
                     length=length,
                     window=self.window,
                     center=self.center)


class Phase(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 nfreqs=257, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.nfreqs = nfreqs
        self.samplerate = samplerate

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'nfreqs={nfreqs},'
                'samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        nfreqs=self.nfreqs,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x, angle1, angle2, print_image=True):
        nc=np.max(np.abs(x))
        x=x/nc
        x = stft(x, self.n_fft, self.n_shift,
                 win_length=self.win_length,
                 window=self.window,
                 center=self.center,
                 pad_mode=self.pad_mode)
        y=np.transpose(x, (1, 2, 0))
        p = np.angle(x)
        angle1 = angle1
        angle2 = angle2
        z = {'stft': x, 'p': p, 'angle1': angle1, 'angle2': angle2}
        return z


class PhaseDOA(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 nfreqs=257, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.nfreqs = nfreqs
        self.samplerate = samplerate

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'nfreqs={nfreqs},'
                'samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        nfreqs=self.nfreqs,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x, angle1, angle2, print_image=False):
        nc=np.max(np.abs(x))
        x=x/nc
        x = stft(x, self.n_fft, self.n_shift,
                 win_length=self.win_length,
                 window=self.window,
                 center=self.center,
                 pad_mode=self.pad_mode)
        y=np.transpose(x, (1, 2, 0))
        room_dim = np.r_[10.,10.]
        array = pra.circular_2D_array(center=room_dim/2, M=6, phi0=0,
                                      radius=35.0e-3)
        doa_music = pra.doa.algorithms['MUSIC'](array, 16000, 512, c=343, num_src=2)
        doa_srp = pra.doa.algorithms['SRP'](array, 16000, 512, c=343, num_src=2)
        doa_tops = pra.doa.algorithms['TOPS'](array, 16000, 512, c=343, num_src=2)
        doa_waves = pra.doa.algorithms['WAVES'](array, 16000, 512, c=343, num_src=2)
        doa_frida = pra.doa.algorithms['FRIDA'](array, 16000, 512, c=343,
                                                num_src=2, max_four=4)
        freq_range = [300, 3500]
        doa_music.locate_sources(y, freq_range=freq_range)
        doa_srp.locate_sources(y, freq_range=freq_range)
        doa_tops.locate_sources(y, freq_range=freq_range)
        doa_waves.locate_sources(y, freq_range=freq_range)
        doa_frida.locate_sources(y, freq_range=freq_range)

        spatial_resp = doa_music.grid.values
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        if max_val - min_val > 0.0:
            spatial_resp_music = (spatial_resp - min_val) / (max_val - min_val)
        else:
            spatial_resp_music = spatial_resp
        peaks_music = doa_music.src_idx
        if peaks_music.shape[0] < 2:
            peaks_music = pra.doa.detect_peaks(spatial_resp_music)
        if peaks_music.shape[0] == 0:
            temp = peaks_music
            peaks_music = np.insert(temp, 0, 0)
        if peaks_music.shape[0] == 1:
            temp = peaks_music
            peaks_music = np.insert(temp, 1, temp[0])

        spatial_resp = doa_srp.grid.values
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        if max_val - min_val > 0.0:
            spatial_resp_srp = (spatial_resp - min_val) / (max_val - min_val)
        else:
            spatial_resp_srp = spatial_resp
        peaks_srp = doa_srp.src_idx
        if peaks_srp.shape[0] < 2:
            peaks_srp = pra.doa.detect_peaks(spatial_resp_srp)
        if peaks_srp.shape[0] == 0:
            temp = peaks_srp
            peaks_srp = np.insert(temp, 0, 0)
        if peaks_srp.shape[0] == 1:
            temp = peaks_srp
            peaks_srp = np.insert(temp, 1, temp[0])

        spatial_resp = doa_tops.grid.values
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        if max_val - min_val > 0.0:
            spatial_resp_tops = (spatial_resp - min_val) / (max_val - min_val)
        else:
            spatial_resp_tops = spatial_resp
        peaks_tops = doa_tops.src_idx
        if peaks_tops.shape[0] < 2:
            peaks_tops = pra.doa.detect_peaks(spatial_resp_tops)
        if peaks_tops.shape[0] == 0:
            temp = peaks_tops
            peaks_tops = np.insert(temp, 0, 0)
        if peaks_tops.shape[0] == 1:
            temp = peaks_tops
            peaks_tops = np.insert(temp, 1, temp[0])

        spatial_resp = doa_waves.grid.values
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        if max_val - min_val > 0.0:
            spatial_resp_waves = (spatial_resp - min_val) / (max_val - min_val)
        else:
            spatial_resp_waves = spatial_resp
        peaks_waves = doa_waves.src_idx
        if peaks_waves.shape[0] < 2:
            peaks_waves = pra.doa.detect_peaks(spatial_resp_waves)
        if peaks_waves.shape[0] == 0:
            temp = peaks_waves
            peaks_waves = np.insert(temp, 0, 0)
        if peaks_waves.shape[0] == 1:
            temp = peaks_waves
            peaks_waves = np.insert(temp, 1, temp[0])

        spatial_resp = np.abs(doa_frida._gen_dirty_img())
        min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        if max_val - min_val > 0.0:
            spatial_resp_frida = (spatial_resp - min_val) / (max_val - min_val)
        else:
            spatial_resp_frida = spatial_resp
        peaks_frida = pra.doa.detect_peaks(spatial_resp_frida)
        if peaks_frida.shape[0] == 0:
            temp = peaks_frida
            peaks_frida = np.insert(temp, 0, 0)
        if peaks_frida.shape[0] == 1:
            temp = peaks_frida
            peaks_frida = np.insert(temp, 1, temp[0])

        if print_image:
            base = 1.
            height = 10.
            true_col = [0, 0, 0]
            phi_plt = doa_music.grid.azimuth

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            c_phi_plt = np.r_[phi_plt, phi_plt[0]]
            c_dirty_img_music = np.r_[spatial_resp_music, spatial_resp_music[0]]
            c_dirty_img_srp = np.r_[spatial_resp_srp, spatial_resp_srp[0]]
            c_dirty_img_tops = np.r_[spatial_resp_tops, spatial_resp_tops[0]]
            c_dirty_img_waves = np.r_[spatial_resp_waves, spatial_resp_waves[0]]
            c_dirty_img_frida = np.r_[spatial_resp_frida, spatial_resp_frida[0]]
            ax.plot(c_phi_plt, base + height * c_dirty_img_music, linewidth=2,
                    alpha=0.55, color='red', linestyle='-',
                    label="MUSIC")
            ax.plot(c_phi_plt, base + height * c_dirty_img_srp, linewidth=2,
                    alpha=0.55, color='green', linestyle='-',
                    label="SRP")
            ax.plot(c_phi_plt, base + height * c_dirty_img_tops, linewidth=2,
                    alpha=0.55, color='yellow', linestyle='-',
                    label="TOPS")
            ax.plot(c_phi_plt, base + height * c_dirty_img_waves, linewidth=2,
                    alpha=0.55, color='blue', linestyle='-',
                    label="WAVES")
            ax.plot(c_phi_plt, base + height * c_dirty_img_frida, linewidth=2,
                    alpha=0.55, color='brown', linestyle='-',
                    label="FRIDA")
            # plot true loc
            azimuth = np.array([angle1, angle2]) / 180. * np.pi
            prediction_music = np.array([peaks_music[0], peaks_music[1]]) / 180. * np.pi
            prediction_srp = np.array([peaks_srp[0], peaks_srp[1]]) / 180. * np.pi
            prediction_tops = np.array([peaks_tops[0], peaks_tops[1]]) / 180. * np.pi
            prediction_waves = np.array([peaks_waves[0], peaks_waves[1]]) / 180. * np.pi
            prediction_frida = np.array([peaks_frida[0], peaks_frida[1]]) / 180. * np.pi
            for angle in azimuth:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color=true_col, alpha=0.6)
            for angle in prediction_music:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color='red', alpha=0.6)
            for angle in prediction_srp:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color='green', alpha=0.6)
            for angle in prediction_tops:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color='yellow', alpha=0.6)
            for angle in prediction_waves:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color='blue', alpha=0.6)
            for angle in prediction_frida:
                ax.plot([angle, angle], [base, base + height], linewidth=2, linestyle='--',
                    color='brown', alpha=0.6)
            K = len(azimuth)
            ax.scatter(azimuth, base + height*np.ones(K), c=np.tile(true_col,
                       (K, 1)), alpha=0.95, marker='*',
                       linewidths=0, s=70,
                       label='true locations')
            ax.scatter(prediction_music, base + height*np.ones(K), c='red',
                       alpha=0.75, marker='D',
                       linewidths=0, s=30)
            ax.scatter(prediction_srp, base + height*np.ones(K), c='green',
                       alpha=0.75, marker='h',
                       linewidths=0, s=30)
            ax.scatter(prediction_tops, base + height*np.ones(K), c='yellow',
                       alpha=0.75, marker='X',
                       linewidths=0, s=30)
            ax.scatter(prediction_waves, base + height*np.ones(K), c='blue',
                       alpha=0.75, marker='o',
                       linewidths=0, s=30)
            ax.scatter(prediction_frida, base + height*np.ones(K), c='brown',
                       alpha=0.75, marker='v',
                       linewidths=0, s=30)

            plt.legend()
            #handles, labels = ax.get_legend_handles_labels()
            #ax.legend(handles, labels, framealpha=0.5,
            #          scatterpoints=1, loc='lower right', fontsize=10,
            #          ncol=1, bbox_to_anchor=(1.4, 0.0),
            #          handletextpad=.2, columnspacing=1.7,
            #          labelspacing=0.1)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      shadow=True, ncol=3)
            ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
            ax.xaxis.set_label_coords(0.5, -0.11)
            ax.set_yticks(np.linspace(0, 1, 2))
            ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
            ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
            plt.tight_layout()
            #ax.set_ylim([0, 1.05 * (base + height)]);
            plt.savefig('DOA.png')

        p = np.angle(x)
        angle1 = angle1
        angle2 = angle2
        z = {'stft': x, 'p': p, 'angle1': angle1, 'angle2': angle2,
             'sf_music': spatial_resp_music, 'sf_srp': spatial_resp_srp,
             'sf_tops': spatial_resp_tops, 'sf_waves': spatial_resp_waves,
             'sf_frida': spatial_resp_frida,
             'mhyp1': peaks_music[0], 'mhyp2': peaks_music[1],
             'fhyp1': peaks_frida[0], 'fhyp2': peaks_frida[1],
             'thyp1': peaks_tops[0], 'thyp2': peaks_tops[1],
             'whyp1': peaks_waves[0], 'whyp2': peaks_waves[1],
             'shyp1': peaks_srp[0], 'shyp2': peaks_srp[1]}
        return z


class IPD(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 nfreqs=257, samplerate=16000,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.nfreqs = nfreqs
        self.samplerate = samplerate

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'nfreqs={nfreqs},'
                'samplerate={samplerate},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        nfreqs=self.nfreqs,
                        samplerate=self.samplerate,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x, angle1, angle2):
        nc=np.max(np.abs(x))
        x=x/nc
        x = stft(x, self.n_fft, self.n_shift,
                 win_length=self.win_length,
                 window=self.window,
                 center=self.center,
                 pad_mode=self.pad_mode)

        mic_pairs = np.array([[0, 3], [1, 4], [2, 5], [0, 1], [2, 3], [4, 5]])
        IPD = []
        for i in range(mic_pairs.shape[0]):
            mic_pair = mic_pairs[i]
            theta = np.angle(x[:, mic_pair[0], :]) - np.angle(x[:, mic_pair[1], :])
            y = np.exp(1j * theta)
            IPD.append(y)
        IPD = np.asarray(IPD)
        IPD = IPD / mic_pairs.shape[0]
        IPD=IPD.transpose(1,0,2)
        IPD_r=np.concatenate((IPD.real,IPD.imag), axis=2)

        angle1 = angle1
        angle2 = angle2
        z = {'p': IPD_r, 'angle1': angle1, 'angle2': angle2}
        return z
