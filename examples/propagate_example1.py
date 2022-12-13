import matplotlib.pyplot as plt

from pybeam import *

if __name__ == '__main__':

    b = TophatBeam(num=512, radius=1.6e-3, width=10e-3)
    mask = CircularMask(b, radius=1.5e-3)
    b2 = mask.apply(b)
    f = 500e-3
    propergator.info(b2, f, 1.5e-3)
    b3 = propergator.propIR(b2, f)
    lens = PhaseLens(b3, focal_length=f)
    b4 = lens.apply(b3)
    delta_z = (1.5 - 0.0001) / 11
    z = 0.001
    plt.figure(1)
    for i in range(1, 11):
        b4 = propergator.propTF(b4, delta_z);

        plt.subplot(2, 5, i)
        s = 'z_distance= %3.1f m' % z
        plt.title(s)

        plt.imshow(b4.intensity, cmap='jet');
        plt.axis('off')

        z = z + delta_z
    plt.show()

    # test FF
    t = SquareBeam(offy=0, offx=0, size=0.022, width=0.5, num=250)
    z = 2000
    propergator.info(t, z, t.size)
    t2 = propergator.propFF(t, z)
    fig, ax = displayInt(t2, num=2, title='propFF', nthroot=3, cross_section=True)

    # show  analytic version
    x = np.linspace(-t.width / 2, t.width / 2, t.num)
    I2 = (4 * t.size ** 2 / (t.wavelength * z)) ** 2 * np.sinc(2 * t.size / (t.wavelength * z) * x) ** 2
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, I2)

    b = Beam(amplitude=np.ones((3, 3)), phase=np.ones((3, 3)) * np.pi)
    t = SquareBeam(offy=0, offx=0, size=0.051, width=0.5)
    propergator.info(t, 1000, 00.051)

    t2 = propergator.propIR(t, 2000)
    t3 = propergator.propTF(t, 2000)
    displayInt(t2, cross_section=True, num=1, title='propIR')
    display(t2, num=2, title='propIR')

    displayInt(t3, cross_section=True, num=3, title='propTF')
    display(t3, num=4, title='propIR')

    fig, axs = plt.subplots(num=10, nrows=4, ncols=2)

    zlist = [1000, 2000, 4000, 20000]
    for ind, z in enumerate(zlist):
        tir = propergator.propIR(t, z)
        ttr = propergator.propTF(t, z)
        bi = tir.intensity
        r = bi.shape[0] // 2
        axs[ind, 0].plot(np.linspace(-tir.width / 2, tir.width / 2, bi.shape[0]), bi[r, :])
        bi = ttr.intensity
        r = bi.shape[0] // 2
        axs[ind, 1].plot(np.linspace(-tir.width / 2, tir.width / 2, bi.shape[0]), bi[r, :])
        axs[ind, 1].set_title('z_distance={} TF'.format(z))
        axs[ind, 0].set_title('z_distance={} IR'.format(z))


    plt.show()