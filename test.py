import argparse
import PSSGenerator as PSG
import numpy as np
import matplotlib.pyplot as plt

def get_signal_vector(s:float, f:float, x:np.ndarray, do_plot:bool) -> np.ndarray:
    samp_per_symb = int(s * f)
    N = x.size 
    signal_length = samp_per_symb * N

    yr = np.empty(signal_length, dtype=np.float32)
    ones = np.ones((samp_per_symb), dtype=np.float32)
    for i in range(N):
        cur_symb = x[i] * ones 
        yr[i * samp_per_symb : i * samp_per_symb + samp_per_symb] = cur_symb

    # insert quadrature phase component to make complex vector
    yi = np.zeros(signal_length, dtype=np.float32)
    yc = np.empty(2 * signal_length, dtype=np.float32)
    yc[0::2] = yr
    yc[1::2] = 0

    if do_plot:
        plt.plot(range(yr.size), yr)
        plt.show()

    return yc


def write_vector_to_file(file:str, y:np.ndarray):
    y_bytes = bytearray(y)
    immutable_bytes = bytes(y_bytes)

    with open(file, "wb") as bin_file:
        bin_file.write(immutable_bytes)


def read_vector_from_file(file:str) -> np.ndarray:
    with open(file, "rb") as bin_file:
        y = np.fromfile(bin_file, dtype=np.float32)

    return y


def cross_correlation(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    c = np.correlate(a, b, mode='full')
    return c


def test(args):
    d = PSG.get_d_sequence(args.nid)
    x = np.array(d, dtype=np.float32) # convert int to float array
    y = get_signal_vector(args.symbol_duration, args.sample_freq, x, False)
    write_vector_to_file(args.write_file, y)
    # extact only real part of y
    yr = y[0::2]
    z = read_vector_from_file(args.read_file)
    n = 1
    z = z[n*y.size:(n+1)*y.size] # nth repeation of received signal
    # extact only real part of z
    zr = z[0::2]
    # zr = np.roll(zr, -3) # test delay
    cc = cross_correlation(yr, zr)
    delay = cc.argmax() - zr.size + 1
    delay_time = delay * (1/args.sample_freq)
    print(f"The estimated delay is {delay} samples, {delay_time} seconds.")

    if args.plot:
        fig, axs = plt.subplots(3)
        fig.suptitle("Cross-correlation")
        axs[0].plot(range(yr.size), yr)
        axs[1].plot(range(zr.size), zr)
        axs[2].plot(range(cc.size), cc)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for PSS signal generator tool')
    parser.add_argument('--nid',
                        default=0,
                        type=int,
                        choices=[0,1,2],
                        help='Nid(2) of the NidCell')
    parser.add_argument('--symbol_duration',
                        default=3e-6,
                        type=float,
                        help='Symbol duraion in seconds')
    parser.add_argument('--sample_freq',
                        default=10e6,
                        type=float,
                        help='Sampling frequency in Hz')
    parser.add_argument('--write_file',
                        type=str,
                        default='tx_usrp_samples.dat',
                        help='File name to write the signal vector to')
    parser.add_argument('--read_file',
                        type=str,
                        default='rx_usrp_samples.dat',
                        help='File name to read the signal vector from')
    parser.add_argument('--plot',
                        default=False,
                        action='store_true')
    args = parser.parse_args()
    test(args)
