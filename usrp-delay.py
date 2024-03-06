import os
import argparse
import PSSGenerator as PSG
import numpy as np
import matplotlib.pyplot as plt
import termplotlib as tplt

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


def send_and_receive_signal(args):
    command = ['sudo', 'build/trx_timed_samples'] 
    if args.usrp_type == 'x300':
        command += ['--subdev', 'A:0']
    else:
        command += ['--subdev', 'A:A']
    command += ['--rate', f'{args.sample_freq}']
    command += ['--freq', f'{args.center_freq}']
    command += ['--bw', '20e6']
    command += ['--args', f'"type={args.usrp_type}"']
    command += ['--first-sample-time', '0.1']
    command += ['--rx-gain', f'{args.rx_gain}']
    command += ['--tx-gain', f'{args.tx_gain}']
    print("Sending and receiving samples")
    command_to_run = f"""{' '.join(command)}"""
    return os.system(command_to_run)

def test(args):
    d = PSG.get_d_sequence(args.nid)
    x = np.array(d, dtype=np.float32) # convert int to float array
    y = get_signal_vector(args.symbol_duration, args.sample_freq, x, False)
    write_vector_to_file(args.write_file, y)
    # extact only real part of y
    if send_and_receive_signal(args):
        print("Error in sending signal")
        return -1
    else:
        print("Success in sending and receiving samples")
    yr = y[0::2]
    zin = read_vector_from_file(args.read_file)
    n = args.repeat - 1
    delays = np.zeros(args.repeat - 2, dtype=int)
    ii = 0
    for i in range(1,n):
        z = zin[i*y.size:(i+1)*y.size] # nth repetition of received signal
        # extact only real part of z
        zr = z[0::2]
        cc = cross_correlation(zr, yr)
        ccmax = abs(cc)
        delays[ii] = ccmax.argmax() - zr.size + 1
        ii += 1
    # get mean from all repetitions
    delay = np.mean(delays)
    delay_time = delay * (1/args.sample_freq)
    delays

    cc_x_plot = [yr.size - x for x in range(cc.size)]
    if args.plot:
        fig, axs = plt.subplots(3)
        fig.suptitle("Cross-correlation")
        axs[0].plot(range(yr.size), yr)
        axs[1].plot(range(zin[n*y.size:(n+1)*y.size:2].size), zin[n*y.size:(n+1)*y.size:2])
        axs[2].plot(cc_x_plot, cc)
        plt.show()
    else:
        print("Tx signal")
        fig1 = tplt.figure()
        fig1.plot(range(yr.size), yr)
        fig1.show()
        print()
        print("Rx signal")
        fig2 = tplt.figure()
        fig2.plot(range(zin[n*y.size:(n+1)*y.size:2].size), zin[n*y.size:(n+1)*y.size:2])
        fig2.show()
        print()
        print("Delay")
        fig3 = tplt.figure()
        fig3.plot(cc_x_plot, cc, label='Delay')
        fig3.show()

    print()
    print(f"The estimated delay is {delay} samples, {delay_time} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for delay estimation tool')
    parser.add_argument('--nid',
                        default=0,
                        type=int,
                        choices=[0,1,2],
                        help='Nid(2) PSS signal')
    parser.add_argument('--symbol_duration',
                        default=3e-6,
                        type=float,
                        help='Symbol duraion in seconds')
    parser.add_argument('--sample_freq',
                        default=10e6,
                        type=float,
                        help='Sampling frequency in Hz')
    parser.add_argument('--center_freq',
                        type=float,
                        default=3.6e9,
                        help='Center frequency of carrier signal')
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
    parser.add_argument('--repeat',
                        type=int,
                        default=10,
                        help='Number of times the signal vector repeats')
    parser.add_argument('--usrp_type',
                        type=str,
                        default='x300',
                        help='Type of USRP')
    parser.add_argument('--rx_gain',
                        type=float,
                        default=0,
                        help='USRP Rx gain')
    parser.add_argument('--tx_gain',
                        type=float,
                        default=0,
                        help='USRP Tx gain')
    args = parser.parse_args()
    test(args)
