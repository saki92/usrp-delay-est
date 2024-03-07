import os
import argparse
import PSSGenerator as PSG
import numpy as np
import matplotlib.pyplot as plt
import termplotlib as tplt
import uhd
import benchmark_rate as br

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
    yc = yr.astype(np.csingle)

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


def tx_usrp_samples(usrp, tx_streamer, tx_vector:np.ndarray, tx_time:uhd.types.TimeSpec, repeat_times:int, result):
    metadata = uhd.types.TXMetadata()
    metadata.time_spec = tx_time 
    rate = usrp.get_tx_rate()
    start_time_ticks = tx_time.to_ticks(rate)
    metadata.end_of_burst = False 
    metadata.has_time_spec = True
    print(f"First tx sample timestamp {metadata.time_spec.to_ticks(rate)}")

    repeat_cnt = 0
    while repeat_cnt < repeat_times:
        num_tx_samps = tx_streamer.send(tx_vector, metadata)
        if num_tx_samps != tx_vector.size:
            logger.error("Didn't send all samples in buffer")
            result.append(0)
            return
        repeat_cnt += 1
        new_time = uhd.types.TimeSpec.from_ticks(start_time_ticks + repeat_cnt * tx_vector.size, rate)
        metadata.time_spec = new_time

    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((1, 0), dtype=np.csingle), metadata)

    result.append(1)

def rx_usrp_samples(usrp, rx_streamer, rx_stream_time, recv_buffer:np.ndarray, result):
    # Craft and send the Stream Command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = recv_buffer.size
    stream_cmd.stream_now = False
    stream_cmd.time_spec = rx_stream_time
    rate = usrp.get_rx_rate()
    print(f"First rx set timestamp {rx_stream_time.to_ticks(rate)}")
    rx_streamer.issue_stream_cmd(stream_cmd)

    metadata = uhd.types.RXMetadata()
    num_rx_samples = rx_streamer.recv(recv_buffer, metadata, 5.0)

    if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
        logger.error("Overflow occured")
    elif metadata.error_code == uhd.types.RXMetadataErrorCode.late:
        logger.error("Late packet occured")
    elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
        logger.error("Timeout occured")

    if num_rx_samples != recv_buffer.size:
        logger.error("%d samples received out of %d", num_rx_samples, recv_buffer.size)
        return False
    
    first_rx_ts = metadata.time_spec.to_ticks(rate)
    print(f"First rx sample timestamp {first_rx_ts}")
    
    result.append(1)
    result.append(first_rx_ts)

def setup_usrp(args):
    usrp = uhd.usrp.MultiUSRP(args.usrp_args)

    if args.subdev:
        usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(args.subdev))
        usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(args.subdev))
    else:
        logger.error("Subdev required")
        return False

    # Set the reference clock
    if args.ref and not br.setup_ref(usrp, args.ref, usrp.get_num_mboards()):
        return False

    usrp.set_master_clock_rate(args.sample_freq)
    logger.info("Master clock is set to %f", usrp.get_master_clock_rate())
    usrp.set_rx_rate(args.sample_freq)
    usrp.set_tx_rate(args.sample_freq)
    return usrp

def send_and_receive_signal(usrp, args, tx_vector:np.ndarray, rx_vector:np.ndarray, rx_time:uhd.types.TimeSpec):
    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    threads = []
    quit_event = br.threading.Event()
    bw = 1 / args.symbol_duration + 1e6
    usrp.set_rx_bandwidth(bw)
    usrp.set_tx_bandwidth(bw)
    st_args = uhd.usrp.StreamArgs('fc32', 'sc16')
    st_args.channels = [0]
    st_args.args = uhd.types.DeviceAddr("")

    rate = usrp.get_tx_rate()
    logger.info("Set sampling rate is %f", rate)
    rx_time_ticks = rx_time.to_ticks(rate) - 1000 # start stream sooner
    rx_stream_time = uhd.types.TimeSpec.from_ticks(rx_time_ticks, rate)
    rx_streamer = usrp.get_rx_stream(st_args)
    rx_results = []
    rx_thread = br.threading.Thread(target=rx_usrp_samples,
                                    args=(usrp, rx_streamer, rx_stream_time, rx_vector, rx_results))
    threads.append(rx_thread)
    rx_thread.start()

    tx_streamer = usrp.get_tx_stream(st_args)
    tx_results = []
    tx_thread = br.threading.Thread(target=tx_usrp_samples,
                                    args=(usrp, tx_streamer, tx_vector, rx_time, args.repeat, tx_results))
    threads.append(tx_thread)
    tx_thread.start()

    quit_event.set()
    for thr in threads:
        thr.join()

    return [rx_results, tx_results]

def send_and_receive_signal_single(args):
    bw = 1 / args.symbol_duration + 1e6
    command = ['sudo', 'build/trx_timed_samples'] 
    if args.usrp_type == 'x300':
        command += ['--subdev', 'A:0']
    else:
        command += ['--subdev', 'A:A']
    command += ['--rate', f'{args.sample_freq}']
    command += ['--freq', f'{args.center_freq}']
    command += ['--bw', f'{bw}']
    command += ['--args', f'"type={args.usrp_type}"']
    command += ['--first-sample-time', '0.1']
    command += ['--rx-gain', f'{args.rx_gain}']
    command += ['--tx-gain', f'{args.tx_gain}']
    command += ['--spb', f'{args.tx_samples}']
    command += ['--repeat', f'{args.repeat}']
    print("Sending and receiving samples")
    command_to_run = f"""{' '.join(command)}"""
    return os.system(command_to_run)

def test(args):
    usrp = setup_usrp(args)
    rate = usrp.get_tx_rate()

    d = PSG.get_d_sequence(args.nid)
    x = np.array(d, dtype=np.float32) # convert int to float array
    y = get_signal_vector(args.symbol_duration, rate, x, False)
    write_vector_to_file(args.write_file, y)
    args.tx_samples = y.size

    recv_buff_size = y.size * args.repeat + 2000
    zin = np.empty((1, recv_buff_size), dtype=np.csingle)
    current_time = usrp.get_time_now()
    tx_time = uhd.types.TimeSpec(current_time.get_real_secs() + args.first_samp_time)
    [rx_results, tx_results] = send_and_receive_signal(usrp, args, y, zin, tx_time)
    if not rx_results[0] and not tx_results[0]:
        print("Error in sending signal")
        return -1
    else:
        print("Success in sending and receiving samples")
    actual_rx_ts = rx_results[1]
    tx_time_tx = tx_time.to_ticks(rate)
    buff_offset = tx_time_tx - actual_rx_ts
    print(f"Buffer offset is {buff_offset}")
    if buff_offset < 0:
        logger.error("Streaming started too late")
        return False
    yr = y.real
    #zin = read_vector_from_file(args.read_file)
    n = args.repeat - 1
    delays = np.zeros(args.repeat - 2, dtype=int)
    ii = 0
    zin = np.transpose(zin[0, buff_offset:buff_offset + y.size * args.repeat])
    for i in range(1,n):
        z = zin[i*y.size:(i+1)*y.size] # nth repetition of received signal
        # extact only real part of z
        #zr = z[0::2]
        zr = z.real
        cc = cross_correlation(zr, yr)
        ccmax = abs(cc)
        delays[ii] = ccmax.argmax() - zr.size + 1
        ii += 1
    # get mean from all repetitions
    delay = np.mean(delays)
    delay_time = delay * (1/rate)
    delays

    cc_x_plot = [yr.size - x for x in range(cc.size)]
    if args.plot:
        fig, axs = plt.subplots(3)
        fig.suptitle("Cross-correlation")
        axs[0].plot(range(yr.size), yr)
        axs[1].plot(range(zin[n*y.size:(n+1)*y.size:2].size), zin[n*y.size:(n+1)*y.size].real)
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
        fig2.plot(range(zin[n*y.size:(n+1)*y.size:2].size), zin[n*y.size:(n+1)*y.size].real)
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
    parser.add_argument('--usrp_args',
                        type=str,
                        default='type=x300',
                        help='Type of USRP')
    parser.add_argument('--rx_gain',
                        type=float,
                        default=0,
                        help='USRP Rx gain')
    parser.add_argument('--tx_gain',
                        type=float,
                        default=0,
                        help='USRP Tx gain')
    parser.add_argument('--first_samp_time',
                        type=float,
                        default=0.2,
                        help='Time to wait for sending the signal')
    parser.add_argument('--subdev',
                        type=str,
                        default='A:0',
                        help='USRP subdevice')
    parser.add_argument('--ref',
                        type=str,
                        default='internal',
                        help='USRP reference clock')
    args = parser.parse_args()

    global logger
    logger = br.logging.getLogger(__name__)
    logger.setLevel(br.logging.DEBUG)
    console = br.logging.StreamHandler()
    logger.addHandler(console)
    formatter = br.LogFormatter(fmt='[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s')
    console.setFormatter(formatter)

    test(args)
