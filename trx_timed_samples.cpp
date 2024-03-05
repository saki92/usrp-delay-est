
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <fstream>
#include <iostream>
#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread.hpp>

using namespace std::chrono_literals;

typedef std::function<uhd::sensor_value_t(const std::string &)> get_sensor_fn_t;

bool check_locked_sensor(std::vector<std::string> sensor_names,
                         const char *sensor_name, get_sensor_fn_t get_sensor_fn,
                         double setup_time) {
  if (std::find(sensor_names.begin(), sensor_names.end(), sensor_name) ==
      sensor_names.end())
    return false;

  const auto setup_timeout =
      std::chrono::steady_clock::now() + (setup_time * 1s);
  bool lock_detected = false;

  std::cout << "Waiting for \"" << sensor_name << "\": ";
  std::cout.flush();

  while (true) {
    if (lock_detected and (std::chrono::steady_clock::now() > setup_timeout)) {
      std::cout << " locked." << std::endl;
      break;
    }
    if (get_sensor_fn(sensor_name).to_bool()) {
      std::cout << "+";
      std::cout.flush();
      lock_detected = true;
    } else {
      if (std::chrono::steady_clock::now() > setup_timeout) {
        std::cout << std::endl;
        throw std::runtime_error(str(
            boost::format(
                "timed out waiting for consecutive locks on sensor \"%s\"") %
            sensor_name));
      }
      std::cout << "_";
      std::cout.flush();
    }
    std::this_thread::sleep_for(100ms);
  }
  std::cout << std::endl;
  return true;
}

void transmit_worker(const std::vector<std::complex<float> *> buffs,
                     uhd::tx_streamer::sptr tx_streamer,
                     uhd::time_spec_t first_tx_time, size_t samples_per_buff,
                     int repeat_times, double sample_rate) {
  uhd::tx_metadata_t md;
  md.start_of_burst = true;
  md.has_time_spec = true;
  md.time_spec = first_tx_time;
  int64_t timestamp = first_tx_time.to_ticks(sample_rate);
  std::cout << boost::format("first tx sample set at %ld") %
                   (md.time_spec.to_ticks(sample_rate))
            << std::endl;
  int repeat_cnt = 0;
  while (repeat_cnt < repeat_times) {
    // send the entire contents of the buffer
    const int samples_sent = tx_streamer->send(buffs, samples_per_buff, md);
    if (samples_sent != samples_per_buff) {
      UHD_LOG_ERROR("TX-STREAM", "The tx_stream timed out sending "
                                     << samples_per_buff << " samples ("
                                     << samples_sent << " sent).");
      return;
    }
    repeat_cnt++;
    md.time_spec = uhd::time_spec_t::from_ticks(
        timestamp + (repeat_cnt * samples_per_buff), sample_rate);
    md.start_of_burst = false;
    std::cout << boost::format("sent %d samples. next ts %ld") %
                     (samples_sent) %
                     (timestamp + (repeat_cnt * samples_per_buff))
              << std::endl;
  }
  // send a mini EOB packet
  md.end_of_burst = true;
  tx_streamer->send("", 0, md);
}

void receive_worker(const std::string &file, uhd::rx_streamer::sptr rx_streamer,
                    uhd::time_spec_t first_rx_time, size_t samples_per_buff,
                    int repeat_times, double sample_rate) {

  std::vector<std::complex<float>> buff(samples_per_buff);
  std::vector<std::complex<float> *> buff_ptr;
  buff_ptr.push_back(&buff.front());

  std::ofstream outfile;
  outfile.open(file.c_str(), std::ofstream::binary);

  bool overflow_message = true;

  // setup streaming
  uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  stream_cmd.stream_now = false;
  stream_cmd.time_spec = first_rx_time;
  std::cout << boost::format("first rx sample set at %ld") %
                   (stream_cmd.time_spec.to_ticks(sample_rate))
            << std::endl;
  rx_streamer->issue_stream_cmd(stream_cmd);

  int repeat_cnt = 0;
  double timeout = 2.0f; // first rx packet. we wait for first_rx_time
  uhd::rx_metadata_t md;
  while (repeat_cnt < repeat_times) {
    int samples_received = 0;
    bool first_rx = true;
    while (samples_received != samples_per_buff) {
      samples_received += rx_streamer->recv(
          (void *)((std::complex<float> *)buff_ptr[0] + samples_received),
          samples_per_buff - samples_received, md, timeout);
      if (first_rx) {
        std::cout << boost::format("First rx timestamp %ld") %
                         (md.time_spec.to_ticks(sample_rate))
                  << std::endl;
        first_rx = false;
      }
    }
    timeout = 0.1f;

    if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
      std::cout << "Timeout while streaming" << std::endl;
      break;
    }
    if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
      if (overflow_message) {
        overflow_message = false;
        std::cerr << boost::format(
                         "Got an overflow indication. Please consider the "
                         "following:\n"
                         "  Your write medium must sustain a rate of %fMB/s.\n"
                         "  Dropped samples will not be written to the file.\n"
                         "  Please modify this example for your purposes.\n"
                         "  This message will not appear again.\n") %
                         (sample_rate * sizeof(std::complex<float>) / 1e6);
      }
      continue;
    }

    if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
      throw std::runtime_error("Receiver error " + md.strerror());
    }

    outfile.write((const char *)&buff.front(),
                  samples_received * sizeof(std::complex<float>));
    repeat_cnt++;
    // std::cout << boost::format("received %d samples with ts %ld") %
    //                  (samples_received) %
    //                  (md.time_spec.to_ticks(sample_rate))
    //           << std::endl;
  }
  stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
  rx_streamer->issue_stream_cmd(stream_cmd);

  outfile.close();
}

namespace po = boost::program_options;

int UHD_SAFE_MAIN(int argc, char *argv[]) {
  std::string args, tx_file, rx_file, subdev, ref, wirefmt;
  int channel, spb, repeat;
  double seconds_in_future, rate, freq, lo_offset, tx_gain, rx_gain, bw,
      setup_time, repeat_delay, first_samp_time;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help", "help message")
      ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
      ("txfile", po::value<std::string>(&tx_file)->default_value("tx_usrp_samples.dat"), "name of the file to read binary samples from")
      ("rxfile", po::value<std::string>(&rx_file)->default_value("rx_usrp_samples.dat"), "name of the file to write binary samples to")
      ("rate", po::value<double>(&rate), "rate of outgoing samples")
      ("freq", po::value<double>(&freq), "RF center frequency in Hz")
      ("subdev", po::value<std::string>(&subdev), "subdevice specification")
      ("ref", po::value<std::string>(&ref)->default_value("internal"), "clock reference (internal, external, mimo, gpsdo)")
      ("channel", po::value<int>(&channel)->default_value(0), "channel number")
      ("lo-offset", po::value<double>(&lo_offset)->default_value(0.0), "Offset for frontend LO in Hz")
      ("tx-gain", po::value<double>(&tx_gain)->default_value(0.0), "gain of the Tx RF chain")
      ("rx-gain", po::value<double>(&rx_gain)->default_value(0.0), "gain of the Rx RF chain")
      ("bw", po::value<double>(&bw), "analog frontend filter bandwidth in Hz")
      ("setup", po::value<double>(&setup_time)->default_value(1.0), "seconds of setup time")
      ("first-sample-time", po::value<double>(&first_samp_time)->default_value(0.1), "delay in sending/receiving the first sample")
      ("wirefmt", po::value<std::string>(&wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
      ("spb", po::value<int>(&spb)->default_value(10000), "samples per buffer")
      ("repeat", po::value<int>(&repeat)->default_value(10), "repeat transmit file times")
  ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << boost::format("USRP Delay Estimator %s") % desc << std::endl;
    return ~0;
  }

  uhd::set_thread_priority_safe();

  // create a usrp device
  std::cout << std::endl;
  std::cout << boost::format("Creating the usrp device with: %s...") % args
            << std::endl;
  uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);
  std::cout << boost::format("Using device: %s") % usrp->get_pp_string()
            << std::endl;

  // always select the subdevice first, the channel mapping affects the other
  // settings
  if (vm.count("subdev")) {
    usrp->set_tx_subdev_spec(subdev);
    usrp->set_rx_subdev_spec(subdev);
  } else {
    std::cerr << "Please specify the subdev with --subdev" << std::endl;
    return ~0;
  }

  // Lock mboard clocks
  if (vm.count("ref")) {
    usrp->set_clock_source(ref);
  }

  /*------------- Tx setup -------------*/
  // set tx sample rate
  if (not vm.count("rate")) {
    std::cerr << "Please specify the sampling frequency with --rate"
              << std::endl;
    return ~0;
  }
  std::cout << boost::format("Setting TX Rate: %f Msps...") % (rate / 1e6)
            << std::endl;
  usrp->set_tx_rate(rate, channel);
  std::cout << boost::format("Actual TX Rate: %f Msps...") %
                   (usrp->get_tx_rate(channel) / 1e6)
            << std::endl
            << std::endl;

  // set tx carrier freq
  if (not vm.count("freq")) {
    std::cerr << "Please specify the center frequency with --freq" << std::endl;
    return ~0;
  }
  std::cout << boost::format("Setting TX Freq: %f MHz...") % (freq / 1e6)
            << std::endl;
  std::cout << boost::format("Setting TX LO Offset: %f MHz...") %
                   (lo_offset / 1e6)
            << std::endl;
  uhd::tune_request_t tune_request(freq, lo_offset);
  usrp->set_tx_freq(tune_request, channel);
  std::cout << boost::format("Actual TX Freq: %f MHz...") %
                   (usrp->get_tx_freq(channel) / 1e6)
            << std::endl
            << std::endl;

  // set tx gain
  std::cout << boost::format("Setting TX Gain: %f dB...") % tx_gain
            << std::endl;
  usrp->set_tx_gain(tx_gain, channel);
  std::cout << boost::format("Actual TX Gain: %f dB...") %
                   usrp->get_tx_gain(channel)
            << std::endl
            << std::endl;

  if (vm.count("bw")) {
    std::cout << boost::format("Setting TX Bandwidth: %f MHz...") % (bw / 1e6)
              << std::endl;
    usrp->set_tx_bandwidth(bw, channel);
    std::cout << boost::format("Actual TX Bandwidth: %f MHz...") %
                     (usrp->get_tx_bandwidth(channel) / 1e6)
              << std::endl
              << std::endl;
  }

  /*------------- Rx setup -------------*/
  // set rx sample rate
  std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate / 1e6)
            << std::endl;
  usrp->set_rx_rate(rate, channel);
  std::cout << boost::format("Actual RX Rate: %f Msps...") %
                   (usrp->get_rx_rate(channel) / 1e6)
            << std::endl
            << std::endl;

  // set rx carrier freq
  std::cout << boost::format("Setting RX Freq: %f MHz...") % (freq / 1e6)
            << std::endl;
  std::cout << boost::format("Setting RX LO Offset: %f MHz...") %
                   (lo_offset / 1e6)
            << std::endl;
  usrp->set_rx_freq(tune_request, channel);
  std::cout << boost::format("Actual RX Freq: %f MHz...") %
                   (usrp->get_rx_freq(channel) / 1e6)
            << std::endl
            << std::endl;

  // set rx gain
  std::cout << boost::format("Setting RX Gain: %f dB...") % rx_gain
            << std::endl;
  usrp->set_rx_gain(rx_gain, channel);
  std::cout << boost::format("Actual RX Gain: %f dB...") %
                   usrp->get_rx_gain(channel)
            << std::endl
            << std::endl;

  if (vm.count("bw")) {
    std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % (bw / 1e6)
              << std::endl;
    usrp->set_rx_bandwidth(bw, channel);
    std::cout << boost::format("Actual RX Bandwidth: %f MHz...") %
                     (usrp->get_rx_bandwidth(channel) / 1e6)
              << std::endl
              << std::endl;
  }

  /*----------------------------------------------------*/
  // check ref and lock detect
  check_locked_sensor(
      usrp->get_rx_sensor_names(channel), "lo_locked",
      [usrp, channel](const std::string &sensor_name) {
        return usrp->get_rx_sensor(sensor_name, channel);
      },
      setup_time);

  if (ref == "external") {
    check_locked_sensor(
        usrp->get_mboard_sensor_names(0), "ref_locked",
        [usrp](const std::string &sensor_name) {
          return usrp->get_mboard_sensor(sensor_name);
        },
        setup_time);
  }

  // create transmit streamer
  std::string cpu_format = "fc32";
  uhd::stream_args_t tx_stream_args(cpu_format, wirefmt);
  uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(tx_stream_args);

  // prepare transmit buffer (read from file)
  std::vector<std::complex<float>> tx_buff(spb);
  std::vector<std::complex<float> *> tx_buffs(tx_stream->get_num_channels(),
                                              &tx_buff.front());
  std::ifstream infile(tx_file.c_str(), std::ifstream::binary);
  int num_tx_samps = 0;
  while (not infile.eof()) {
    infile.read((char *)&tx_buff.front(),
                tx_buff.size() * sizeof(std::complex<float>));
    num_tx_samps += int(infile.gcount() / sizeof(std::complex<float>));
  }
  std::cout << boost::format("Read %d samples from file %s") % (num_tx_samps) % (tx_file.c_str()) << std::endl;
  infile.close();

  // create a receive streamer
  uhd::stream_args_t rx_stream_args(cpu_format, wirefmt);
  std::vector<size_t> rx_channel_num;
  rx_channel_num.push_back(0); // we use only one channel
  rx_stream_args.channels = rx_channel_num;
  uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(rx_stream_args);

  // reset usrp time to prepare for transmit/receive
  std::cout << boost::format("Setting device timestamp to 0...") << std::endl;
  usrp->set_time_now(uhd::time_spec_t(0.0));

  uhd::time_spec_t current_time = usrp->get_time_now();

  uhd::time_spec_t first_sample_time =
      current_time + uhd::time_spec_t(first_samp_time);

  // start transmit worker thread
  std::thread transmit_thread([&]() {
    transmit_worker(tx_buffs, tx_stream, first_sample_time, num_tx_samps,
                    repeat, rate);
  });

  // start receive worker in same thread
  int64_t rx_first_samp = first_sample_time.to_ticks(
      rate); //-48; // b200 seem to start streaming with a dealy of 48 samples
  uhd::time_spec_t rx_first_sample_time =
      uhd::time_spec_t::from_ticks(rx_first_samp, rate);
  receive_worker(rx_file, rx_stream, rx_first_sample_time, num_tx_samps, repeat,
                 rate);

  if (transmit_thread.joinable()) {
    transmit_thread.join();
  };
  // finished
  std::cout << std::endl << "Done!" << std::endl << std::endl;
  return EXIT_SUCCESS;
}
