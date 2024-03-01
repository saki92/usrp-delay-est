
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread.hpp>

namespace po = boost::program_options;

int UHD_SAFE_MAIN(int argc, char *argv[]) {
  std::string args, file, subdev, ref;
  double seconds_in_future, rate, freq, gain, power, repeat_delay;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help", "help message")
      ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
      ("file", po::value<std::string>(&file)->default_value("usrp_samples.dat"), "name of the file to read binary samples from")
      ("rate", po::value<double>(&rate), "rate of outgoing samples")("freq", po::value<double>(&freq), "RF center frequency in Hz")
      ("gain", po::value<double>(&gain), "gain for the RF chain")
      ("power", po::value<double>(&power), "transmit power")
      ("subdev", po::value<std::string>(&subdev), "subdevice specification")
      ("ref", po::value<std::string>(&ref), "clock reference (internal, external, mimo, gpsdo)")
      ("repeat", "repeatedly transmit file")
  ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << boost::format("USRP Delay Estimator %s") % desc << std::endl;
    return ~0;
  }

  bool repeat = vm.count("repeat") > 0;

  uhd::set_thread_priority_safe();

  return EXIT_SUCCESS;
}
