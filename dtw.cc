#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"
#include "seg/dtw.h"

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw",
        "Calculate DTW distance",
        {
            {"frame-batch", "", true},
            {"target", "", true},
            {"target-norm", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    std::ifstream frame_batch { args.at("frame-batch") };

    std::ifstream target_ifs { args.at("target") };
    std::vector<std::vector<double>> target = speech::load_frame_batch(target_ifs);
    target_ifs.close();

    while (1) {
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        double d = dtw::dtw(frames, target);

        if (ebt::in(std::string("target-norm"), args)) {
            d = d / target.size();
        }

        std::cout << "dist: " << d << std::endl;
    }

    return 0;
}
