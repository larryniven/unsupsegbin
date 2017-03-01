#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"
#include "seg/dtw.h"
#include <random>

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw",
        "Calculate DTW distance",
        {
            {"frame-batch", "", true},
            {"nsegs", "", true},
            {"duration", "", true},
            {"seed", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    std::unordered_map<std::string, std::string> args = ebt::parse_args(argc, argv, spec);

    speech::batch_indices frame_batch;

    frame_batch.open(args.at("frame-batch"));

    int nsegs = std::stoi(args.at("nsegs"));

    int seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    std::default_random_engine gen {seed};

    std::vector<std::string> parts = ebt::split(args.at("duration"), ",");
    std::vector<int> durs;
    for (auto& p: parts) {
        durs.push_back(std::stoi(p));
    }

    int max_dur = *std::max_element(durs.begin(), durs.end());

    std::uniform_int_distribution<int> dur_dist{0, int(durs.size() - 1)};

    int nsample = 0;

    while (nsample < nsegs) {
        int idx = nsample % frame_batch.pos.size();

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(idx));

        std::uniform_int_distribution<int> start_dist{0, int(frames.size() - max_dur - 1)};

        int start_time = start_dist(gen);
        int di = dur_dist(gen);

        int end_time = std::min<int>(start_time + durs[di], frames.size());

        std::cout << nsample << ".logmel" << std::endl;

        for (int i = start_time; i < end_time; ++i) {
            for (int j = 0; j < frames[i].size(); ++j) {
                std::cout << frames[i][j];
                if (j != frames[i].size() - 1) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }

        std::cout << "." << std::endl;

        ++nsample;
    }

    return 0;
}
