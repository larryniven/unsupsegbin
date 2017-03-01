#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"
#include "unsupseg/dtw.h"

using seg_t = std::vector<std::vector<double>>;

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw-embed",
        "Calculate distance based on DTW-embedding",
        {
            {"frame-batch", "", true},
            {"basis-batch", "", true},
            {"target", "", true}
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

    std::vector<seg_t> basis;
    std::ifstream basis_batch { args.at("basis-batch") };

    while (1) {
        seg_t frames = speech::load_frame_batch(basis_batch);

        if (!basis_batch) {
            break;
        }

        basis.push_back(frames);
    }

    basis_batch.close();

    std::ifstream target_ifs { args.at("target") };
    seg_t target = speech::load_frame_batch(target_ifs);
    target_ifs.close();

    la::vector<double> target_embed = dtw::embed(target, basis);
    la::imul(target_embed, 1.0 / la::norm(target_embed));

    std::ifstream frame_batch { args.at("frame-batch") };

    while (1) {
        seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        la::vector<double> seg_embed = dtw::embed(seg, basis);
        la::imul(seg_embed, 1.0 / la::norm(seg_embed));

        std::cout << "dist: " << la::dot(seg_embed, target_embed) << std::endl;
    }

    return 0;
}
