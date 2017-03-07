#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"
#include "unsupseg/embed.h"
#include <random>

using seg_t = std::vector<std::vector<double>>;

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw-embed-kmeans",
        "Cluster reference vectors with k-means",
        {
            {"frame-batch", "", true},
            {"basis-batch", "", true},
            {"centers", "", true},
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

    la::tensor<double> basis_tensor = embed::to_tensor(basis);

    std::ifstream frame_batch;
    frame_batch.open(args.at("frame-batch"));

    std::vector<la::vector<double>> centers;

    if (ebt::in(std::string("centers"), args)) {
        std::ifstream centers_ifs { args.at("centers") };

        std::string line;

        while (std::getline(centers_ifs, line)) {
            auto parts = ebt::split(line);

            la::vector<double> v;
            v.resize(parts.size());

            int d = 0;
            for (auto& p: parts) {
                v(d) = std::stod(p);
                ++d;
            }

            centers.push_back(v);
        }
    }

    std::vector<std::pair<double, int>> stat;
    stat.resize(centers.size());

    int nsample = 0;

    while (1) {
        seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        la::tensor<double> seg_tensor = embed::to_tensor(seg);

        la::tensor<double> seg_embed = embed::conv_embed(seg_tensor, basis_tensor);
        la::imul(seg_embed, 1.0 / la::norm(seg_embed));

        double inf = std::numeric_limits<double>::infinity();
        double min = inf;
        int argmin = -1;

        for (int k = 0; k < centers.size(); ++k) {
            double dist = la::norm(la::sub(centers[k], seg_embed.as_vector()));

            if (dist < min) {
                min = dist;
                argmin = k;
            }
        }

        stat[argmin].first += min;
        stat[argmin].second += 1;

        std::cout << "sample: " << nsample << std::endl;
        std::cout << "id: " << argmin << std::endl;
        std::cout << std::endl;

        ++nsample;
    }

    for (int i = 0; i < stat.size(); ++i) {
        std::cout << "cluster " << i << ": " << stat[i].first / stat[i].second << std::endl;
    }

    return 0;
}

