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
            {"k", "", true},
            {"centers", "", false},
            {"output-centers", "", true},
            {"iter", "", true},
            {"seed", "", false},
            {"shuffle", "", false},
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

    int iter = std::stod(args.at("iter"));
    int kcluster = std::stoi(args.at("k"));

    int seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    std::default_random_engine gen { seed };

    std::string output_centers = args.at("output-centers");

    speech::batch_indices frame_batch;
    frame_batch.open(args.at("frame-batch"));

    std::vector<int> sample_indices;
    sample_indices.resize(frame_batch.pos.size());
    for (int i = 0; i < frame_batch.pos.size(); ++i) {
        sample_indices[i] = i;
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(sample_indices.begin(), sample_indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < pos.size(); ++i) {
            frame_batch.pos[i] = pos[sample_indices[i]];
        }
    }

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

    std::vector<std::pair<la::vector<double>, int>> stat;
    for (int k = 0; k < kcluster; ++k) {
        la::vector<double> v;
        v.resize(basis.size());
        stat.push_back(std::make_pair(v, 0));
    }

    for (int i = 0; i < iter; ++i) {

        std::vector<int> cluster_id;
        double loss = 0;

        int nsample = 0;

        while (nsample < frame_batch.pos.size()) {
            seg_t seg = speech::load_frame_batch(frame_batch.at(nsample));

            la::vector<double> seg_embed = embed::dtw_embed(seg, basis);
            la::imul(seg_embed, 1.0 / la::norm(seg_embed));

            double inf = std::numeric_limits<double>::infinity();
            double min = inf;
            int argmin = -1;

            if (centers.size() < kcluster) {
                centers.push_back(seg_embed);
                argmin = centers.size() - 1;
                min = 0;
            } else {
                for (int k = 0; k < kcluster; ++k) {
                    double dist = la::norm(la::sub(centers[k], seg_embed));

                    if (dist < min) {
                        min = dist;
                        argmin = k;
                    }
                }

            }

            loss += min;
            cluster_id.push_back(argmin);
            la::iadd(stat[argmin].first, seg_embed);
            stat[argmin].second += 1;

            std::cout << "sample: " << nsample << std::endl;
            std::cout << "id: " << argmin << std::endl;
            std::cout << "running loss: " << loss / (nsample + 1) << std::endl;
            std::cout << std::endl;

            ++nsample;
        }

        std::cout << "loss: " << loss / nsample << std::endl;

        for (int k = 0; k < kcluster; ++k) {
            centers[k] = la::mul(stat[k].first, 1.0 / stat[k].second);
        }

        std::ofstream centers_ofs { "centers-tmp" };
        for (int k = 0; k < kcluster; ++k) {
            for (int d = 0; d < centers[k].size(); ++d) {
                centers_ofs << centers[k](d);
                if (d != centers[k].size() - 1) {
                    centers_ofs << " ";
                }
            }
            centers_ofs << std::endl;
        }
        centers_ofs.close();

    }

    std::ofstream centers_ofs { output_centers };
    for (int k = 0; k < kcluster; ++k) {
        for (int d = 0; d < centers[k].size(); ++d) {
            centers_ofs << centers[k](d);
            if (d != centers[k].size() - 1) {
                centers_ofs << " ";
            }
        }
        centers_ofs << std::endl;
    }
    centers_ofs.close();

    return 0;
}

