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
        "conv-embed-kmeans",
        "Cluster conv embedding vectors with k-means",
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

    la::tensor<double> basis_tensor = embed::to_tensor(basis);

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

    for (int i = 0; i < iter; ++i) {

        std::vector<int> cluster_id;
        double loss = 0;

        int nsample = 0;

        while (nsample < frame_batch.pos.size()) {
            seg_t seg = speech::load_frame_batch(frame_batch.at(nsample));

            la::tensor<double> seg_tensor = embed::to_tensor(seg);

            la::tensor<double> seg_embed = embed::conv_embed(seg_tensor, basis_tensor);
            std::cout << "embed: " << seg_embed.vec_size() << std::endl;
            la::imul(seg_embed, 1.0 / la::norm(seg_embed));

            double inf = std::numeric_limits<double>::infinity();
            double min = inf;
            int argmin = -1;

            if (centers.size() < kcluster) {
                centers.push_back(la::vector<double>(seg_embed.as_vector()));
                argmin = centers.size() - 1;
                min = 0;
            } else {
                for (int k = 0; k < kcluster; ++k) {
                    double dist = la::norm(la::sub(centers[k], seg_embed.as_vector()));

                    if (dist < min) {
                        min = dist;
                        argmin = k;
                    }
                }

            }

            loss += min;
            cluster_id.push_back(argmin);

            while (stat.size() < centers.size()) {
                la::vector<double> v;
                v.resize(centers.front().size());
                stat.push_back(std::make_pair(v, 0));
            }

            la::iadd(stat[argmin].first, seg_embed.as_vector());
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

