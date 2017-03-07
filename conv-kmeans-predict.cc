#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"
#include "unsupseg/embed.h"
#include "autodiff/autodiff.h"
#include "nn/tensor-tree.h"
#include "nn/nn.h"
#include <random>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree()
{
    tensor_tree::vertex root;

    root.children.push_back(tensor_tree::make_tensor("filters"));

    return std::make_shared<tensor_tree::vertex>(root);
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "conv-kmeans-predict",
        "Predict with learned filters",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"cluster", "", true},
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

    std::shared_ptr<tensor_tree::vertex> param = make_tensor_tree();
    tensor_tree::load_tensor(param, args.at("param"));

    std::ifstream frame_batch;
    frame_batch.open(args.at("frame-batch"));

    auto& filters = tensor_tree::get_tensor(param->children[0]);

    la::tensor<double> filter_energy;
    filter_energy.resize({filters.size(2)});

    for (int i = 0; i < filters.size(0); ++i) {
        for (int j = 0; j < filters.size(1); ++j) {
            for (int c = 0; c < filters.size(2); ++c) {
                filter_energy({c}) += filters({i, j, c}) * filters({i, j, c});
            }
        }
    }

    std::vector<std::vector<double>> stat;
    stat.resize(filters.size(2));

    auto less = [](double const& p1, double const& p2) {
        return p1 > p2;
    };

    int nsample = 0;

    while (1) {
        embed::seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        std::cerr << "sample: " << nsample << "\r";

        la::tensor<double> seg_tensor = embed::to_tensor(seg);

        if (seg_tensor.size(0) < filters.size(0) || seg_tensor.size(1) < filters.size(1)) {
            continue;
        }

        la::tensor<double> seg_lin;
        seg_lin.resize({seg_tensor.size(0) - filters.size(0) + 1,
            seg_tensor.size(1) - filters.size(1) + 1,
            filters.size(0) * filters.size(1)});

        la::corr_linearize_valid(seg_lin, seg_tensor, filters.size(0), filters.size(1));

        la::tensor<double> patch_energy;
        patch_energy.resize({seg_lin.size(0), seg_lin.size(1)});

        for (int i = 0; i < seg_lin.size(0); ++i) {
            for (int j = 0; j < seg_lin.size(1); ++j) {
                double sum = 0;
                for (int c = 0; c < seg_lin.size(2); ++c) {
                    sum += seg_lin({i, j, c}) * seg_lin({i, j, c});
                }
                patch_energy({i, j}) = sum;
            }
        }


        la::tensor<double> res = la::mul(seg_lin, filters);

        for (int i = 0; i < res.size(0); ++i) {
            for (int j = 0; j < res.size(1); ++j) {
                for (int c = 0; c < res.size(2); ++c) {
                    res({i, j, c}) = patch_energy({i, j}) - 2 * res({i, j, c}) + filter_energy({c});
                }
            }
        }

        double inf = std::numeric_limits<double>::infinity();
        la::tensor<double> min;
        min.resize({res.size(2)}, inf);
        std::vector<std::pair<int, int>> argmin;
        argmin.resize(res.size(2));

        for (int i = 0; i < res.size(0); ++i) {
            for (int j = 0; j < res.size(1); ++j) {
                for (int c = 0; c < res.size(2); ++c) {
                    if (res({i, j, c}) < min({c})) {
                        min({c}) = res({i, j, c});
                        argmin[c] = std::make_pair(i, j);
                    }
                }
            }
        }

        for (int c = 0; c < res.size(2); ++c) {
            stat[c].push_back(min({c}));
        }

        ++nsample;
    }

    std::cerr << std::endl;

    std::vector<std::vector<double>> stat_heap = stat;
    for (int c = 0; c < stat_heap.size(); ++c) {
        std::make_heap(stat_heap[c].begin(), stat_heap[c].end(), less);
    }

    std::vector<double> threshold;

    for (int c = 0; c < stat_heap.size(); ++c) {
        double t;
        for (int k = 0; k < std::min<int>(30, stat_heap[c].size()); ++k) {
            std::pop_heap(stat_heap[c].begin(), stat_heap[c].end(), less);
            t = stat_heap[c].back();
            stat_heap[c].pop_back();
        }
        threshold.push_back(t);
    }

    frame_batch.close();
    frame_batch.open(args.at("frame-batch"));

    nsample = 0;
    int cluster = std::stoi(args.at("cluster"));

    while (1) {
        embed::seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        std::cerr << "sample: " << nsample << "\r";

        la::tensor<double> seg_tensor = embed::to_tensor(seg);

        if (seg_tensor.size(0) < filters.size(0) || seg_tensor.size(1) < filters.size(1)) {
            continue;
        }

        if (stat[cluster][nsample] < threshold[cluster]) {
            std::cout << nsample << ".logmel" << std::endl;

            for (int i = 0; i < seg.size(); ++i) {
                for (int j = 0; j < seg[i].size(); ++j) {
                    std::cout << seg[i][j] << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "." << std::endl;
        }

        ++nsample;
    }

    std::cerr << std::endl;

    return 0;
}

