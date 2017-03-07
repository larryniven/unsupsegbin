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

void mvnorm(la::tensor<double>& t)
{
    double sum = 0;
    double sum_sq = 0;

    double *data = t.data();

    for (int i = 0; i < t.vec_size(); ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    double mean = sum / t.vec_size();
    double stddev = std::sqrt((sum_sq - sum) / t.vec_size());

    for (int i = 0; i < t.vec_size(); ++i) {
        data[i] = (data[i] - mean) / stddev;
    }
}

void mvnorm_filters(la::tensor<double>& t)
{
    for (int c = 0; c < t.size(2); ++c) {
        la::tensor<double> v;
        v.resize({t.size(0), t.size(1)});

        for (int i = 0; i < t.size(0); ++i) {
            for (int j = 0; j < t.size(1); ++j) {
                v({i, j}) = t({i, j, c});
            }
        }

        mvnorm(v);

        for (int i = 0; i < t.size(0); ++i) {
            for (int j = 0; j < t.size(1); ++j) {
                t({i, j, c}) = v({i, j});
            }
        }
    }
}

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "conv-kmeans-learn",
        "Learn conv filters with k-means",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"output-param", "", true},
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

    std::vector<std::vector<std::pair<double, int>>> stat;
    stat.resize(filters.size(2));

    std::vector<la::tensor<double>> examples;

    auto less = [](std::pair<double, int> const& p1, std::pair<double, int> const& p2) {
        return p1.first > p2.first;
    };

    int nsample = 0;

    while (1) {
        embed::seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        la::tensor<double> seg_tensor = embed::to_tensor(seg);

        if (seg_tensor.size(0) < filters.size(0) || seg_tensor.size(1) < filters.size(1)) {
            continue;
        }

        la::tensor<double> seg_lin;
        seg_lin.resize({seg_tensor.size(0) - filters.size(0) + 1, seg_tensor.size(1) - filters.size(1) + 1,
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

        double loss_sum = 0;

        for (int c = 0; c < min.size(0); ++c) {
            loss_sum += min({c});
        }

        std::cout << "sample: " << nsample << std::endl;
        std::cout << "loss: " << loss_sum / min.size(0) << std::endl;
        std::cout << std::endl;

        for (int c = 0; c < res.size(2); ++c) {
            la::tensor<double> example;
            example.resize({seg_lin.size(2)});

            for (int d = 0; d < seg_lin.size(2); ++d) {
                example.data()[d] = seg_lin({argmin[c].first, argmin[c].second, d});
            }

            examples.push_back(example);
            stat[c].push_back(std::make_pair(min({c}), examples.size() - 1));
            std::push_heap(stat[c].begin(), stat[c].end(), less);
        }

        ++nsample;
    }

    std::vector<int> count;
    count.resize(stat.size());

    std::vector<la::tensor<double>> sum;
    for (int c = 0; c < stat.size(); ++c) {
        la::tensor<double> v;
        v.resize({filters.size(0), filters.size(1)});
        sum.push_back(v);
    }

    int total_count = 0;
    double loss = 0;

    for (int c = 0; c < stat.size(); ++c) {
        for (int k = 0; k < std::min<int>(30, stat[c].size()); ++k) {
            std::pop_heap(stat[c].begin(), stat[c].end(), less);
            loss += stat[c].back().first;
            int index = stat[c].back().second;
            la::iadd(sum[c], examples[index]);
            count[c] += 1;
            total_count += 1;
            stat[c].pop_back();
        }
    }

    std::cout << "loss: " << loss / total_count << std::endl;

    for (int c = 0; c < filters.size(2); ++c) {
        for (int i = 0; i < filters.size(0); ++i) {
            for (int j = 0; j < filters.size(1); ++j) {
                filters({i, j, c}) = sum[c]({i, j}) / count[c];
            }
        }
    }

    std::ofstream param_ofs { args.at("output-param") };
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    return 0;
}

