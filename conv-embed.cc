#include "la/la.h"
#include "ebt/ebt.h"
#include <fstream>
#include <algorithm>
#include "speech/speech.h"

using seg_t = std::vector<std::vector<double>>;

la::tensor<double> to_tensor(std::vector<std::vector<double>> const& seg)
{
    la::tensor<double> result;
    result.resize({(unsigned int)(seg.size()), (unsigned int)(seg.front().size())});

    for (int i = 0; i < result.size(0); ++i) {
        for (int j = 0; j < result.size(1); ++j) {
            result({i, j}) = seg[i][j];
        }
    }

    return result;
}

la::tensor<double> to_tensor(std::vector<std::vector<std::vector<double>>> const& basis)
{
    la::tensor<double> result;
    result.resize({(unsigned int)(basis.front().size()),
        (unsigned int)(basis.front().front().size()), (unsigned int)(basis.size())});

    for (int i = 0; i < result.size(0); ++i) {
        for (int j = 0; j < result.size(1); ++j) {
            for (int c = 0; c < result.size(2); ++c) {
                result({i, j, c}) = basis[c][i][j];
            }
        }
    }

    return result;
}

la::tensor<double> embed(la::tensor<double> const& seg, la::tensor<double> const& basis)
{
    la::tensor<double> seg_lin;

    seg_lin.resize({seg.size(0), seg.size(1), basis.size(0) * basis.size(1)});

    la::corr_linearize(seg_lin, seg, basis.size(0), basis.size(1));

    la::tensor<double> corr = la::mul(seg_lin, basis);

    double inf = std::numeric_limits<double>::infinity();

    la::tensor<double> result;
    result.resize({corr.size(2)}, -inf);

    for (int i = 0; i < corr.size(0); ++i) {
        for (int j = 0; j < corr.size(1); ++j) {
            for (int c = 0; c < corr.size(2); ++c) {
                result({c}) = std::max(result({c}), corr({i, j, c}));
            }
        }
    }

    return result;
}

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

    la::tensor<double> basis_tensor = to_tensor(basis);

    std::ifstream target_ifs { args.at("target") };
    seg_t target = speech::load_frame_batch(target_ifs);
    target_ifs.close();

    la::tensor<double> target_tensor = to_tensor(target);

    la::tensor<double> target_embed = embed(target_tensor, basis_tensor);
    la::imul(target_embed, 1.0 / la::norm(target_embed));

    std::ifstream frame_batch { args.at("frame-batch") };

    while (1) {
        seg_t seg = speech::load_frame_batch(frame_batch);

        if (!frame_batch) {
            break;
        }

        la::tensor<double> seg_tensor = to_tensor(seg);

        la::tensor<double> seg_embed = embed(seg_tensor, basis_tensor);
        la::imul(seg_embed, 1.0 / la::norm(seg_embed));

        std::cout << "dist: " << la::dot(seg_embed, target_embed) << std::endl;
    }

    return 0;
}
