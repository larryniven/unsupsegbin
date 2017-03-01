#include "ebt/ebt.h"
#include "nn/tensor-tree.h"
#include "speech/speech.h"
#include "nn/lstm-frame.h"
#include "seg/dtw.h"
#include "nn/lstm-tensor-tree.h"
#include <random>
#include <algorithm>

std::vector<std::shared_ptr<autodiff::op_t>>
to_op(std::vector<std::vector<double>> const& frames,
    autodiff::computation_graph& comp_graph);

std::shared_ptr<tensor_tree::vertex>
make_tensor_tree(int layer);

std::shared_ptr<autodiff::op_t>
embed(std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    int layer,
    std::shared_ptr<tensor_tree::vertex> var_tree);

struct prediction_env {

    prediction_env(std::unordered_map<std::string, std::string> const& args);

    std::ifstream seg_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    std::unordered_map<std::string, std::string> args;

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw-lstm-predict",
        "Predict distance",
        {
            {"seg-batch", "", true},
            {"target", "", true},
            {"param", "", true},
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

    std::unordered_map<std::string, std::string> args
        = ebt::parse_args(argc, argv, spec);

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(
    std::unordered_map<std::string, std::string> const& args)
    : args{args}
{
    seg_batch.open(args.at("seg-batch"));

    std::string line;
    std::ifstream param_ifs {args.at("param")};
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();
}

void prediction_env::run()
{
    int nsample = 0;

    std::ifstream target_ifs { args.at("target") };
    std::vector<std::vector<double>> target = speech::load_frame_batch(target_ifs);
    target_ifs.close();

    while (1) {
        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<lstm::transcriber> transcriber;

        std::vector<std::vector<double>> seg = speech::load_frame_batch(seg_batch);
        ++nsample;

        if (!seg_batch) {
            break;
        }

        auto seg_op = to_op(seg, comp_graph);
        auto e1 = embed(seg_op, layer, var_tree);
        auto target_op = to_op(target, comp_graph);
        auto e2 = embed(target_op, layer, var_tree);

        auto dist = autodiff::norm(autodiff::sub(e1, e2));

        std::cout << "dist: " << autodiff::get_output<double>(dist) << std::endl;
    }

}

std::vector<std::shared_ptr<autodiff::op_t>>
to_op(std::vector<std::vector<double>> const& frames,
    autodiff::computation_graph& comp_graph)
{
    std::vector<std::shared_ptr<autodiff::op_t>> result;

    for (auto& f: frames) {
        result.push_back(comp_graph.var(la::tensor<double>{la::vector<double>{f}}));
    }

    return result;
}

std::shared_ptr<tensor_tree::vertex>
make_tensor_tree(int layer)
{
    // tensor_tree::vertex result { tensor_tree::tensor_t::nil };

    lstm::multilayer_lstm_tensor_tree_factory factory {
        std::make_shared<lstm::bi_lstm_tensor_tree_factory>(
        lstm::bi_lstm_tensor_tree_factory {
            std::make_shared<lstm::dyer_lstm_tensor_tree_factory>(
                lstm::dyer_lstm_tensor_tree_factory{})
        }),
        layer
    };

    // result.children.push_back(factory());
    // result.children.push_back(tensor_tree::make_tensor("projection weight"));
    // result.children.push_back(tensor_tree::make_tensor("projection bias"));

    // return std::make_shared<tensor_tree::vertex>(result);

    return std::shared_ptr<tensor_tree::vertex>(factory());
}

std::shared_ptr<autodiff::op_t>
embed(std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    int layer,
    std::shared_ptr<tensor_tree::vertex> var_tree)
{
    std::shared_ptr<lstm::step_transcriber> step;

    step = std::make_shared<lstm::dyer_lstm_step_transcriber>(
        lstm::dyer_lstm_step_transcriber{});

    lstm::layered_transcriber result;

    for (int i = 0; i < layer; ++i) {
        std::shared_ptr<lstm::transcriber> trans;

        trans = std::make_shared<lstm::lstm_transcriber>(
            lstm::lstm_transcriber { step });

        trans = std::make_shared<lstm::bi_transcriber>(
            lstm::bi_transcriber { trans });

        result.layer.push_back(trans);
    }

    std::shared_ptr<lstm::transcriber> trans = std::make_shared<lstm::layered_transcriber>(result);

    std::vector<std::shared_ptr<autodiff::op_t>> feat = (*trans)(var_tree, seg_frames);

    return autodiff::add(feat);
}

