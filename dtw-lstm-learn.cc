#include "ebt/ebt.h"
#include "nn/tensor-tree.h"
#include "speech/speech.h"
#include "nn/lstm-frame.h"
#include "unsupseg/dtw.h"
#include "nn/lstm-tensor-tree.h"
#include <random>
#include <algorithm>

std::vector<std::vector<double>>
sample_seg(std::vector<std::vector<double>> const& frames,
    std::default_random_engine& gen);

std::vector<std::shared_ptr<autodiff::op_t>>
to_op(std::vector<std::vector<double>> const& frames,
    autodiff::computation_graph& comp_graph);

std::shared_ptr<tensor_tree::vertex>
make_tensor_tree(int layer);

std::shared_ptr<autodiff::op_t>
embed(std::vector<std::shared_ptr<autodiff::op_t>> const& seg_frames,
    int layer,
    std::shared_ptr<tensor_tree::vertex> var_tree);

struct learning_env {

    learning_env(std::unordered_map<std::string, std::string> const& args);

    speech::batch_indices frame_batch;

    int layer;
    std::shared_ptr<tensor_tree::vertex> param;

    double step_size;
    double clip;

    int seed;
    std::default_random_engine gen;

    std::shared_ptr<tensor_tree::optimizer> opt;

    std::unordered_map<std::string, std::string> args;

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "dtw-lstm-learn",
        "Train an LSTM to produce embeddings that respect DTW",
        {
            {"frame-batch", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", true},
            {"output-opt-data", "", true},
            {"step-size", "", true},
            {"clip", "", false},
            {"const-step-update", "", false},
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

    std::unordered_map<std::string, std::string> args
        = ebt::parse_args(argc, argv, spec);

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(
    std::unordered_map<std::string, std::string> const& args)
    : args{args}
{
    frame_batch.open(args.at("frame-batch"));

    std::string line;
    std::ifstream param_ifs {args.at("param")};
    std::getline(param_ifs, line);
    layer = std::stoi(line);
    param = make_tensor_tree(layer);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    if (ebt::in(std::string("const-step-update"), args)) {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt(param, step_size));
    } else {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt(param, step_size));
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    std::getline(opt_data_ifs, line);
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    gen = std::default_random_engine{seed};

    if (ebt::in(std::string("shuffle"), args)) {
        std::vector<int> sample_indices;
        for (int i = 0; i < frame_batch.pos.size(); ++i) {
            sample_indices.push_back(i);
        }

        std::shuffle(sample_indices.begin(), sample_indices.end(), gen);

        std::vector<unsigned long> pos = frame_batch.pos;
        for (int i = 0; i < sample_indices.size(); ++i) {
            frame_batch.pos[i] = pos[sample_indices.at(i)];
        }
    }
}

void learning_env::run()
{
    int nsample = 0;

    while (nsample < frame_batch.pos.size() - 2) {
        autodiff::computation_graph comp_graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(comp_graph, param);

        std::shared_ptr<lstm::transcriber> transcriber;

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<std::vector<double>> seg1 = sample_seg(frames, gen);
        ++nsample;
        auto seg1_op = to_op(seg1, comp_graph);
        auto e1 = embed(seg1_op, layer, var_tree);
        std::cout << "seg 1: " << seg1.size() << std::endl;

        frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<std::vector<double>> seg2 = sample_seg(frames, gen);
        ++nsample;
        auto seg2_op = to_op(seg2, comp_graph);
        auto e2 = embed(seg2_op, layer, var_tree);
        std::cout << "seg 2: " << seg2.size() << std::endl;

        frames = speech::load_frame_batch(frame_batch.at(nsample));
        std::vector<std::vector<double>> seg3 = sample_seg(frames, gen);
        ++nsample;
        auto seg3_op = to_op(seg3, comp_graph);
        auto e3 = embed(seg3_op, layer, var_tree);
        std::cout << "seg 3: " << seg3.size() << std::endl;

        double d12 = dtw::dtw(seg1, seg2);
        double d13 = dtw::dtw(seg1, seg3);

        std::shared_ptr<autodiff::op_t> near;
        std::shared_ptr<autodiff::op_t> far;

        double loss;

        if (d12 < d13) {
            near = autodiff::norm(autodiff::sub(e1, e2));
            far = autodiff::norm(autodiff::sub(e1, e3));

            loss = std::max<double>(0.0, (d13 - d12)
                - autodiff::get_output<double>(near) + autodiff::get_output<double>(far));
        } else {
            near = autodiff::norm(autodiff::sub(e1, e3));
            far = autodiff::norm(autodiff::sub(e1, e2));

            loss = std::max<double>(0.0, (d12 - d13)
                - autodiff::get_output<double>(near) + autodiff::get_output<double>(far));
        }

        std::cout << "loss: " << loss << std::endl;

        if (loss > 0) {
            near->grad = std::make_shared<double>(-1);
            far->grad = std::make_shared<double>(1);

            auto topo_order = autodiff::natural_topo_order(comp_graph);
            autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

            auto grad = make_tensor_tree(layer);
            tensor_tree::copy_grad(grad, var_tree);

            double n = tensor_tree::norm(grad);

            std::cout << "grad norm: " << n << std::endl;

            if (ebt::in(std::string("clip"), args)) {
                if (n > clip) {
                    tensor_tree::imul(grad, clip / n);
                    std::cout << "gradient clipped" << std::endl;
                }
            }

            auto vars = tensor_tree::leaves_pre_order(param);
            la::tensor<double> const& v = tensor_tree::get_tensor(vars[2]);

            double v1 = v.data()[0];

            opt->update(grad);

            double v2 = v.data()[0];

            std::cout << "weight: " << v1 << " update: " << v2 - v1
                << " rate: " << (v2 - v1) / v1 << std::endl;

        }

        std::cout << "norm: " << tensor_tree::norm(param) << std::endl;

        std::cout << std::endl;
    }

    std::ofstream param_ofs { args.at("output-param") };
    param_ofs << layer << std::endl;
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { args.at("output-opt-data") };
    opt_data_ofs << layer << std::endl;
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();

}

std::vector<std::vector<double>>
sample_seg(std::vector<std::vector<double>> const& frames,
    std::default_random_engine& gen)
{
    std::uniform_int_distribution<int> start_dist {0, int(frames.size() - 2)};

    int start_time = start_dist(gen);

    std::uniform_int_distribution<int> dur_dist {1, 15};

    int dur = dur_dist(gen);

    int end_time = std::min<int>(start_time + dur * 4, frames.size() - 1);

    std::cout << "start: " << start_time << " end: " << end_time << std::endl;

    std::vector<std::vector<double>> seg_frames;

    for (int i = start_time; i < end_time; ++i) {
        seg_frames.push_back(frames.at(i));
    }

    return seg_frames;
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

