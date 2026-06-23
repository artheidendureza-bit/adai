#include "adaicpp/adaicpp.h"
#include <gtest/gtest.h>
#include <memory>

using namespace adaicpp;

class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        initialize();
        device_ = DeviceManager::instance().get_cpu_device();
        graph_ = std::make_unique<Graph>();
    }
    
    std::shared_ptr<Device> device_;
    std::unique_ptr<Graph> graph_;
};

TEST_F(GraphTest, AddPlaceholder) {
    auto placeholder = graph_->add_placeholder("x", device_);
    
    EXPECT_NE(placeholder, nullptr);
    EXPECT_EQ(placeholder->name(), "x");
    EXPECT_EQ(placeholder->type(), NodeType::Placeholder);
    EXPECT_FALSE(placeholder->is_static());
}

TEST_F(GraphTest, AddParameter) {
    auto value = Tensor::ones({2, 3}, device_);
    auto parameter = graph_->add_parameter("w", std::make_shared<Tensor>(value));
    
    EXPECT_NE(parameter, nullptr);
    EXPECT_EQ(parameter->name(), "w");
    EXPECT_EQ(parameter->type(), NodeType::Parameter);
    EXPECT_TRUE(parameter->is_static());
}

TEST_F(GraphTest, AddOperation) {
    auto x = graph_->add_placeholder("x", device_);
    auto y = graph_->add_placeholder("y", device_);
    
    auto op = graph_->add_op("add", {x, y},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return add(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    EXPECT_NE(op, nullptr);
    EXPECT_EQ(op->name(), "add");
    EXPECT_EQ(op->type(), NodeType::Operation);
    EXPECT_TRUE(op->is_static());
}

TEST_F(GraphTest, TopologicalSort) {
    auto x = graph_->add_placeholder("x", device_);
    auto w = graph_->add_parameter("w", std::make_shared<Tensor>(Tensor::ones({2, 2}, device_)));
    
    auto op1 = graph_->add_op("mul1", {x, w},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return mul(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    auto op2 = graph_->add_op("relu1", {op1},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return relu(*inputs[0]);
        },
        nullptr, true, device_);
    
    graph_->compile();
    
    const auto& sorted = graph_->sorted_nodes();
    EXPECT_EQ(sorted.size(), 4);
    
    // x and w should come before op1, which should come before op2
    size_t x_idx = 0, w_idx = 0, op1_idx = 0, op2_idx = 0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i]->name() == "x") x_idx = i;
        if (sorted[i]->name() == "w") w_idx = i;
        if (sorted[i]->name() == "mul1") op1_idx = i;
        if (sorted[i]->name() == "relu1") op2_idx = i;
    }
    
    EXPECT_LT(op1_idx, op2_idx);
}

TEST_F(GraphTest, ForwardPass) {
    auto x = graph_->add_placeholder("x", device_);
    auto w = graph_->add_parameter("w", std::make_shared<Tensor>(Tensor::ones({2, 2}, device_) * 2.0f));
    
    auto op = graph_->add_op("mul", {x, w},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return mul(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    graph_->compile();
    
    auto input = Tensor::ones({2, 2}, device_);
    std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
    feeds["x"] = std::make_shared<Tensor>(input);
    
    graph_->forward(feeds);
    
    auto output = graph_->get_value("mul");
    ASSERT_NE(output, nullptr);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = output->data();
        for (size_t i = 0; i < output->size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 2.0f);
        }
    }
}

TEST_F(GraphTest, RemoveNode) {
    auto x = graph_->add_placeholder("x", device_);
    EXPECT_TRUE(graph_->has_node("x"));
    
    graph_->remove_node("x");
    EXPECT_FALSE(graph_->has_node("x"));
}

TEST_F(GraphTest, GetNode) {
    auto x = graph_->add_placeholder("x", device_);
    auto retrieved = graph_->get_node("x");
    
    EXPECT_EQ(retrieved->name(), "x");
}

TEST_F(GraphTest, DeadCodeElimination) {
    auto x = graph_->add_placeholder("x", device_);
    auto y = graph_->add_placeholder("y", device_);
    
    auto op1 = graph_->add_op("add1", {x, y},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return add(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    auto op2 = graph_->add_op("mul1", {x, y},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return mul(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    // Only keep op1 as output
    graph_->optimize({"add1"});
    
    EXPECT_TRUE(graph_->has_node("x"));
    EXPECT_TRUE(graph_->has_node("y"));
    EXPECT_TRUE(graph_->has_node("add1"));
    EXPECT_FALSE(graph_->has_node("mul1"));  // Should be removed
}

TEST_F(GraphTest, ConstantFolding) {
    auto w1 = graph_->add_parameter("w1", std::make_shared<Tensor>(Tensor::ones({2, 2}, device_) * 2.0f));
    auto w2 = graph_->add_parameter("w2", std::make_shared<Tensor>(Tensor::ones({2, 2}, device_) * 3.0f));
    
    auto op = graph_->add_op("mul", {w1, w2},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return mul(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    graph_->optimize({"mul"});
    
    // After constant folding, the operation should be replaced with a parameter
    auto node = graph_->get_node("mul");
    EXPECT_EQ(node->type(), NodeType::Parameter);
}

TEST_F(GraphTest, SetInputs) {
    auto x = graph_->add_placeholder("x", device_);
    auto y = graph_->add_placeholder("y", device_);
    
    auto op = graph_->add_op("add", {x, y},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return add(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    auto z = graph_->add_placeholder("z", device_);
    graph_->set_inputs("add", {x, z});
    
    EXPECT_EQ(op->inputs().size(), 2);
    EXPECT_EQ(op->inputs()[1]->name(), "z");
}

TEST_F(GraphTest, ZeroGrad) {
    auto x = graph_->add_placeholder("x", device_);
    auto w = graph_->add_parameter("w", std::make_shared<Tensor>(Tensor::ones({2, 2}, device_)));
    
    auto op = graph_->add_op("mul", {x, w},
        [](const std::vector<std::shared_ptr<Tensor>>& inputs) {
            return mul(*inputs[0], *inputs[1]);
        },
        nullptr, true, device_);
    
    graph_->compile();
    
    auto input = Tensor::ones({2, 2}, device_);
    std::unordered_map<std::string, std::shared_ptr<Tensor>> feeds;
    feeds["x"] = std::make_shared<Tensor>(input);
    
    graph_->forward(feeds);
    graph_->backward("mul");
    
    // Check that gradients are computed
    EXPECT_NE(w->grad(), nullptr);
    
    // Zero gradients
    graph_->zero_grad();
    
    // Check that gradients are cleared
    EXPECT_EQ(w->grad(), nullptr);
}
