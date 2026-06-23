#include "adaicpp/adaicpp.h"
#include <gtest/gtest.h>
#include <memory>

using namespace adaicpp;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = DeviceManager::instance().get_cpu_device();
    }
    
    std::shared_ptr<Device> device_;
};

TEST_F(TensorTest, DefaultConstruction) {
    Tensor t;
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.ndim(), 0);
}

TEST_F(TensorTest, ShapeConstruction) {
    std::vector<size_t> shape = {2, 3, 4};
    Tensor t(shape, device_);
    
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.size(), 24);
}

TEST_F(TensorTest, FillConstruction) {
    std::vector<size_t> shape = {2, 3};
    Tensor t(shape, 5.0f, device_);
    
    EXPECT_EQ(t.shape(), shape);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 5.0f);
        }
    }
}

TEST_F(TensorTest, Zeros) {
    std::vector<size_t> shape = {3, 4};
    Tensor t = Tensor::zeros(shape, device_);
    
    EXPECT_EQ(t.shape(), shape);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 0.0f);
        }
    }
}

TEST_F(TensorTest, Ones) {
    std::vector<size_t> shape = {2, 2};
    Tensor t = Tensor::ones(shape, device_);
    
    EXPECT_EQ(t.shape(), shape);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 1.0f);
        }
    }
}

TEST_F(TensorTest, Add) {
    Tensor a({2, 2}, 1.0f, device_);
    Tensor b({2, 2}, 2.0f, device_);
    
    Tensor c = add(a, b);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = c.data();
        for (size_t i = 0; i < c.size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 3.0f);
        }
    }
}

TEST_F(TensorTest, Mul) {
    Tensor a({2, 2}, 3.0f, device_);
    Tensor b({2, 2}, 4.0f, device_);
    
    Tensor c = mul(a, b);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = c.data();
        for (size_t i = 0; i < c.size(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 12.0f);
        }
    }
}

TEST_F(TensorTest, ReLU) {
    Tensor t({5}, device_);
    if (device_->type() == DeviceType::CPU) {
        float* data = t.data();
        data[0] = -2.0f;
        data[1] = -1.0f;
        data[2] = 0.0f;
        data[3] = 1.0f;
        data[4] = 2.0f;
    }
    
    Tensor result = relu(t);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = result.data();
        EXPECT_FLOAT_EQ(data[0], 0.0f);
        EXPECT_FLOAT_EQ(data[1], 0.0f);
        EXPECT_FLOAT_EQ(data[2], 0.0f);
        EXPECT_FLOAT_EQ(data[3], 1.0f);
        EXPECT_FLOAT_EQ(data[4], 2.0f);
    }
}

TEST_F(TensorTest, Sigmoid) {
    Tensor t({1}, 0.0f, device_);
    Tensor result = sigmoid(t);
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = result.data();
        EXPECT_FLOAT_EQ(data[0], 0.5f);
    }
}

TEST_F(TensorTest, MatMul) {
    Tensor a({2, 3}, device_);
    Tensor b({3, 2}, device_);
    
    if (device_->type() == DeviceType::CPU) {
        float* a_data = a.data();
        float* b_data = b.data();
        
        // Initialize a
        a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
        a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;
        
        // Initialize b
        b_data[0] = 7.0f; b_data[1] = 8.0f;
        b_data[2] = 9.0f; b_data[3] = 10.0f;
        b_data[4] = 11.0f; b_data[5] = 12.0f;
    }
    
    Tensor c = matmul(a, b);
    
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2}));
    
    if (device_->type() == DeviceType::CPU) {
        const float* data = c.data();
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
        EXPECT_FLOAT_EQ(data[0], 58.0f);
        EXPECT_FLOAT_EQ(data[1], 64.0f);
        EXPECT_FLOAT_EQ(data[2], 139.0f);
        EXPECT_FLOAT_EQ(data[3], 154.0f);
    }
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 3}, device_);
    Tensor reshaped = t.reshape({3, 2});
    
    EXPECT_EQ(reshaped.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(reshaped.size(), t.size());
}

TEST_F(TensorTest, CopyFrom) {
    Tensor a({2, 2}, 5.0f, device_);
    Tensor b({2, 2}, device_);
    
    b.copy_from(a);
    
    if (device_->type() == DeviceType::CPU) {
        const float* a_data = a.data();
        const float* b_data = b.data();
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_FLOAT_EQ(a_data[i], b_data[i]);
        }
    }
}
