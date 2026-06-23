#include "adaicpp/adaicpp.h"
#include <iostream>

using namespace adaicpp;

int main() {
    try {
        std::cout << "Test 1: Initialize library" << std::endl;
        initialize();
        std::cout << "Library initialized" << std::endl;
        
        std::cout << "\nTest 2: Create tensor" << std::endl;
        auto t = Tensor::ones({2, 3});
        std::cout << "Tensor created, shape: [";
        for (size_t dim : t.shape()) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "\nTest 3: Tensor operations" << std::endl;
        auto a = Tensor::ones({2, 2});
        auto b = scalar_mul(Tensor::ones({2, 2}), 2.0f);
        auto c = add(a, b);
        std::cout << "Addition successful" << std::endl;
        
        std::cout << "\nTest 4: Create graph" << std::endl;
        auto graph = std::make_unique<Graph>();
        std::cout << "Graph created" << std::endl;
        
        std::cout << "\nTest 5: Add placeholder" << std::endl;
        auto x = graph->add_placeholder("x");
        std::cout << "Placeholder added" << std::endl;
        
        std::cout << "\nTest 6: Add parameter" << std::endl;
        auto w = graph->add_parameter("w", std::make_shared<Tensor>(Tensor::ones({2, 3})));
        std::cout << "Parameter added" << std::endl;
        
        std::cout << "\nTest 7: Compile graph" << std::endl;
        graph->compile();
        std::cout << "Graph compiled" << std::endl;
        
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
