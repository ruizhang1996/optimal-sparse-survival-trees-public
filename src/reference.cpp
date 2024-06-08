#include "reference.hpp"
#include <functional>
#include <vector>

std::vector<float> Reference::labels = std::vector<float>();

void Reference::initialize_labels(std::istream & labels){
    //read loss
    Encoder encoder(labels);
    Reference::labels = encoder.read_numerical_targets();

};

void Reference::normalize_labels(float loss_normalizer) {
    std::vector<float> &labels = Reference::labels;
    for (int i = 0; i < labels.size(); i++) {
        labels[i] = labels[i] / loss_normalizer;
    }
}
