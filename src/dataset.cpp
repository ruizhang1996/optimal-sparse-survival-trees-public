#include "dataset.hpp"

Dataset::Dataset(void) {}
Dataset::~Dataset(void) {}

Dataset::Dataset(std::istream & data_source) { load(data_source); }

// Loads the binary-encoded data set into precomputed form:
void Dataset::load(std::istream & data_source) {
    // Construct all rows, features in binary form
    construct_bitmasks(data_source);
    // Normalize target column
    normalize_data();
    // Build cluster and cluster target indicating whether a point is the equivalent set

    // Compute inverse probability of censoring weights
    compute_ipcw(this -> inverse_prob_censoring_weights);

    if (Configuration::verbose) {
        std::cout << "Dataset Dimensions: " << height() << " x " << width() << " x " << depth() << std::endl;
    }
    return;
}

void Dataset::clear(void) {
    this -> features.clear();
    this -> targets.clear();
    this -> rows.clear();
    this -> feature_rows.clear();
}

void Dataset::construct_bitmasks(std::istream & data_source) {
    this -> encoder = Encoder(data_source);
    std::vector< Bitmask > rows = this -> encoder.read_binary_rows();
    unsigned int number_of_samples = this -> encoder.samples(); // Number of samples in the dataset
    unsigned int number_of_binary_features = this -> encoder.binary_features(); // Number of source features
    // unsigned int number_of_binary_targets = this -> encoder.binary_targets(); // Number of target features
    this -> _size = number_of_samples;
    // this -> weights = encoder.get_weights();
    this -> rows = this -> encoder.read_binary_rows();

    this -> features.resize(number_of_binary_features, number_of_samples);
    this -> feature_rows.resize(number_of_samples, number_of_binary_features);
    // Sort samples based on target if L1 loss
    std::vector<double> targets = encoder.read_numerical_targets();
    this -> targets = targets;
    auto compi = [targets](size_t i, size_t j) {
        return targets[i] < targets[j];
    };
    this -> censoring = Bitmask(number_of_samples);
    // must sort on target
    std::vector<int> target_order(number_of_samples);
    std::iota(target_order.begin(), target_order.end(), 0);
    std::sort(target_order.begin(), target_order.end(), compi);
    std::vector<double> sorted_targets(number_of_samples);
    // std::vector<double> sorted_weights(number_of_samples);
    std::vector<Bitmask> sorted_rows(number_of_samples);
    for (int i = 0; i < target_order.size(); i++) {
        sorted_targets[i] = targets[target_order[i]];
        // sorted_weights[i] = weights[target_order[i]];
        sorted_rows[i] = rows[target_order[i]];
    }
    this -> targets = sorted_targets;
    // this -> weights = sorted_weights;
    this -> rows = sorted_rows;
    // unique target values
    std::set< std::string > target_string_values = this -> encoder.get_target_values();
    for (auto it = target_string_values.begin(); it != target_string_values.end(); ++it){
        this -> target_values.emplace_back(atof((*it).c_str()));
    }
    std::sort(target_values.begin(), target_values.end());
    int k = 0;
    this -> targets_mapping.resize(number_of_samples);
    for (unsigned int i = 0; i < number_of_samples; ++i) {
        // add for survival trees, binary target of censoring
        this -> censoring.set(i, bool(this -> rows[i][number_of_binary_features]));
        // map target to value index
        if (this -> targets[i] != this -> target_values[k]){
            k += 1;
        }
        this -> targets_mapping[i] = k;
        for (unsigned int j = 0; j < number_of_binary_features; ++j) {
            this -> features[j].set(i, bool(this -> rows[i][j]));
            this -> feature_rows[i].set(j, bool(this -> rows[i][j]));
        }
    }

    this -> shape = std::tuple< int, int, int >(this -> rows.size(), this -> features.size(), this -> target_values.size());
};


// TODO: investigate 
float Dataset::distance(Bitmask const & set, unsigned int i, unsigned int j, unsigned int id) const {
    return 0;
}


void Dataset::target_value(Bitmask capture_set, std::string & prediction_value) const{
    prediction_value = std::to_string(0);
}



double Dataset::compute_ibs(Bitmask capture_set) const{
    int max = capture_set.size();
    std::vector<double> S(target_values.size(), 1);
    int i = capture_set.scan(0, true);
    double prod = 1.0;
    unsigned int number_of_death = 0;
    unsigned int number_of_known_alive = capture_set.count();
    unsigned int number_of_sample_current_time = 0;
    while (i < max){
        int next = capture_set.scan(i + 1, true);
        number_of_death += censoring.get(i);
        number_of_sample_current_time += 1;
        // if i is the last captured point
        if (next >= max){
            prod *= (1 - (float) number_of_death / (float) number_of_known_alive);
            for (int j = targets_mapping[i]; j < target_values.size(); ++j) {
                S[j] = prod;
            }
        }
        else if (targets_mapping[i] != targets_mapping[next]){
            prod *= (1 - (float) number_of_death / (float) number_of_known_alive);
            for (int j = targets_mapping[i]; j < targets_mapping[next]; ++j) {
                S[j] = prod;
            }
            number_of_known_alive -= number_of_sample_current_time;
            number_of_sample_current_time = 0;
            number_of_death = 0;
        }
        i = next;
    }
    double ibs = 0;
//    for (int j = capture_set.scan(0, true); j< max; j = capture_set.scan(j + 1, true)) {
//        for (int k = 0; k < target_values.size() - 1; ++k) {
//            if (target_values[k] < targets[j]){
//                ibs += pow(S[k] - 1, 2) * inverse_prob_censoring_weights[k] * (target_values[k + 1] - target_values[k]);
//            } else if (censoring.get(j)){
//                ibs += pow(S[k], 2) * inverse_prob_censoring_weights[targets_mapping[j]] * (target_values[k + 1] - target_values[k]);
//            }
//        }
//    }
    int j = capture_set.scan(0, true);
    int prev_j = 0;
    int num_included = 0;
    while (j < max){
        for (int k = targets_mapping[prev_j]; k < targets_mapping[j]; ++k) {
            ibs += pow(S[k] - 1, 2) * inverse_prob_censoring_weights[k] * (target_values[k + 1] - target_values[k]) * (capture_set.count() - num_included);
        }
        if (censoring.get(j)){
            for (int k = targets_mapping[j]; k < target_values.size() - 1; ++k) {
                ibs += pow(S[k], 2) * inverse_prob_censoring_weights[targets_mapping[j]] * (target_values[k + 1] - target_values[k]);
            }
        }
        num_included ++;
        prev_j = j;
        j = capture_set.scan(j + 1, true);
    }
//    if (std::abs(ibs - new_ibs) >= std::numeric_limits<float>::epsilon()){
//        std::cout << ibs << ", " << new_ibs << std::endl;
//    }

    return ibs / size();
}
double Dataset::compute_ibs(std::vector< int > capture_set_idx) const{
    return 0;
}
void Dataset::normalize_data() {
    // largest target
    double loss_normalizer;
    loss_normalizer = this -> targets[size() - 1];

    for (int i = 0; i < size(); i++) {
        targets[i] = targets[i] / loss_normalizer;
    }
    for (int i = 0; i < this -> target_values.size(); ++i) {
        target_values[i] = target_values[i] / loss_normalizer;
    }
    this -> _normalizer = loss_normalizer;

    std::cout << "loss_normalizer: " << loss_normalizer << std::endl;
}



// @param feature_index: selects the feature on which to split
// @param positive: determines whether to provide the subset that tests positive on the feature or tests negative on the feature
// @param set: pointer to bit blocks which indicate the original set before splitting
// @modifies set: set will be modified to indicate the positive or negative subset after splitting
// @notes the set in question is an array of the type bitblock. this allows us to specify the set using a stack-allocated array
void Dataset::subset(unsigned int feature_index, bool positive, Bitmask & set) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(set, !positive);
    if (Configuration::depth_budget != 0){ set.set_depth_budget(set.get_depth_budget()-1);} //subproblems have one less depth_budget than their parent
}

void Dataset::subset(unsigned int feature_index, Bitmask & negative, Bitmask & positive) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(negative, true);
    this -> features[feature_index].bit_and(positive, false);
    if (Configuration::depth_budget != 0){
        negative.set_depth_budget(negative.get_depth_budget()-1);
        positive.set_depth_budget(positive.get_depth_budget()-1);
    } //subproblems have one less depth_budget than their parent
}

// Performance Boost ideas:
// 1. Store everything in summary 
// 2. Introduce scope and apply it to kmeans so that it could be even tighter 
// 3. Check equiv (points lower bound + 2 * reg) before using Kmeans to
//    determine if we need more split as it has a way lower overhead

void Dataset::summary(Bitmask const & capture_set, float & info, float & potential, float & min_loss, float & max_loss, unsigned int & target_index, unsigned int id) const {
    summary_calls++;
    Bitmask & buffer = State::locals[id].columns[0];
    //unsigned int * distribution; // The frequencies of each class
    //distribution = (unsigned int *) alloca(sizeof(unsigned int) * depth());

    unsigned int cost_minimizer = 0;

    //max_loss = compute_ibs(capture_set);
    //float max_cost_reduction = 0.0;
    float equivalent_point_loss = 0.0;
    //float support = (float)(capture_set.count()) / (float)(height());
    float information = 0.0;
    
    // if (summary_calls > 30000) {
    //     equivalent_point_loss = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    // } else {
    //     equivalent_point_loss = compute_kmeans_lower_bound(capture_set);
    // }
    // assert(min_cost + Configuration::regularization < equivalent_point_loss_1 || equivalent_point_loss_1 < equivalent_point_loss);
    // equivalent_point_loss = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    ibs_accessor stored_ibs_accessor;
    if (State::graph.ibs.find(stored_ibs_accessor, capture_set)) {
        max_loss = stored_ibs_accessor->second;
        stored_ibs_accessor.release();
    } else {
        max_loss = compute_ibs(capture_set);
        auto new_ibs = std::make_pair(capture_set, max_loss);
        State::graph.ibs.insert(new_ibs);
        compute_ibs_calls++;
    }

    // float equivalent_point_loss_1 = 2 * Configuration::regularization + compute_equivalent_points_lower_bound(capture_set);
    // float max_loss_1 = min_cost + Configuration::regularization;
    // float diff = equivalent_point_loss - equivalent_point_loss_1;

    // float gap = max_loss_1 - equivalent_point_loss_1;
    // if (gap > 0.0001) {
    //     float percent = diff / gap;
    //     summary_calls_has_gap++;
    //     cum_percent += percent;
        
    // }

    min_loss = 0;
    potential = max_loss;
    info = information;
    target_index = cost_minimizer;
}

// Assume that data is already of the right size
void Dataset::tile(Bitmask const & capture_set, Bitmask const & feature_set, Tile & tile, std::vector< int > & order, unsigned int id) const {
    tile.content() = capture_set;
    tile.width(0);
    return;
}


unsigned int Dataset::height(void) const {
    return std::get<0>(this -> shape);
}

unsigned int Dataset::width(void) const {
    return std::get<1>(this -> shape);
}

unsigned int Dataset::depth(void) const {
    return std::get<2>(this -> shape);
}

unsigned int Dataset::size(void) const {
    return this -> _size;
}

bool Dataset::index_comparator(const std::pair< unsigned int, unsigned int > & left, const std::pair< unsigned int, unsigned int > & right) {
    return left.second < right.second;
}

void Dataset::compute_ipcw(std::vector<double> & ipcw){
    ipcw.resize(target_values.size(), -1);
    std::vector<int> number_of_death(this-> target_values.size(), -1);
    std::vector<int> number_of_known_alive(this-> target_values.size(), -1);
    double prod = 1.0;
    for (int i = 0; i < size(); ++i) {
        if (number_of_known_alive[targets_mapping[i]] == -1){
            if (i > 0) {
                prod *= (1 - (double) number_of_death[targets_mapping[i-1]] / (double) number_of_known_alive[targets_mapping[i-1]]);
                if (prod > 0) {ipcw[targets_mapping[i-1]] = 1 / prod;}
                else {ipcw[targets_mapping[i-1]] = 0;}
            }
            number_of_known_alive[targets_mapping[i]] = size() - i;
            number_of_death[targets_mapping[i]] = 0;
        }
        if (censoring.get(i) < 1){number_of_death[targets_mapping[i]] += 1;}
    }
    prod *= (1 - (double) number_of_death[targets_mapping[size() - 1]] / (double) number_of_known_alive[targets_mapping[size() - 1]]);
    if (prod > 0){ipcw[targets_mapping[size() - 1]] = 1 / prod;}
    else{ipcw[targets_mapping[size() - 1]] = 0;}
}
