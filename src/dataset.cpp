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
    construct_clusters();
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
    if ( (float) target_values.size() / this -> targets.size() > 0.1){
        unsigned int num_bucket;
        if (Configuration::number_of_buckets > 0) {
            num_bucket = Configuration::number_of_buckets;
        } else {
            num_bucket = std::round(0.1 * this -> targets.size());
        }
        double bucket_width  = (this -> target_values.back() - this -> target_values[0]) / num_bucket;
        double max_value = target_values.back();
        this -> target_values.resize(1);

        for (int i = 0; i < num_bucket - 1; ++i) {
            this -> target_values.emplace_back(this -> target_values.back() + bucket_width);
        }
        assert(target_values.back() < max_value);
        this -> target_values.emplace_back(max_value);
        int target_value_idx = 0;
        for (int i = 0; i < number_of_samples; ++i) {
            if (target_value_idx < this -> target_values.size() - 1 && this -> target_values[target_value_idx + 1] <= this -> targets[i]){
                target_value_idx ++;
            }
            assert(this -> targets[i] >= this -> target_values[target_value_idx]);
            this -> targets[i] = this -> target_values[target_value_idx];
        }
    }
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

void Dataset::construct_clusters(void) {
    std::vector< Bitmask > keys(height(), width());
    for (unsigned int i = 0; i < height(); ++i) {
        for (unsigned int j = 0; j < width(); ++j) {
            keys[i].set(j, bool(this -> rows[i][j]));
        }
    }

    // Step 1: Construct a map from the binary features to their clusters,
    // indicated by their indices in capture set
    std::unordered_map< Bitmask, std::vector< int > > clusters;
    for (int i = 0; i < height(); ++i) {
        Bitmask const & key = keys.at(i);
        clusters[key].emplace_back(i);
    }

    // Step 2: Convert clusters map into an array by taking the mean of each
    // cluster, initialize unsorted order, and initialize data index to cluster
    // index mapping
    std::vector< std::vector<int> > cluster_indices; // points index in this cluster
    std::vector< int > clustered_mapping(size());
    // std::vector<double> cluster_weights;
    int cluster_idx = 0;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
        std::vector< int > const & cluster = it -> second;
        std::vector< int > indices;
        for (int idx : cluster) {
            indices.emplace_back(idx);
            clustered_mapping[idx] = cluster_idx;
        }
        std::sort(indices.begin(), indices.end());
        cluster_indices.emplace_back(indices);
        cluster_idx ++;
    }



    this -> cluster_indices = cluster_indices;
    this -> clustered_mapping = clustered_mapping;

}

float Dataset::distance(Bitmask const & set, unsigned int i, unsigned int j, unsigned int id) const {
    return 0;
}


void Dataset::target_value(Bitmask capture_set, std::string & prediction_value) const{
    prediction_value = std::to_string(0);
}



double Dataset::compute_ibs(Bitmask capture_set, std::vector<int> & cumulative_death_per_target_values, std::vector<int> & num_death_per_target_values, std::vector<double> & survival_function) const{
    int max = capture_set.size();
    std::vector<double> S(target_values.size(), 1);
    int i = capture_set.scan(0, true);
    double prod = 1.0;
    unsigned int number_of_death = 0;
    unsigned int number_of_known_alive = capture_set.count();
    unsigned int number_of_sample_current_time = 0;
//    num_death_per_target_values.resize(target_values.size(), 0);
//    cumulative_death_per_target_values.resize(target_values.size(), 0);
//    int cumulative_death = 0;
    while (i < max){
        int next = capture_set.scan(i + 1, true);
        number_of_death += censoring.get(i);
        number_of_sample_current_time += 1;
        // if i is the last captured point
        if (next >= max){
            prod *= (1 - (float) number_of_death / (float) number_of_known_alive);
//            num_death_per_target_values[targets_mapping[i]] = number_of_death;
//            cumulative_death += number_of_death;
            for (int j = targets_mapping[i]; j < target_values.size(); ++j) {
                S[j] = prod;
//                cumulative_death_per_target_values[j] = cumulative_death;
            }
        }
        else if (targets_mapping[i] != targets_mapping[next]){
            prod *= (1 - (float) number_of_death / (float) number_of_known_alive);
//            num_death_per_target_values[targets_mapping[i]] = number_of_death;
//            cumulative_death += number_of_death;
            for (int j = targets_mapping[i]; j < targets_mapping[next]; ++j) {
                S[j] = prod;
//                cumulative_death_per_target_values[j] = cumulative_death;
            }
            number_of_known_alive -= number_of_sample_current_time;
            number_of_sample_current_time = 0;
            number_of_death = 0;
        }
        i = next;
    }
    double ibs = 0.0;
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
    double prev_ibs_right;
    bool calculated = false;
    while (j < max){
        for (int k = targets_mapping[prev_j]; k < targets_mapping[j]; ++k) {
            ibs += (S[k] - 1) * (S[k] - 1) * inverse_prob_censoring_weights[k] * (target_values[k + 1] - target_values[k]) * (capture_set.count() - num_included);
        }
        if (targets_mapping[prev_j] != targets_mapping[j]) calculated = false;
        if (censoring.get(j)){
            if (targets_mapping[prev_j] == targets_mapping[j] && calculated){
                ibs += prev_ibs_right;
            } else{
                double tmp = 0.0;
                for (int k = targets_mapping[j]; k < target_values.size() - 1; ++k) {
                    tmp += S[k] * S[k] * inverse_prob_censoring_weights[targets_mapping[j]] * (target_values[k + 1] - target_values[k]);
                }
                ibs += tmp;
                prev_ibs_right = tmp;
                calculated = true;
            }
        }
        num_included ++;
        prev_j = j;
        j = capture_set.scan(j + 1, true);
    }
//    if (std::abs(ibs - new_ibs) >= std::numeric_limits<float>::epsilon()){
//        std::cout << ibs << ", " << new_ibs << std::endl;
//    }
    survival_function = S;
    return ibs / size();
}
double Dataset::compute_lowerbound(Bitmask capture_set, std::vector<int> cumulative_death_per_target_values, std::vector<int> num_death_per_target_values, std::vector<double> S) const{
    int max = capture_set.size();


    double res = 0;
//    std::vector< int > count(cluster_indices.size());
//    for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
//        std::vector<int> cluster = cluster_indices[clustered_mapping[i]];
//        if (cluster.size() > 1 && count[clustered_mapping[i]] == 0) {
//            // start compute lower bound for this equivalent set
//            std::vector<double> S_low;
//            std::vector<double> S_high;
//            int S_idx = 0; // index to compute S
//            double prod = 1.0;
//            double prod_high = 1.0;
//            for (int j = 0; j < cluster.size(); ++j) {
//                while (S_idx < targets_mapping[cluster[j]]){
//                    int cumulative_before;
//                    int cumulative_total;
//                    if (targets_mapping[cluster[j]] > 0){ cumulative_total = cumulative_death_per_target_values[targets_mapping[cluster[j]] - 1];}
//                    else {cumulative_total = 0;}
//                    if (S_idx > 0) {cumulative_before = cumulative_death_per_target_values[S_idx - 1];}
//                    else {cumulative_before = 0;}
//                    prod *= (1 - num_death_per_target_values[S_idx] / (cumulative_total + (int) cluster.size() - j - cumulative_before));
//                    S_low.emplace_back(prod);
//                    S_high.emplace_back(prod_high);
//                    S_idx ++;
//                }
//                // now S_idx == targets_mapping[cluster[j]]:
//                if (S_idx == targets_mapping[cluster[j]]){
//                    prod *= (1 - num_death_per_target_values[S_idx] / ((int) cluster.size() - j + num_death_per_target_values[S_idx]));
//                    S_low.emplace_back(prod);
//                    prod_high += (1 - 1/ (size() - cluster[j]));
//                    S_high.emplace_back(prod_high);
//                    S_idx++;
//                }
//            }
//            for (int j = 0; j < cluster.size(); ++j) {
//
//                int S_high_idx = targets_mapping[cluster[0]];
//                while (S_high_idx < targets_mapping[cluster[j]]){
//                    res += pow(S_high[S_high_idx] - 1, 2) * inverse_prob_censoring_weights[S_high_idx] * (target_values[S_high_idx + 1] - target_values[S_high_idx]);
//                    S_high_idx ++;
//                }
//
//                if (censoring.get(cluster[j])){
//                    int S_low_idx = targets_mapping[cluster[j]];
//                    while (S_low_idx < targets_mapping[cluster[cluster.size() - 1] - 1]){
//                        res += pow(S_low[S_low_idx], 2) * inverse_prob_censoring_weights[S_low_idx] * (target_values[S_low_idx + 1] - target_values[S_low_idx]);
//                        S_low_idx ++;
//                    }
//                }
//            }
//        }
//        count[clustered_mapping[i]]++;
//    }

    return res / size();
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

void Dataset::summary(Bitmask const & capture_set, float & info, float & potential, float & min_loss, float & guaranteed_min_loss, float & max_loss, unsigned int & target_index, unsigned int id) const {
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
    lb_accessor stored_lb_accessor;
    guaranteed_min_loss = 0;

    if (State::graph.ibs.find(stored_ibs_accessor, capture_set)) {
        max_loss = stored_ibs_accessor->second;
        if (Configuration::reference_LB){
            State::graph.lb.find(stored_lb_accessor, capture_set);
            min_loss = stored_lb_accessor -> second;
            stored_lb_accessor.release();
        } else{
            min_loss = guaranteed_min_loss;
        }
        stored_ibs_accessor.release();
    } else {
        std::vector<int> num_death_per_target_values;
        std::vector<int> cumulative_death_per_target_values;
        std::vector<double> S;
        max_loss = compute_ibs(capture_set, cumulative_death_per_target_values, num_death_per_target_values, S);
//        min_loss = compute_lowerbound(capture_set, cumulative_death_per_target_values, num_death_per_target_values, S);
//        std::cout << max_loss << min_loss <<std::endl;
        if (Configuration::reference_LB){
            //calculate reference model's error on this capture set, use as estimate for min_loss (possible overestimate)
            float reference_model_loss = 0.0;
            int max = capture_set.size();
            for (int i = capture_set.scan(0, true); i < max; i = capture_set.scan(i + 1, true)) {
                reference_model_loss += Reference::labels[i];
            }
            min_loss = reference_model_loss;
        } else {
            // when not using a reference model, we do not want min_loss to be an overestimate
            // so we set min_loss to match guaranteed_min_loss
            min_loss = guaranteed_min_loss;
        }
        auto new_ibs = std::make_pair(capture_set, max_loss);
        State::graph.ibs.insert(new_ibs);
        if (Configuration::reference_LB){
            auto new_lb = std::make_pair(capture_set, min_loss);
            State::graph.lb.insert(new_lb);
        }

        compute_ibs_calls++;

    }

    potential = max_loss - min_loss;
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
