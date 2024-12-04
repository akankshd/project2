import numpy as np
import time

class NearestNeighborClassifier:
    def __init__(self):
        self.train_data = []

    def train(self, train_instances):
        self.train_data = train_instances

    def test(self, test_instance):
        min_distance = float('inf')
        predicted_class = None

        test_features = np.array(test_instance['features'])

        for instance in self.train_data:
            train_features = np.array(instance['features'])
            distance = np.linalg.norm(test_features - train_features)
            if distance < min_distance:
                min_distance = distance
                predicted_class = instance['class']

        return predicted_class

class Validator:
    def __init__(self, classifier, data):
        self.classifier = classifier
        self.data = data

    def evaluate(self, feature_subset):
        correct = 0
        total = len(self.data)

        for i in range(total):
            test_instance = {
                'class': self.data[i]['class'],
                'features': [self.data[i]['features'][f - 1] for f in feature_subset]  
            }
            train_instances = [
                {
                    'class': instance['class'],
                    'features': [instance['features'][f - 1] for f in feature_subset]
                }
                for j, instance in enumerate(self.data) if j != i
            ]

            # Train the classifier
            self.classifier.train(train_instances)

            # Test the classifier
            predicted = self.classifier.test(test_instance)

            # Check if prediction is correct
            if predicted == test_instance['class']:
                correct += 1

        accuracy = correct / total
        return accuracy

def load_data(file_path):
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue  # Skip empty lines
            class_label = int(float(parts[0]))
            features = [float(x) for x in parts[1:]]
            data.append({'class': class_label, 'features': features})

    feature_array = np.array([instance['features'] for instance in data])

    # normalization
    means = feature_array.mean(axis=0)
    stds = feature_array.std(axis=0)
    stds[stds == 0] = 1

    # more normalization
    for instance in data:
        instance['features'] = [(x - m) / s for x, m, s in zip(instance['features'], means, stds)]

    return data

def evaluate_feature_set(selected_features, data):
  
    if not selected_features:
        print("No features selected. Cannot compute accuracy.")
        return 0.0

    start_time = time.time()
    classifier = NearestNeighborClassifier()
    validator = Validator(classifier, data)
    accuracy = validator.evaluate(selected_features) * 100  # Convert to percentage
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluated feature set {sorted(selected_features)} with accuracy: {accuracy:.2f}% (Time: {elapsed_time:.2f}s)")
    return accuracy

def forward_selection(total_features, data):
    current_features = set()
    best_overall_accuracy = 0
    best_feature_set = set()

    # Evaluate accuracy with no features
    initial_accuracy = evaluate_feature_set(current_features, data)
    print('Using no features and NN evaluation, I get an accuracy of {:.1f}%'.format(initial_accuracy))
    print("Beginning search.")

    for i in range(1, total_features + 1):
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0

        for feature in range(1, total_features + 1):
            if feature not in current_features:
                # Try adding this feature to the current set
                features_to_evaluate = current_features.copy()
                features_to_evaluate.add(feature)

                # Evaluate the accuracy of this set
                accuracy = evaluate_feature_set(features_to_evaluate, data)
                print("Using feature(s) {} accuracy is {:.1f}%".format(
                    sorted(features_to_evaluate), accuracy))

                # Update if this subset gives the best accuracy so far
                if accuracy >= best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = feature
                    best_features_so_far = features_to_evaluate.copy()

        # If a feature was identified to add
        if feature_to_add_at_this_level is not None:
            current_features.add(feature_to_add_at_this_level)

            # Update the best overall subset if this accuracy is higher
            if best_accuracy_so_far >= best_overall_accuracy:
                best_overall_accuracy = best_accuracy_so_far
                best_feature_set = current_features.copy()

                # Print the best feature set and accuracy for this iteration
                print("Feature set {} was best, accuracy is {:.1f}%".format(
                    sorted(current_features), best_overall_accuracy))
            else:
                # No improvement in accuracy
                print("(Warning, Accuracy has decreased!)")
                break
        else:
            break

    # Final result
    print("Finished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))


def backward_elimination(total_features, data):
    # Initialize the current feature set with all features
    current_features = set(range(1, total_features + 1))
    best_overall_accuracy = evaluate_feature_set(current_features, data)
    best_feature_set = current_features.copy()

    print("Starting with all features {}".format(sorted(current_features)))
    print("Initial accuracy is {:.1f}%".format(best_overall_accuracy))

    while len(current_features) > 1:
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0  

        # Iterate through each feature to evaluate its impact on accuracy when removed
        for feature in current_features:
            # Create a copy of the current features and remove one feature
            features_to_evaluate = current_features.copy()
            features_to_evaluate.remove(feature)

            # Evaluate the accuracy of the reduced feature set
            accuracy = evaluate_feature_set(features_to_evaluate, data)
            print("Removing feature {}, remaining set {}, accuracy is {:.1f}%".format(
                feature, sorted(features_to_evaluate), accuracy))

            # Update if this subset gives the best accuracy so far
            if accuracy >= best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = feature
                best_features_so_far = features_to_evaluate.copy()

        # If a feature was identified to remove
        if feature_to_remove_at_this_level is not None:
            current_features = best_features_so_far

            # Update the best overall subset if this accuracy is higher
            if best_accuracy_so_far >= best_overall_accuracy:
                best_overall_accuracy = best_accuracy_so_far
                best_feature_set = current_features.copy()

            # Print the best feature set and accuracy for this iteration
            print("Feature set {} was best, accuracy is {:.1f}%".format(
                sorted(current_features), best_overall_accuracy))

            # Stop if only one feature remains
            if len(current_features) == 1:
                break
        else:
            print("(Warning, Removing any feature decreases performance!)")
            break

    # Final result
    print("Finished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))



import time

def main():
    print("Welcome to Akanksh and Saachi's Feature Selection Algorithm.")
    
    # Testing for small and large datasets, code for each
    print("\n--- Testing Small Dataset ---")
    try:
        # Step 1: Load small dataset
        print("Step 1: Loading small dataset...")
        start_time = time.time()
        small_data = load_data("small-test-dataset.txt")
        end_time = time.time()
        print(f"Loaded small dataset with {len(small_data)} instances and {len(small_data[0]['features'])} features.")
        print(f"Time taken to load dataset: {end_time - start_time:.2f} seconds")

        # Step 2: Test the accuracy for features {3, 5, 7}
        print("\nStep 2: Evaluating feature subset {3, 5, 7}...")
        small_selected_features = {3, 5, 7}
        classifier = NearestNeighborClassifier()
        validator = Validator(classifier, small_data)
        start_time = time.time()
        small_accuracy = validator.evaluate(small_selected_features)
        end_time = time.time()
        print(f"Accuracy for features {small_selected_features}: {small_accuracy * 100:.2f}%")
        print(f"Time taken to evaluate feature subset: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error testing small dataset: {e}")

    print("\n--- Testing Large Dataset ---")
    try:
        # Step 1: Load large dataset
        print("Step 1: Loading large dataset...")
        start_time = time.time()
        large_data = load_data("large-test-dataset.txt")
        end_time = time.time()
        print(f"Loaded large dataset with {len(large_data)} instances and {len(large_data[0]['features'])} features.")
        print(f"Time taken to load dataset: {end_time - start_time:.2f} seconds")

        # Step 2: Test the accuracy for features {1, 15, 27}
        print("\nStep 2: Evaluating feature subset {1, 15, 27}...")
        large_selected_features = {1, 15, 27}
        classifier = NearestNeighborClassifier()
        validator = Validator(classifier, large_data)
        start_time = time.time()
        large_accuracy = validator.evaluate(large_selected_features)
        end_time = time.time()
        print(f"Accuracy for features {large_selected_features}: {large_accuracy * 100:.2f}%")
        print(f"Time taken to evaluate feature subset: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error testing large dataset: {e}")

if __name__ == '__main__':
    main()
    