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
                continue
            class_label = int(float(parts[0]))  # Class is the first column
            features = [float(x) for x in parts[1:]]  # Features are the remaining columns
            data.append({'class': class_label, 'features': features})

    feature_array = np.array([instance['features'] for instance in data])
    means = feature_array.mean(axis=0)
    stds = feature_array.std(axis=0)
    stds[stds == 0] = 1  # Avoid division by zero

    for instance in data:
        instance['features'] = [(x - m) / s for x, m, s in zip(instance['features'], means, stds)]

    return data

def evaluate_feature_set(selected_features, data):
  

    start_time = time.time()
    classifier = NearestNeighborClassifier()
    validator = Validator(classifier, data)
    accuracy = validator.evaluate(selected_features) * 100  # Convert to percentage
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluated feature set {sorted(selected_features)} with accuracy: {accuracy:.2f}%")
    return accuracy

def forward_selection(total_features, data):
    current_features = set()
    best_overall_accuracy = 0
    best_feature_set = set()

    print("\nRunning nearest neighbor with no features (default rate), using \"leave-one-out\" evaluation.")
    initial_accuracy = evaluate_feature_set(current_features, data)
    print(f"I get an accuracy of {initial_accuracy:.1f}%\n")
    print("Beginning search.\n")

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
                print(f"Using feature(s) {sorted(features_to_evaluate)}, accuracy is {accuracy:.1f}%")

                # Update if this subset gives the best accuracy so far
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = feature

        # Add the best feature at this level
        if feature_to_add_at_this_level:
            current_features.add(feature_to_add_at_this_level)

            # Check if this is the best overall accuracy so far
            if best_accuracy_so_far > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_so_far
                best_feature_set = current_features.copy()
                print(f"Feature set {sorted(current_features)} was best, accuracy is {best_overall_accuracy:.1f}%\n")
            else:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima.)\n")
        else:
            break

    print("\nFinished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))

def backward_elimination(total_features, data):
    current_features = set(range(1, total_features + 1))
    best_overall_accuracy = evaluate_feature_set(current_features, data)
    best_feature_set = current_features.copy()

    print("\nStarting with all features {}\n".format(sorted(current_features)))
    print(f"Initial accuracy is {best_overall_accuracy:.1f}%\n")

    while len(current_features) > 1:
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0  

        for feature in current_features:
            # Create a copy of the current features and remove one feature
            features_to_evaluate = current_features.copy()
            features_to_evaluate.remove(feature)

            # Evaluate the accuracy of the reduced feature set
            accuracy = evaluate_feature_set(features_to_evaluate, data)
            print(f"Removing feature {feature}, remaining set {sorted(features_to_evaluate)}, accuracy is {accuracy:.1f}%")

            # Update if this subset gives the best accuracy so far
            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = feature

        # Remove the feature with the best accuracy
        if feature_to_remove_at_this_level is not None:
            current_features.remove(feature_to_remove_at_this_level)

            # Check if this is the best overall accuracy so far
            if best_accuracy_so_far > best_overall_accuracy:
                best_overall_accuracy = best_accuracy_so_far
                best_feature_set = current_features.copy()
                print(f"Feature set {sorted(current_features)} was best, accuracy is {best_overall_accuracy:.1f}%\n")
            else:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima.)\n")
        else:
            break

    print("\nFinished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))

import time

def main():
    print("Welcome to Akanksh and Saachi's Feature Selection Algorithm.")

    # Step 1: Input dataset file name (user does this)
    dataset_file = input("Type in the name of the file to test: ").strip()
    
    try:
        # Load the dataset (from the user)
        print(f"\nLoading dataset {dataset_file}...")
        data = load_data(dataset_file)
        print(f"This dataset has {len(data[0]['features'])} features (not including the class attribute) with {len(data)} instances.")
        print("Please wait while I normalize the data... Done!")
    except Exception as e: # error checking if the dataset is invalid -> helps developers
        print(f"Error loading dataset: {e}")
        return



    total_features = len(data[0]['features'])

    # Step 2: Choose the algorithm
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    algorithm_choice = input("Enter your choice: ").strip()

    # Step 3: Run the chosen algorithm
    if algorithm_choice == "1":
        print("\nRunning Forward Selection...\n")
        forward_selection(total_features, data)
    elif algorithm_choice == "2":
        print("\nRunning Backward Elimination...\n")
        backward_elimination(total_features, data)
    else:
        print("Invalid choice. Please restart and select either 1 or 2.")

if __name__ == '__main__':
    main()