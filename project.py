import random

def evaluate_feature_set(selected_features):
    # function that will generate random
    return random.uniform(0, 100)

def forward_selection(total_features):
    # Initialize the current feature set as empty
    current_features = set()
    best_overall_accuracy = 0
    best_feature_set = set()

    # Evaluate accuracy with no features
    initial_accuracy = evaluate_feature_set(current_features)
    # text required in project guideline
    print('Using no features and "random" evaluation, I get an accuracy of {:.1f}%'.format(initial_accuracy))
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
                accuracy = evaluate_feature_set(features_to_evaluate)
                print("Using feature(s) {} accuracy is {:.1f}%".format(
                    sorted(features_to_evaluate), accuracy)) # gets the best accruacy

                # update if this subset gives the best accuracy so far

                
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
                    sorted(current_features), best_accuracy_so_far))
            else:
                # No improvement in accuracy
                print("(Warning, Accuracy has decreased!)") # project guideline
                break 
        else:
            break

    # print statements from the guideline
    print("Finished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))

def backward_elimination(total_features):
    # Initialize the current feature set with all features
    current_features = set(range(1, total_features + 1))
    best_overall_accuracy = evaluate_feature_set(current_features)
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
            accuracy = evaluate_feature_set(features_to_evaluate)
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
                sorted(current_features), best_accuracy_so_far))

            # Stop if only one feature remains
            if len(current_features) == 1:
                break
        else:
            print("(Warning, Removing any feature decreases performance!)")
            break

    # Print the final result
    print("Finished search!! The best feature subset is {}, which has an accuracy of {:.1f}%".format(
        sorted(best_feature_set), best_overall_accuracy))


def main():
    print("Welcome to Akanksh and Saachi's Feature Selection Algorithm.")
    total_features = int(input("Please enter total number of features: "))
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = input()

    if choice == '1':
        forward_selection(total_features)
    elif choice == '2':
        backward_elimination(total_features)
    else:
        print("Invalid choice or special algorithm not implemented.")


if __name__ == '__main__':
    main()
