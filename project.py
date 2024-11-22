import random

def evaluate_feature_set(selected_features):
    return random.uniform(0, 100)

def backward_elimination(total_features):
    current_features = set(range(1, total_features + 1))
    best_overall_accuracy = evaluate_feature_set(current_features)
    best_feature_set = current_features.copy()

    print("Starting with all features {}".format(sorted(current_features)))
    print("Initial accuracy is {:.1f}%".format(best_overall_accuracy))



def main():
    print("Welcome to Your Name's Feature Selection Algorithm.")
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
