import subprocess

# Define the full paths of scripts
SCRIPT_PATHS = {
    1: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/t1.py",
    2: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/t22.py",
    3: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/t3.py",
    4: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO/b2.py",
    5: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/YOLO_2/b1.py",
    6: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/Yolo_3/1a.py",
    7: "/Users/vrajalpeshkumarmodi/Downloads/Cricket/Yolo_3/5.py",
}

def run_script(script_path):
   
    print(f"Running {script_path}...")
    subprocess.run(["python3", script_path], check=True)

def display_menu():
    
    print("Select a task to run:")
    print("1. Object Detection")
    print("2. Ball Tracking & Speed")
    print("3. Counterfactual Generation on Ball")
    print("4. Bat swing and angle")
    print("5. Shot execution (Drive, defensive, aggressive, leaving)")
    print("6. Player Positioning")
    print("7. Shot Style Vision")
    print("8. Exit")

def main():

    while True:
        display_menu()
        try:
            choice = int(input("Enter the number of your choice: "))
            if choice == 1:
                run_script(SCRIPT_PATHS[1])
            elif choice == 2:
                run_script(SCRIPT_PATHS[2])
            elif choice == 3:
                run_script(SCRIPT_PATHS[3])
            elif choice == 4:
                run_script(SCRIPT_PATHS[4])
            elif choice == 5:
                run_script(SCRIPT_PATHS[5])
            elif choice == 6:
                run_script(SCRIPT_PATHS[6])
            elif choice == 7:
                run_script(SCRIPT_PATHS[7])    
            elif choice == 8:
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please choose again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyError:
            print("Invalid choice. Please choose a valid task.")

if __name__ == "__main__":
    main()
