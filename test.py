def find_jumps_in_second_column(file_path):
    """
    Loads a text file, extracts the second column values, sorts them,
    and finds line numbers where jumps greater than 1 occur.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of line numbers where jumps greater than 1 occur, or None if an error occurred.
    """
    try:
        # Load the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract and convert to float, storing both line number and value
        second_column_data = [(i + 1, float(line.split(",")[1])) for i, line in enumerate(lines)]

        # Sort by the second column value
        sorted_data = sorted(second_column_data, key=lambda x: x[1])
        print(sorted_data[300:400])
        # Find jumps greater than 1
        jump_lines = []
        for i in range(len(sorted_data) - 1):
            if sorted_data[i + 1][1] - sorted_data[i][1] > 1:
                jump_lines.append(sorted_data[i + 1][0])  # Store the line number after the jump

        return jump_lines

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_and_sort_second_column(input_file_path, output_file_path="sorted.txt"):
    """
    Extracts the second column values from a text file, sorts them, and saves them to a new file.

    Args:
        input_file_path (str): The path to the input text file.
        output_file_path (str, optional): The path to the output file. Defaults to "sorted.txt".
    """
    try:
        # Load the file
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        # Extract and convert to float
        second_column_values = [float(line.split(",")[1]) for line in lines]

        # Sort
        sorted_values = sorted(second_column_values)

        # Write to output file
        with open(output_file_path, 'w') as output_file:
            for value in sorted_values:
                output_file.write(str(value) + "\n")

        print(f"Sorted values written to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_sort_and_mark_jumps(input_file_path, output_file_path="sorted.txt"):
    """
    Extracts the second column values, sorts them, writes to a file, and marks jumps > 1 with "Error".

    Args:
        input_file_path (str): Path to the input text file.
        output_file_path (str, optional): Path to the output file. Defaults to "sorted.txt".
    """
    try:
        # Load the file
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        # Extract and convert to float, storing both line number and value
        second_column_data = [(i + 1, float(line.split(",")[1])) for i, line in enumerate(lines)]

        # Sort by the second column value
        sorted_data = sorted(second_column_data, key=lambda x: x[1])

        print("Sorted last: ",sorted_data[-1])
        # Write to output file, marking jumps with "Error"
        with open(output_file_path, 'w') as output_file:
            for i, (line_num, value) in enumerate(sorted_data):
                if i > 0 and value - sorted_data[i - 1][1] > 1:  # Check for jump
                    output_file.write(f"{value} Error \n")  # Add "Error" with whitespace
                    print(f"Abrupt jump (greater than 1) found after line {line_num - 1}")
                else:
                    output_file.write(str(value) + "\n")  # Write value without "Error"

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Replace 'your_detections.txt' with the actual path to your file
input_file_path = "detections.txt"
extract_sort_and_mark_jumps(input_file_path)