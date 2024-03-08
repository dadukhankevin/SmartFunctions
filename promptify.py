import os

def process_python_files(directory, output_file):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".md")):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                    output_file.write(f"filename: {file}\n\n")
                    output_file.write(content)
                    output_file.write("\n\n...\n\n")

# Directory to start the search from
start_directory = "."

# Output file name
output_filename = "python_files_content.txt"

with open(output_filename, "w") as output_file:
    process_python_files(start_directory, output_file)

print(f"Python files content has been written to {output_filename}")