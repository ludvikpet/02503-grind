from collections import deque
import sys
import os

def split_solution_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = deque(f.readlines())
    
    in_dir = os.path.dirname(input_file)
    basename = os.path.basename(input_file).split('.')[0] # Get file basename

    output_file_sol =f"{basename}_sol.py"   # Solution file
    output_file_ex = f"{basename}_ex.py"    # Exercise file

    with open(output_file_sol, 'w', encoding='utf-8') as f_sol, \
         open(output_file_ex, 'w', encoding='utf-8') as f_nosol:

        while lines:

            # Retrieve line and its stripped version
            line = lines.popleft()
            s_line = line.strip()

            # If single line solution, only write line to f_sol
            if s_line.startswith("#SOL"):
                f_sol.write(line.replace("#SOL ",""))

            # If block solution: skip current line and append all lines to f_sol until """ is found
            elif s_line.startswith("\"\"\"SOL"):
                line = lines.popleft() 
                while not line.strip().startswith("\"\"\""):
                    f_sol.write(line)
                    line = lines.popleft()
                
            # If line doesn't contain solution, write to both files
            else:
                f_sol.write(line)
                f_nosol.write(line)


if __name__ == "__main__":
    input_file = sys.argv[1]
    split_solution_file(
        input_file=input_file,
    )
