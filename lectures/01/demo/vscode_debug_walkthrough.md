# VS Code debugging walkthrough

Use this alongside `vscode_debug_sample.py` to show breakpoints, watches, and stepping.

## Setup
- Open `lectures/01/demo/vscode_debug_sample.py` in VS Code.
- Install the Python extension and select the correct interpreter.
- Start debugging via Run and Debug (green play button) so the input prompts appear in the Debug Console.

## Known bugs to uncover
1) BMI formula uses `weight / height` instead of `weight / (height ** 2)`.
2) `get_bmi_category` returns an undefined `category` variable because of the `catgory` typo.
3) Off-by-one loop in `print_recommendations` triggers `IndexError` and skips the first recommendation.

## Flow to demo
- Set a breakpoint on `bmi = calculate_bmi(...)` and step into the function to spot the formula bug.
- Add a watch on `bmi` and rerun; show how the wrong value propagates to the category.
- Continue until the `NameError` surfaces; fix the typo and rerun.
- Trigger the `IndexError` in the recommendations loop, then adjust the range to `range(len(numbers))` and index safely.
- Rerun start-to-finish to confirm the script prints all four recommendations.
