# Installing a PyMC environment

## For VS Code 
1. Install [pixi](https://pixi.sh/latest/advanced/installation/):
    - Run this in PowerShell:
        - `powershell -ExecutionPolicy ByPass -c "irm -useb <https://pixi.sh/install.ps1> | iex"`
2. Create the environment in a project directory
   - Download this repository as a zip file and extract it into an empty director
     - This will be your project direcotry
     - Open a command prompt in the new direcotry
     - In the command prompt type:
      - In the cloned repository run `pixi install`
        - This will install the environment you need for PyMC
3. Set up VS Code
   - Make sure `pixi` extension to VS Code is installed
   - Type `code .` to start VS Code
      - Or click Repository -> Open repository in VS Code in GitHub Desktop
      - VS Code should automatically recognize the environment and suggest it as the default Python environment
4. Run the demo files
   - There are two demo files:
      - Straight python: `pymc_demo.py` 
      - Jupyter notebook: `pymc_demo.ipynb`
  
## For PyCharm
1. Install necessary software
   - Install [pixi](https://pixi.sh/latest/advanced/installation/):
      - Run this in PowerShell:
        - `powershell -ExecutionPolicy ByPass -c "irm -useb <https://pixi.sh/install.ps1> | iex"`
   - Install [mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows)
     - Download the installer and run it
2. Create the environment in the project directory
   - Download this repository as a zip file and extract it into an empty directory
     - This will be your project direcotry
   - Open a command prompt in the new direcotry
   - With the current directory being the project direcotry, type into the commmand prompt:
      - `pixi project export conda-environment > environment.yml `
        - This will create an environment specification file for mamba/conda 
          - Mamba / conda are compatible with PyCharm
      - `mamba env create -f environment.yml -n project_name`
        - Replace `project_name` with the name for the environment
        - PyCharm expects this to be the name of the directory
      - `mamba activate project_name`
      - `mamba install m2w64-toolchain`
        - This installs the compiler that pymc uses
3. Setup PyCharm
   - Open PyCharm and use it to open the project directory 
   - Go to Settings (gear bar upper right or Ctrl-Alt-S) -> Project -> Python Interpreter
   - Click `add interpreter` -> `add local interpreter`
     - Under `Type` choose Conda
     - Under `Name` choose the `project_name`
       - You may need to type it in
   - Re-read the package list
     - `File` -> `Invalidate caches` and then `Ivalidate and restart`
4. Run the demo files
   - There are two demo files:
      - Straight python: `pymc_demo.py` 
      - Jupyter notebook: `pymc_demo.ipynb`

## For Spyder
There is probably a way to get Spyder to work with a regular install, but I got it to work by installing it into the environment

1. Install [pixi](https://pixi.sh/latest/advanced/installation/):
    - Run this in PowerShell:
        - `powershell -ExecutionPolicy ByPass -c "irm -useb <https://pixi.sh/install.ps1> | iex"`
2. Create the environment in a project directory
   - Download this repository as a zip file and extract it into an empty director
     - This will be your project direcotry
     - Open a command prompt in the new direcotry
     - In the command prompt type:
      - In the cloned repository run `pixi install`
        - This will install the environment you need for PyMC
    - Install `spyder` into the `pixi` environment
      - `pixi add spyder numpy scipy pandas matplotlib sympy cython`
3. Run Spyder from within the pixi environment
   - Still in the command prompt and still in the project directory, type:
     - `pixi shell`
     - `spyder`
   - This will open a spyder fully setup with the appropriate environment
4. Run the demo files
   - Spyder doesn't support Jupyter notebooks, so you only have the straight Python:
      - `pymc_demo.py` 
  