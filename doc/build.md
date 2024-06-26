## How to build the project

This page documents how you can build the project and the Python extension module manually.

### Step 1: Install required development tools

OSST uses `CMake` as its default cross-platform build system and `Ninja` as the default generator for parallel builds.  
OSST relies on `scikit-build` to compile the Python extension and build the redistributable wheel file.  
OSST exploits `delocate`, `auditwheel` and `delvewheel` to copy all required 3rd-party dynamic libraries into the wheel file on macOS, Ubuntu and Windows respectively.

**macOS:**

```bash
brew install cmake
brew install ninja
brew install pkg-config
pip3 install --upgrade scikit-build
pip3 install --upgrade delocate
```

**Ubuntu:**

```bash
sudo apt install -y cmake
sudo apt install -y ninja-build
sudo apt install -y pkg-config
pip3 install --upgrade scikit-build
pip3 install --upgrade auditwheel
sudo apt install -y patchelf # Required by auditwheel
```

**Windows:**

**Please make sure that you launch the Powershell as Admin.**

**Step 1.1:** Install Chocolatey

In addition to Windows Package Manager (a.k.a. `winget`), [Chocolatey](https://chocolatey.org/) is used to install tools that are not yet provided by `winget`.  
Please follow this [guide](https://chocolatey.org/install#individual) or use the following commands to install Chocolatey.

```ps1
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**Step 1.2:** Install vcpkg

OSST requires the C++ package manager `vcpkg` to install all necessary C and C++ libraries on Windows.  
Please follow this [guide](https://vcpkg.io/en/getting-started.html) or use the following commands to install `vcpkg` to `C:\vcpkg`.  

```ps1
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

Once you have installed `vcpkg`, for example, to `C:\vcpkg`, you need to...
- Update your `PATH` variable to include `C:\vcpkg`.
- Add a new environment variable `VCPKG_INSTALLATION_ROOT` with a value of `C:\vcpkg`.

The following Powershell script modifies the system environment permanently.
In other words, all users can see these two new variables.

```ps1
$vcpkg = "C:\vcpkg"
$old = (Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH).path
$new = "$old;$vcpkg"
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $new
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name VCPKG_INSTALLATION_ROOT -Value $vcpkg
```

You can verify whether the new variable `VCPKG_INSTALLATION_ROOT` is set properly by typing the following command in Powershell:  
Note that you may need to restart your terminal or reboot your computer to apply the changes.

```ps1
$ENV:VCPKG_INSTALLATION_ROOT
```

**Step 1.3:** Install required development tools

```ps1
winget install Kitware.CMake
choco install -y ninja
choco install -y pkgconfiglite
pip3 install --upgrade scikit-build
pip3 install --upgrade delvewheel
```

If `choco install -y pkgconfiglite` reports SSL/TLS-related errors, please use the following commands to install it manually.

```ps1
Invoke-WebRequest -UserAgent "Wget" -Uri "http://downloads.sourceforge.net/project/pkgconfiglite/0.28-1/pkg-config-lite-0.28-1_bin-win32.zip" -OutFile "C:\pkgconfig.zip"
Expand-Archive "C:\pkgconfig.zip" -DestinationPath "C:\"
Copy-Item "C:\pkg-config-lite-0.28-1\bin\pkg-config.exe" "C:\Windows"
Remove-Item "C:\pkg-config-lite-0.28-1" -Force -Recurse
Remove-Item "C:\pkgconfig.zip"
```

### Step 2: Install required 3rd-party libraries

OSST relies on `IntelTBB` and `GMP`.  
The following header-only libraries are already included in this repository.
- [**nlohmann/json**](https://github.com/nlohmann/json) - JSON Parser
- [**ben-strasser/fast-cpp-csv-parser**](https://github.com/ben-strasser/fast-cpp-csv-parser) - CSV Parser

**macOS:**

```bash
brew install tbb
brew install gmp
```

**Ubuntu:**

```bash
sudo apt install -y libtbb-dev
sudo apt install -y libgmp-dev
```

**Windows:**

```ps1
vcpkg install tbb:x64-windows
vcpkg install gmp:x64-windows
```

### Step 3: Build the project

You can build the OSST project by...
- Using the automatic build script `build.py` (OSST CLI + Python Wheel)
- Using `scikit-build` manually (OSST CLI + Python Wheel)
- Using `cmake` manually (OSST CLI Only)

#### Method 1:

This repository ships a `build.py` script that builds the library and the Python wheel automatically.

```bash
python3 build.py
```

#### Method 2:

**Step 3.1:** Build the C++ library and the Python wheel

Please adjust the API version to your Python version accordingly.  
For example, if you are using Python 3.9, set the version to be `--py-limited-api=cp39`.  
Please adjust the number of threads `-j8` accordingly.

```bash
# Debug Build
python3 setup.py bdist_wheel --py-limited-api=cp37 --build-type=Debug -G Ninja -- -- -j8

# Release Build
python3 setup.py bdist_wheel --py-limited-api=cp37 --build-type=Release -G Ninja -- -- -j8
```

You can find the command line tools in `_skbuild/<platform-specific>/cmake-build/` and the wheel file in `dist/`.

**Step 3.2:** Add all required 3rd-party libraries to the wheel file

Please adjust the name of your wheel file accordingly.

**macOS:**

```bash
delocate-wheel -w dist -v dist/osst-0.1.0-cp310-cp310-macosx_12_0_x86_64.whl
```

**Ubuntu:**

```bash
auditwheel repair -w dist --plat linux_x86_64 dist/osst-1.0.5-cp310-cp310-linux_x86_64.whl
```

**Windows:**

```ps1
python3 -m delvewheel repair --no-mangle-all --add-path "$ENV:VCPKG_INSTALLATION_ROOT\installed\x64-windows\bin" dist/osst-0.1.0-cp310-cp310-win_amd64.whl -w dist
```

You will find the fixed wheel file in `dist`.

#### Method 3:

If you build the project on Ubuntu or macOS, you need to remove the `-DCMAKE_TOOLCHAIN_FILE=...` option.  
If you build the project on Windows, you must run the following commands in a **Developer Powershell for Visual Studio 2019/2022**.  
Please adjust the number of threads `--parallel 8` accordingly.

**Debug Build:**

```bash
# Create the build directory
mkdir build

# Generate all necessary files for the Ninja build system
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ENV:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug 

# Compile the project from scratch with 8 threads
cmake --build build --config Debug --clean-first --parallel 8

# Launch the OSST executable and its tests
.\build\Debug\osst.exe
.\build\Debug\osst_tests.exe
```

**Release Build:**

```bash
# Create the build directory
mkdir build

# Generate all necessary files for the Ninja build system
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=$ENV:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release

# Compile the project from scratch with 8 threads
cmake --build build --config Release --clean-first --parallel 8

# Launch the OSST executable and its tests
.\build\Release\osst.exe
.\build\Release\osst_tests.exe
```

### Step 4: Build the manylinux wheel (Optional)

```bash
# Change the working directory to this repository
cd path/to/osst

# Run the CentOS docker provided by manylinux
# The repository is mapped at /source
docker run -i -t --mount type=bind,source=`pwd`,target=/source quay.io/pypa/manylinux2014_x86_64

# CentOS Shell
# Install required development tools
yum install -y zip
yum install -y cmake
yum install -y ninja-build
yum install pkgconfig
python3.9 -m pip install --upgrade scikit-build
python3.9 -m pip install --upgrade auditwheel
yum install -y patchelf

# Install the VCPKG package manager
git clone https://github.com/Microsoft/vcpkg.git 
./vcpkg/bootstrap-vcpkg.sh
export PATH=/vcpkg:$PATH
export VCPKG_INSTALLATION_ROOT=/vcpkg

# Install required 3rd-party libraries
vcpkg install tbb
vcpkg install gmp

# Change the working directory to the repository
cd /source

# Build the project and the wheel file
python3.9 build.py

# Quit the CentOS shell
exit
```

You can find the manylinux wheel in `dist`.

### Step 5: Run the experiment with the example dataset 

Install the wheel file in `dist` and all required Python packages.  
You may then execute the script `osst/example.py` to run the experiment with the example dataset.  
Please adjust the name of your wheel file accordingly.

```bash
pip3 install dist/osst-0.1.0-cp310-cp310-macosx_12_0_x86_64.whl
pip3 install attrs packaging editables pandas sklearn sortedcontainers gmpy2 matplotlib
python3 osst/example.py
```

If you are using Python 3.12, please install `gmpy2==2.2.0a1`.

```bash
pip3 install dist/osst-0.1.0-cp310-cp310-macosx_12_0_x86_64.whl
pip3 install attrs packaging editables pandas sklearn sortedcontainers gmpy2==2.2.0a1 matplotlib
python3 osst/example.py
```