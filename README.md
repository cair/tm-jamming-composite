# TM-Composites - Application: Jamming Detection

This project focuses on utilizing TM-Composites to analyze radio frequency (RF) signals for the purpose of detecting jamming activities. By leveraging advanced signal processing and analysis techniques, this application aims to identify disruptions in RF communications, enhancing security and reliability in various operational environments.

## Prerequisites

The application is designed to be flexible and should be compatible with a wide range of systems. However, it has been specifically optimized for use on DGX-2 Machines, utilizing the following setup:

- **Visual Studio Code (VSCode)**: An open-source code editor developed by Microsoft for Windows, Linux, and macOS.
- **Remote - SSH Extension for VSCode**: This extension allows VSCode to connect to and work seamlessly with remote servers over SSH, providing an integrated development environment (IDE) experience.
- **Development Containers (Devcontainers)**: Utilizing Devcontainers within VSCode enables you to create a consistent and fully configured development environment on any machine. This is particularly useful for projects that require specific tools, runtimes, and dependencies.

### System Requirements

- **NVIDIA DGX-2 Machine**: For optimal performance and compatibility, it is recommended to use an NVIDIA DGX-2 Machine. Ensure that the system is properly set up and configured for remote development.
- **Docker**: Ensure Docker is installed on the DGX-2 Machine to use Devcontainers.
- **Visual Studio Code**: Installed on your local machine with the Remote - SSH and Remote - Containers extensions added.

## Usage

Follow these steps to set up and start using the project within a Devcontainer on a DGX-2 machine:

1. **Install Visual Studio Code**: If not already installed, download and install VSCode on your local machine from the [official website](https://code.visualstudio.com/).

2. **Install Required Extensions**:
   - Open VSCode, go to the Extensions view by clicking on the square icon on the sidebar, or pressing `Ctrl+Shift+X`.
   - Search for and install the "Remote - SSH" and "Remote - Containers" extensions.

3. **Connect to the DGX-2 Machine**:
   - Open the Command Palette by pressing `F1` or `Ctrl+Shift+P`, then type "Remote-SSH: Connect to Host..." and hit Enter.
   - Enter the SSH details for your DGX-2 Machine as prompted.

4. **Set Up the Devcontainer**:
   - Once connected, open the project folder in VSCode.
   - A notification may appear asking if you want to reopen the project in a container. Select "Reopen in Container". If the notification does not appear, you can open the Command Palette and select "Remote-Containers: Reopen in Container".

5. **Run the Project**:
   - With the project open in the Devcontainer, you should now have all the necessary dependencies and environment configured. Follow your project's specific run instructions to start analyzing RF signals for jamming detection.

By following these instructions, you can ensure a consistent and efficient development environment for working with TM-Composites and jamming detection on RF signals, optimized for DGX-2 Machines.
