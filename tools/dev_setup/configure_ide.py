#!/usr/bin/env python3
"""
Market Research System v1.0 - IDE Configuration Script
Created: January 2022
Purpose: Automatically configure popular IDEs for optimal development experience
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List


class IDEConfigurator:
    """Configure popular IDEs for the Market Research System."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.src_path = self.project_root / "src"
        self.data_path = self.project_root / "data"
        self.python_path = sys.executable

    def configure_vscode(self) -> None:
        """Configure Visual Studio Code settings."""
        print("🔧 Configuring Visual Studio Code...")
        
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        # Settings.json
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.pylintEnabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.formatting.blackPath": "./venv/bin/black",
            "python.sortImports.args": ["--profile", "black"],
            "python.testing.pytestEnabled": True,
            "python.testing.unittestEnabled": False,
            "python.testing.pytestArgs": ["tests"],
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/.pytest_cache": True,
                "**/venv": True,
                "data/raw/**": True,
                "data/cache/**": True,
                "logs/**": True,
                "reports/daily/**": True,
                "reports/weekly/**": True
            },
            "jupyter.askForKernelRestart": False,
            "jupyter.interactiveWindowMode": "perFile",
            "files.associations": {
                "*.yml": "yaml",
                "*.yaml": "yaml"
            },
            "yaml.validate": True,
            "python.analysis.extraPaths": [
                "./src"
            ],
            "python.analysis.autoImportCompletions": True,
            "terminal.integrated.env.linux": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "terminal.integrated.env.osx": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "terminal.integrated.env.windows": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }

        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)

        # Launch.json for debugging
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Daily Data Collection",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/data_collection/collect_daily_data.py",
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Generate Daily Report",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/reporting/generate_daily_report.py",
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Technical Analysis",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/analysis/calculate_indicators.py",
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                },
                {
                    "name": "Current File",
                    "type": "python",
                    "request": "launch",
                    "program": "${file}",
                    "console": "integratedTerminal",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}/src"
                    }
                }
            ]
        }

        with open(vscode_dir / "launch.json", "w") as f:
            json.dump(launch_config, f, indent=2)

        # Extensions.json
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.flake8",
                "ms-python.black-formatter",
                "ms-python.isort",
                "ms-python.mypy-type-checker",
                "ms-toolsai.jupyter",
                "ms-vscode.test-adapter-converter",
                "redhat.vscode-yaml",
                "ms-vscode.vscode-json",
                "streetsidesoftware.code-spell-checker",
                "ms-vscode.makefile-tools",
                "ms-vscode-remote.remote-containers",
                "github.vscode-pull-request-github",
                "gitlens.gitlens",
                "visualstudioexptteam.vscodeintellicode"
            ]
        }

        with open(vscode_dir / "extensions.json", "w") as f:
            json.dump(extensions, f, indent=2)

        # Tasks.json for common tasks
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Install Dependencies",
                    "type": "shell",
                    "command": "./tools/dev_setup/install_dependencies.sh",
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Run Tests",
                    "type": "shell",
                    "command": "python",
                    "args": ["-m", "pytest", "tests/", "-v"],
                    "group": "test",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Collect Daily Data",
                    "type": "shell",
                    "command": "python",
                    "args": ["scripts/data_collection/collect_daily_data.py"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Generate Daily Report",
                    "type": "shell",
                    "command": "python",
                    "args": ["scripts/reporting/generate_daily_report.py"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Format Code",
                    "type": "shell",
                    "command": "black",
                    "args": ["src/", "scripts/", "tests/"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Lint Code",
                    "type": "shell",
                    "command": "flake8",
                    "args": ["src/", "scripts/", "tests/"],
                    "group": "test",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                }
            ]
        }

        with open(vscode_dir / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)

        print("✅ VS Code configuration completed")

    def configure_pycharm(self) -> None:
        """Configure PyCharm IDE settings."""
        print("🔧 Configuring PyCharm...")
        
        idea_dir = self.project_root / ".idea"
        idea_dir.mkdir(exist_ok=True)

        # Create project structure XML
        modules_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://$PROJECT_DIR$/.idea/market-research-system.iml" filepath="$PROJECT_DIR$/.idea/market-research-system.iml" />
    </modules>
  </component>
</project>'''

        with open(idea_dir / "modules.xml", "w") as f:
            f.write(modules_xml)

        # Create module definition
        module_iml = '''<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder url="file://$MODULE_DIR$/src" isTestSource="false" />
      <sourceFolder url="file://$MODULE_DIR$/tests" isTestSource="true" />
      <excludeFolder url="file://$MODULE_DIR$/venv" />
      <excludeFolder url="file://$MODULE_DIR$/data/raw" />
      <excludeFolder url="file://$MODULE_DIR$/data/cache" />
      <excludeFolder url="file://$MODULE_DIR$/logs" />
      <excludeFolder url="file://$MODULE_DIR$/reports/daily" />
      <excludeFolder url="file://$MODULE_DIR$/reports/weekly" />
    </content>
    <orderEntry type="inheritedJdk" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
  <component name="PyDocumentationSettings">
    <option name="format" value="GOOGLE" />
  </component>
  <component name="TestRunnerService">
    <option name="PROJECT_TEST_RUNNER" value="pytest" />
  </component>
</module>'''

        with open(idea_dir / "market-research-system.iml", "w") as f:
            f.write(module_iml)

        # PyCharm inspection profiles
        inspections_dir = idea_dir / "inspectionProfiles"
        inspections_dir.mkdir(exist_ok=True)

        profiles_settings = '''<component name="InspectionProjectProfileManager">
  <settings>
    <option name="USE_PROJECT_PROFILE" value="true" />
    <version value="1.0" />
  </settings>
</component>'''

        with open(inspections_dir / "profiles_settings.xml", "w") as f:
            f.write(profiles_settings)

        print("✅ PyCharm configuration completed")

    def configure_jupyter(self) -> None:
        """Configure Jupyter Lab/Notebook settings."""
        print("🔧 Configuring Jupyter...")
        
        jupyter_dir = self.project_root / ".jupyter"
        jupyter_dir.mkdir(exist_ok=True)

        # Jupyter config
        jupyter_config = f'''# Market Research System - Jupyter Configuration
c.NotebookApp.notebook_dir = '{self.project_root}'
c.NotebookApp.open_browser = True
c.NotebookApp.port = 8888
c.NotebookApp.

not complted