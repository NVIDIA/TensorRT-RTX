# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from datetime import datetime
from pathlib import Path


class TestLicenseHeaders:
    """Test suite to verify that all relevant files have the required NVIDIA license header."""

    # The expected license text (without comment markers)
    COPYRIGHT_YEAR_PREFIX = "SPDX-FileCopyrightText: Copyright (c) "
    COPYRIGHT_YEAR_SUFFIX = " NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    EXPECTED_LICENSE_LINES = [
        f"{COPYRIGHT_YEAR_PREFIX}2025{COPYRIGHT_YEAR_SUFFIX}",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        'Licensed under the Apache License, Version 2.0 (the "License");',
        "you may not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "http://www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        'distributed under the License is distributed on an "AS IS" BASIS,',
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
    ]

    @classmethod
    def get_project_root(cls):
        """Get the project root directory."""
        # Go up from demo/tests to the project root
        current_dir = Path(__file__).parent.parent.parent
        return current_dir

    @classmethod
    def find_files_by_pattern(cls, root_path, patterns):
        """Find all files matching the given patterns."""
        files = []
        for pattern in patterns:
            files.extend(root_path.rglob(pattern))
        return files

    @classmethod
    def extract_license_with_octothorpe_comments(cls, file_path, skip_shebang):
        """Extract license header from Python or CMake file (lines starting with #)."""
        license_lines = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    stripped = line.strip()

                    # Skip the shebang line at the beginning, if present (e.g. Python)
                    if skip_shebang and i == 0 and stripped.startswith("#!"):
                        continue

                    if stripped.startswith("#"):
                        license_lines.append(stripped.removeprefix("#").strip())
                    else:
                        # First non-comment line - stop processing
                        break
        except UnicodeDecodeError:
            # Skip binary files
            return []

        return cls.normalize_license_lines(license_lines)

    @classmethod
    def normalize_license_lines(cls, lines):
        """Normalize license lines by removing empty lines at start/end and extra whitespace."""
        # Remove empty lines from the beginning and end
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()

        return lines

    @classmethod
    def extract_license_from_cpp_file(cls, file_path):
        """Extract license header from C++ file (block comment /* ... */)."""
        license_lines = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    content = line.strip()

                    if content == "/*":
                        # Start of block comment
                        continue
                    elif content == "*/":
                        # End of block comment
                        break
                    elif content.startswith("*"):
                        # License content line
                        license_lines.append(content[1:].strip())
                    else:
                        # First non-comment line - stop processing
                        break
        except UnicodeDecodeError:
            # Skip binary files
            return []

        return cls.normalize_license_lines(license_lines)

    @classmethod
    def validate_copyright_year(cls, copyright_line):
        """Validate that copyright line has exact structure with current year"""
        current_year = datetime.now().year

        # Allowed formats:
        # "SPDX-FileCopyrightText: Copyright (c) YYYY NVIDIA CORPORATION & AFFILIATES. All rights reserved."
        # or
        # "SPDX-FileCopyrightText: Copyright (c) YYYY-YYYY NVIDIA CORPORATION & AFFILIATES. All rights reserved."

        if not copyright_line.startswith(cls.COPYRIGHT_YEAR_PREFIX):
            return False
        if not copyright_line.endswith(cls.COPYRIGHT_YEAR_SUFFIX):
            return False

        # Extract the year part between prefix and suffix
        year_part = copyright_line[len(cls.COPYRIGHT_YEAR_PREFIX) : -len(cls.COPYRIGHT_YEAR_SUFFIX)]

        # Validate year format and current year requirement
        if "-" in year_part:
            # Range like "2023-2025"
            if not re.match(r"^\d{4}-\d{4}$", year_part):
                return False
            start_year, end_year = year_part.split("-")
            return int(start_year) < current_year == int(end_year)
        else:
            # Single year like "2025"
            if not re.match(r"^\d{4}$", year_part):
                return False
            return int(year_part) == current_year

    @classmethod
    def is_license_compatible(cls, extracted_lines, expected_lines):
        """Check if the extracted license is compatible with expected format."""
        if len(extracted_lines) != len(expected_lines):
            return False

        for i, (extracted, expected) in enumerate(zip(extracted_lines, expected_lines)):
            if i == 0 and not cls.validate_copyright_year(extracted):
                # First line - validate copyright with exact structure and flexible years
                return False
            elif i != 0 and extracted != expected:
                # All other lines must match exactly
                return False

        return True

    @classmethod
    def check_license_header(cls, file_path, file_type):
        """Check if file has the correct license header."""
        if file_type == "Python":
            extracted_lines = cls.extract_license_with_octothorpe_comments(file_path, skip_shebang=True)
        elif file_type == "C++":
            extracted_lines = cls.extract_license_from_cpp_file(file_path)
        elif file_type == "CMake":
            extracted_lines = cls.extract_license_with_octothorpe_comments(file_path, skip_shebang=False)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        return (
            cls.is_license_compatible(extracted_lines, cls.EXPECTED_LICENSE_LINES),
            extracted_lines,
        )

    @classmethod
    def test_all_files_have_license_header(self):
        """Test that all relevant files have the required license header."""
        root_path = self.get_project_root()

        # Define file types with their patterns and type names
        file_checks = [
            (["*.py"], "Python"),
            (["CMakeLists.txt", "CMakeLists*.txt"], "CMake"),
            (["*.cpp", "*.c", "*.h", "*.hpp"], "C++"),
        ]

        all_missing = []
        all_incorrect = []

        for patterns, file_type in file_checks:
            files = self.find_files_by_pattern(root_path, patterns)

            for file_path in files:
                # Skip empty __init__.py files
                if file_type == "Python" and file_path.name == "__init__.py":
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read().strip()
                            if not content:
                                continue
                    except Exception:
                        pass  # If any issues are encountered, continue with normal processing

                has_correct_license, extracted = self.check_license_header(file_path, file_type)
                if not extracted:
                    all_missing.append((file_path, file_type))
                elif not has_correct_license:
                    all_incorrect.append((file_path, file_type, extracted))

        # Create consolidated error message
        error_msg = []
        if all_missing:
            error_msg.append(f"Files missing license header ({len(all_missing)}):")
            for file_path, file_type in all_missing:
                error_msg.append(f"  - {file_path} ({file_type})")

        if all_incorrect:
            error_msg.append(f"\nFiles with incorrect license header ({len(all_incorrect)}):")
            for file_path, file_type, extracted in all_incorrect:
                error_msg.append(f"  - {file_path} ({file_type})")
                error_msg.append(f"    Found: {extracted[:3]}...")

        assert not all_missing and not all_incorrect, "\n".join(error_msg)
