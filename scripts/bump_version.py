#!/usr/bin/env python3
"""
Version bumping utility for QuantSim.

Usage:
    python scripts/bump_version.py patch    # 0.1.0 -> 0.1.1
    python scripts/bump_version.py minor    # 0.1.0 -> 0.2.0
    python scripts/bump_version.py major    # 0.1.0 -> 1.0.0
    python scripts/bump_version.py 0.2.5    # Set specific version
"""

import sys
import re
from pathlib import Path

def get_current_version():
    """Get current version from __init__.py"""
    init_file = Path("quantsim/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find version in quantsim/__init__.py")
    return match.group(1)

def parse_version(version_str):
    """Parse version string into components"""
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError("Version must be in format MAJOR.MINOR.PATCH")
    return [int(p) for p in parts]

def bump_version(current_version, bump_type):
    """Bump version based on type"""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        # Assume it's a specific version
        parse_version(bump_type)  # Validate format
        return bump_type

def update_version_in_file(file_path, old_version, new_version):
    """Update version in a specific file"""
    content = Path(file_path).read_text()
    
    if file_path.endswith('__init__.py'):
        # Update __version__ = "x.y.z"
        pattern = r'(__version__ = ["\'])[^"\']+(["\'])'
        replacement = f'\\g<1>{new_version}\\g<2>'
    elif file_path.endswith('pyproject.toml'):
        # Update version = "x.y.z"
        pattern = r'(version = ["\'])[^"\']+(["\'])'
        replacement = f'\\g<1>{new_version}\\g<2>'
    else:
        raise ValueError(f"Don't know how to update version in {file_path}")
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print(f"⚠️  No version found to update in {file_path}")
        return False
    
    Path(file_path).write_text(new_content)
    print(f"✅ Updated {file_path}: {old_version} -> {new_version}")
    return True

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    bump_type = sys.argv[1]
    
    try:
        # Get current version
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        # Calculate new version
        new_version = bump_version(current_version, bump_type)
        print(f"New version: {new_version}")
        
        # Confirm with user
        response = input(f"\nUpdate version from {current_version} to {new_version}? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled")
            sys.exit(0)
        
        # Update files
        files_updated = 0
        
        # Update __init__.py
        if update_version_in_file("quantsim/__init__.py", current_version, new_version):
            files_updated += 1
        
        # Update pyproject.toml
        if update_version_in_file("pyproject.toml", current_version, new_version):
            files_updated += 1
        
        if files_updated > 0:
            print(f"\n🎉 Successfully updated {files_updated} files!")
            print("\nNext steps:")
            print("1. Review changes: git diff")
            print("2. Commit changes: git add . && git commit -m 'Bump version to {}'".format(new_version))
            print("3. Create tag: git tag v{}".format(new_version))
            print("4. Push: git push origin main --tags")
            print("5. Create GitHub Release to trigger PyPI publish")
        else:
            print("❌ No files were updated")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 