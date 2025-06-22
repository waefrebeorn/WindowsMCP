# Tests for the WuBu resource loader utility.
# Verifies that paths to resources (config files, assets) are correctly resolved.

import unittest
import os
import sys
import yaml # For creating dummy config file
import shutil # For robustly cleaning up directories

# Adjust path to import from wubu.utils
# This structure assumes tests are run from project root (e.g., `python -m unittest discover tests`)
# Project root
#   - src
#     - wubu
#       - utils
#         - resource_loader.py
#   - tests
#     - utils
#       - test_resource_loader.py
try:
    from wubu.utils.resource_loader import get_base_path, load_config, get_resource_path
except ImportError:
    # Fallback for simpler execution context (e.g. running file directly from tests/utils)
    # This is fragile; prefer running tests with proper Python path setup.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(tests_dir)
    src_dir_path = os.path.join(project_root_dir, "src")
    if src_dir_path not in sys.path:
        sys.path.insert(0, src_dir_path)
    if project_root_dir not in sys.path: # Also add project root for configs etc.
        sys.path.insert(0, project_root_dir)
    from wubu.utils.resource_loader import get_base_path, load_config, get_resource_path


class TestWuBuResourceLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths once for all tests in this class."""
        cls.project_root = get_base_path() # Should be project root
        cls.wubu_package_dir_in_src = os.path.join(cls.project_root, "src", "wubu") # Path to src/wubu/
        cls.assets_dir_at_root = os.path.join(cls.project_root, "assets") # Path to project_root/assets/

        # Ensure dummy directories for tests exist
        os.makedirs(os.path.join(cls.wubu_package_dir_in_src, "sounds", "effects"), exist_ok=True)
        os.makedirs(os.path.join(cls.wubu_package_dir_in_src, "images", "ui"), exist_ok=True)
        os.makedirs(os.path.join(cls.assets_dir_at_root, "general_assets", "subfolder"), exist_ok=True)


    def setUp(self):
        """Set up for each individual resource loader test."""
        self.test_config_filename = "_test_wubu_config.yaml"
        self.test_config_path_at_root = os.path.join(self.project_root, self.test_config_filename)
        self.test_config_path_in_src = os.path.join(self.project_root, "src", self.test_config_filename)


        self.dummy_config_data = {
            'wubu_name': 'TestWuBu System',
            'logging': {'level': 'TEST'},
            'tts': {'default_voice': 'TestWuBuVoice'}
        }
        # Create dummy config at root for primary test
        with open(self.test_config_path_at_root, 'w') as f:
            yaml.dump(self.dummy_config_data, f)

        # Dummy resource files
        self.sound_file_rel_path_in_wubu = os.path.join("sounds", "effects", "test_wubu_sound.wav")
        self.abs_sound_file_path = os.path.join(self.wubu_package_dir_in_src, self.sound_file_rel_path_in_wubu)
        with open(self.abs_sound_file_path, "w") as f: f.write("dummy wubu sound data")

        self.image_file_rel_path_in_wubu = os.path.join("images", "ui", "test_wubu_image.png")
        self.abs_image_file_path = os.path.join(self.wubu_package_dir_in_src, self.image_file_rel_path_in_wubu)
        with open(self.abs_image_file_path, "w") as f: f.write("dummy wubu image data")

        self.asset_file_rel_path_in_assets = os.path.join("general_assets", "subfolder", "test_wubu_asset.txt")
        self.abs_asset_file_path = os.path.join(self.assets_dir_at_root, self.asset_file_rel_path_in_assets)
        with open(self.abs_asset_file_path, "w") as f: f.write("dummy wubu asset data")


    def tearDown(self):
        """Clean up dummy files created for each test."""
        if os.path.exists(self.test_config_path_at_root): os.remove(self.test_config_path_at_root)
        if os.path.exists(self.test_config_path_in_src): os.remove(self.test_config_path_in_src)
        if os.path.exists(self.abs_sound_file_path): os.remove(self.abs_sound_file_path)
        if os.path.exists(self.abs_image_file_path): os.remove(self.abs_image_file_path)
        if os.path.exists(self.abs_asset_file_path): os.remove(self.abs_asset_file_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up directories created for the class, if empty."""
        # Use shutil.rmtree to remove directories and their contents if needed, be careful.
        # For this example, we'll only try to remove if they are empty after file deletions.
        # A more robust way is to remove specific dummy subdirs created in setUpClass.
        if os.path.exists(os.path.join(cls.wubu_package_dir_in_src, "sounds", "effects")):
            try: os.rmdir(os.path.join(cls.wubu_package_dir_in_src, "sounds", "effects"))
            except OSError: pass # Not empty
        if os.path.exists(os.path.join(cls.wubu_package_dir_in_src, "sounds")):
            try: os.rmdir(os.path.join(cls.wubu_package_dir_in_src, "sounds"))
            except OSError: pass
        # ... and so on for other created directories.
        # For simplicity in this example, we might leave empty dirs, or use shutil.rmtree on known dummy roots.
        # If assets_dir_at_root was created SOLELY for this test, we could rmtree it.
        # if os.path.exists(cls.assets_dir_at_root): shutil.rmtree(cls.assets_dir_at_root)


    def test_get_base_path_resolution(self):
        """Test get_base_path() - basic check for non-None and directory."""
        self.assertTrue(os.path.isdir(self.project_root), f"Base path '{self.project_root}' is not a directory.")
        # Expect 'src' or 'tests' to be under project_root for typical project structure
        self.assertTrue(os.path.isdir(os.path.join(self.project_root, "src")) or \
                        os.path.isdir(os.path.join(self.project_root, "tests")),
                        f"Base path {self.project_root} doesn't seem to be project root (missing src/ or tests/).")

    def test_load_config_success_from_root(self):
        config = load_config(self.test_config_filename) # Should find it at project root
        self.assertIsNotNone(config)
        self.assertEqual(config.get('wubu_name'), 'TestWuBu System')

    def test_load_config_success_from_src_fallback(self):
        # Remove from root, create in src/
        if os.path.exists(self.test_config_path_at_root): os.remove(self.test_config_path_at_root)
        os.makedirs(os.path.join(self.project_root, "src"), exist_ok=True)
        with open(self.test_config_path_in_src, 'w') as f:
            yaml.dump(self.dummy_config_data, f)

        config = load_config(self.test_config_filename) # Should find it in src/
        self.assertIsNotNone(config)
        self.assertEqual(config.get('wubu_name'), 'TestWuBu System')

    def test_load_config_file_not_found_anywhere(self):
        if os.path.exists(self.test_config_path_at_root): os.remove(self.test_config_path_at_root)
        if os.path.exists(self.test_config_path_in_src): os.remove(self.test_config_path_in_src)
        config = load_config("completely_non_existent_config_for_wubu.yaml")
        self.assertIsNone(config)

    def test_get_resource_path_within_wubu_package(self):
        """Test resolving resources within the src/wubu/ package."""
        # Sound file: src/wubu/sounds/effects/test_wubu_sound.wav
        # `get_resource_path` expects path relative to `src/wubu/`
        resolved_sound_path = get_resource_path(os.path.join("sounds", "effects"), "test_wubu_sound.wav")
        self.assertEqual(os.path.normpath(resolved_sound_path), os.path.normpath(self.abs_sound_file_path))
        self.assertTrue(os.path.exists(resolved_sound_path))

        # Image file: src/wubu/images/ui/test_wubu_image.png
        resolved_image_path = get_resource_path(os.path.join("images", "ui"), "test_wubu_image.png")
        self.assertEqual(os.path.normpath(resolved_image_path), os.path.normpath(self.abs_image_file_path))
        self.assertTrue(os.path.exists(resolved_image_path))

    def test_get_resource_path_fallback_to_root_assets_dir(self):
        """Test fallback to project_root/assets/ if not in wubu package."""
        # Asset file: project_root/assets/general_assets/subfolder/test_wubu_asset.txt
        # `get_resource_path` first looks in src/wubu/general_assets/subfolder/...
        # Since it won't be there, it should fall back to project_root/assets/general_assets/subfolder/...

        # Ensure it's NOT in the wubu package path for this test
        path_in_wubu_pkg_that_should_not_exist = os.path.join(self.wubu_package_dir_in_src, "general_assets", "subfolder", "test_wubu_asset.txt")
        self.assertFalse(os.path.exists(path_in_wubu_pkg_that_should_not_exist), "Test asset should not exist in wubu package for this fallback test.")

        resolved_asset_path = get_resource_path(os.path.join("general_assets", "subfolder"), "test_wubu_asset.txt")
        self.assertEqual(os.path.normpath(resolved_asset_path), os.path.normpath(self.abs_asset_file_path))
        self.assertTrue(os.path.exists(resolved_asset_path))

    def test_get_resource_path_non_existent_file_returns_primary_path(self):
        """Test that get_resource_path returns the primary expected path even if file doesn't exist."""
        non_existent_rel_path = os.path.join("non_existent_type", "non_existent_subdir")
        non_existent_filename = "no_such_wubu_file.xyz"

        expected_primary_path = os.path.join(self.wubu_package_dir_in_src, non_existent_rel_path, non_existent_filename)

        resolved_path = get_resource_path(non_existent_rel_path, non_existent_filename)
        self.assertEqual(os.path.normpath(resolved_path), os.path.normpath(expected_primary_path))
        self.assertFalse(os.path.exists(resolved_path))


if __name__ == '__main__':
    print("Running WuBu ResourceLoader tests...")
    # For robust testing from project root: python -m unittest tests.utils.test_resource_loader
    unittest.main()
