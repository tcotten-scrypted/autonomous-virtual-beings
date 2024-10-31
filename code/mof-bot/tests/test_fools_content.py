import unittest
from unittest.mock import patch, mock_open
import fools_content

class TestFoolsContent(unittest.TestCase):
    
    @patch("fools_content.open", new_callable=mock_open, read_data='{"@handle_1": ["content_1", "content_2", "content_3"], "@handle_2": ["content_1", "content_2", "content_3"]}')
    def test_load_available_content_success(self, mock_file):
        # Test that available_content loads correctly from the sample JSON
        fools_content.load_available_content()
        
        # Check that available_content matches the sample data structure
        expected_content = {
            "@handle_1": ["content_1", "content_2", "content_3"],
            "@handle_2": ["content_1", "content_2", "content_3"]
        }
        self.assertEqual(fools_content.available_content, expected_content)
        
    @patch("fools_content.open", new_callable=mock_open, read_data='{"@handle_1": ["content_1", "content_2", "content_3"], "@handle_2": ["content_1", "content_2", "content_3"]}')
    def test_summarize_correct_counts(self, mock_file):
        # Load the sample data and run summarize to set num_fools and num_posts_per_fool
        fools_content.load_available_content()
        fools_content.summarize()
        
        # Check that num_fools matches the number of top-level keys
        self.assertEqual(fools_content.num_fools, 2)
        
        # Check that num_posts_per_fool contains the correct counts of posts per handle
        self.assertEqual(fools_content.num_posts_per_fool, [3, 3])

    @patch("fools_content.open", side_effect=FileNotFoundError)
    def test_load_available_content_file_not_found(self, mock_file):
        # Test that available_content is empty when the file is not found
        fools_content.load_available_content()
        
        # Check that available_content is set to an empty dictionary
        self.assertEqual(fools_content.available_content, {})

    @patch("fools_content.open", new_callable=mock_open, read_data='{invalid_json}')
    def test_load_available_content_json_decode_error(self, mock_file):
        # Test that available_content is empty when JSON is invalid
        fools_content.load_available_content()
        
        # Check that available_content is set to an empty dictionary
        self.assertEqual(fools_content.available_content, {})

if __name__ == "__main__":
    unittest.main()
