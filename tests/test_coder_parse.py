"""The hybrid JSON-header + raw-tag coder response parser."""

import unittest

from coding_agent.roles import parse_coder_response


class CoderParseTest(unittest.TestCase):
    def test_pure_json_backward_compatible(self):
        a = parse_coder_response('{"action": "read_file", "path": "x.py"}')
        self.assertEqual(a.action, "read_file")
        self.assertEqual(a.path, "x.py")

    def test_write_file_with_xml_content_hard_chars(self):
        raw = (
            '{"rationale": "add", "action": "write_file", "path": "m.py"}\n'
            '<content>\n'
            'def f():\n'
            '    s = "a \\"q\\" {braces} \\\\ backslash"\n'
            '    return """triple\nquoted"""\n'
            '</content>'
        )
        a = parse_coder_response(raw)
        self.assertEqual(a.action, "write_file")
        self.assertEqual(a.path, "m.py")
        self.assertIn("{braces}", a.content)      # braces didn't break JSON scan
        self.assertIn('triple\nquoted', a.content)  # newline preserved
        self.assertTrue(a.content.startswith("def f():"))  # framing newline trimmed

    def test_patch_file_with_xml_bodies(self):
        raw = (
            '{"action": "patch_file", "path": "m.py"}\n'
            '<old_text>return -1</old_text><new_text>return mid</new_text>'
        )
        a = parse_coder_response(raw)
        self.assertEqual(a.action, "patch_file")
        self.assertEqual(a.old_text, "return -1")
        self.assertEqual(a.new_text, "return mid")

    def test_content_with_braces_does_not_confuse_header(self):
        # The header JSON must be recovered even though the body has braces.
        raw = '{"action":"write_file","path":"a.py"}\n<content>d = {"k": 1}\n</content>'
        a = parse_coder_response(raw)
        self.assertEqual(a.path, "a.py")
        self.assertIn('{"k": 1}', a.content)


if __name__ == "__main__":
    unittest.main()
