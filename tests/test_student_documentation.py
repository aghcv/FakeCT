"""Protect the offline guide's local resources, navigation and runnable entry points."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import unittest
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self,text):
        super().__init__();self.links=[];self.resources=[];self.ids=[];self.feed(text)
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if 'id' in attrs:self.ids.append(attrs['id'])
        if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
        if tag in ('script','img','iframe') and 'src' in attrs:self.resources.append(attrs['src'])
        if tag=='link' and 'href' in attrs:self.resources.append(attrs['href'])


class StudentDocumentationTests(unittest.TestCase):
    def test_offline_guides_have_valid_local_links_and_anchors(self):
        for relative in ('docs/student/index.html','docs/student/testing.html'):
            with self.subTest(guide=relative):
                path=ROOT/relative;document=Links(path.read_text())
                self.assertEqual(len(document.ids),len(set(document.ids)))
                for href in document.links+document.resources:
                    url=urlsplit(href)
                    if url.scheme or url.netloc:
                        self.assertNotIn(href,document.resources,'Guide cannot require a network resource')
                        continue
                    target=(path.parent/unquote(url.path)).resolve() if url.path else path
                    self.assertTrue(target.exists(),href)
                    if url.fragment and target.suffix=='.html':
                        self.assertIn(url.fragment,Links(target.read_text()).ids,href)

    def test_documented_student_clis_start_without_an_ml_backend(self):
        commands=(('student_demo.py','--smoke-test'),('prepare_cohort.py','--preview-mode'),
                  ('model_experiment.py','summarize'))
        for script,option in commands:
            with self.subTest(script=script):
                result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],
                                      text=True,capture_output=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(option,result.stdout)


if __name__=='__main__':unittest.main()
